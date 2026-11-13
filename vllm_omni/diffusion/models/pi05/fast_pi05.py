# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fast bfloat16 denoising loop for π0.5 (iscadev).

The action expert runs on ``chunk_size`` rows per request (10-50), so every
GEMM of a denoising step is a weight stream: the time of a step is set by how
fast 622 MB of expert weights (bf16) and the KV cache move through the GPU and
by the number of kernels launched, not by FLOPs. The kernels here are shaped
for that:

* ``_skinny_gemm_kernel``: a split-K GEMM over ``BLOCK_M = 16`` rows that
  writes float32 partial sums, so that enough CTAs stream the weight in
  parallel to reach a useful fraction of HBM bandwidth. The reduction over
  the splits is folded into the kernel that consumes the result (attention,
  gated residual + AdaRMS norm, output head). The gate/up GEMM runs unsplit
  with ``gelu(gate) · up`` as its epilogue.
* ``_attn_kernel`` / ``_attn_combine_kernel``: one flash-decoding style
  kernel for the single-KV-head attention, split over the prefix keys of the
  cache row. It reads Q and the suffix K/V straight from the QKV partials
  (applying RoPE itself), so the suffix K/V is never written to the cache.
* The Euler update ``x_t += dt · v_t`` is applied by the output-head kernel,
  so a whole schedule of steps runs with no host work in between and the
  loop is captured as one CUDA graph.

Eight kernels per layer, against the nine of the fused path, each shaped for
10-50 rows. Numerics: bfloat16 weights and activations, float32 residual stream,
accumulation and softmax, as the eager bfloat16 path; the reduction orders differ from it.
Every row is computed independently of the other rows of its launch, so a
request gets the same bits alone and in a formed batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

if HAS_TRITON:

    @triton.jit
    def _bf16(x):
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _gelu_tanh(x):
        x_cube = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * x_cube)
        return 0.5 * x * (1.0 + tldevice.tanh(inner))

    @triton.jit
    def _skinny_gemm_kernel(
        X,
        W,
        P,
        OUT,
        M,
        N,
        K,
        stride_xm,
        stride_wn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLITS: tl.constexpr,
        DUAL: tl.constexpr,
        GELU_OUT: tl.constexpr,
    ):
        """``P[s] = X[:, Ks] · W[:, Ks]ᵀ`` for K-split ``s``; with ``DUAL`` also for
        the second weight stacked under ``W`` (rows ``N..2N``), into ``P[s, :, N:]``.
        With ``GELU_OUT`` (``DUAL`` and one split) ``OUT = gelu(X W₁ᵀ) · (X W₂ᵀ)``
        in bf16 is stored instead of partials.

        ``X`` is ``(M, K)`` bf16 (rows ``stride_xm`` apart), ``W`` is ``(N | 2N, K)``
        bf16 row-major, ``P`` is ``(SPLITS, M, N | 2N)`` float32.
        """
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_m = tl.program_id(2)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rm[:, None] < M
        col_mask = rn[None, :] < N
        k_per = K // SPLITS
        k_base = pid_s * k_per
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if DUAL:
            acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, k_per, BLOCK_K):
            rk = k_base + kk + tl.arange(0, BLOCK_K)
            x = tl.load(X + rm[:, None] * stride_xm + rk[None, :], mask=row_mask, other=0.0)
            w = tl.load(W + rn[None, :] * stride_wn + rk[:, None], mask=col_mask, other=0.0)
            acc = tl.dot(x, w, acc)
            if DUAL:
                w2 = tl.load(W + (rn[None, :] + N) * stride_wn + rk[:, None], mask=col_mask, other=0.0)
                acc2 = tl.dot(x, w2, acc2)
        if GELU_OUT:
            out = _bf16(_gelu_tanh(_bf16(acc))) * _bf16(acc2)
            tl.store(OUT + rm[:, None] * N + rn[None, :], out.to(tl.bfloat16), mask=row_mask & col_mask)
        else:
            n_total = N
            if DUAL:
                n_total = 2 * N
            p = P + pid_s * M * n_total + rm[:, None] * n_total + rn[None, :]
            tl.store(p, acc, mask=row_mask & col_mask)
            if DUAL:
                tl.store(p + N, acc2, mask=row_mask & col_mask)

    @triton.jit
    def _entry_kernel(
        XT,
        W_IN,
        B_IN,
        MOD,
        RES_OUT,
        XN_OUT,
        eps,
        A: tl.constexpr,
        N: tl.constexpr,
    ):
        """Row ``r``: ``x = x_t · W_inᵀ + b`` in float32 (the action embedding; ``W_IN`` is ``W_inᵀ``, ``(A, N)``) into
        ``RES_OUT``, and its AdaRMS norm under ``MOD = (scale | shift | gate)`` into
        ``XN_OUT`` (bf16)."""
        row = tl.program_id(0)
        cols = tl.arange(0, N)
        acc = tl.load(B_IN + cols).to(tl.float32)
        for a in tl.static_range(A):
            x_a = tl.load(XT + row * A + a).to(tl.float32)
            acc += x_a * tl.load(W_IN + a * N + cols).to(tl.float32)
        tl.store(RES_OUT + row * N + cols, acc)
        var = tl.sum(acc * acc, axis=0) / N
        normed = acc * tldevice.rsqrt(var + eps)
        scale = tl.load(MOD + cols).to(tl.float32)
        shift = tl.load(MOD + N + cols).to(tl.float32)
        tl.store(XN_OUT + row * N + cols, (normed * (1.0 + scale) + shift).to(tl.bfloat16))

    @triton.jit
    def _attn_kernel(
        P,
        COS,
        SIN,
        K,
        V,
        MASK,
        PART_O,
        PART_M,
        PART_L,
        M,
        q_len,
        prefix_len,
        keys_per_split,
        kv_row_stride,
        kv_batch_stride,
        mask_row_stride,
        mask_batch_stride,
        scale,
        SPLITS: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_S: tl.constexpr,
        KV_SPLITS: tl.constexpr,
    ):
        """Program ``(b, r, s)``: rows ``r`` (over heads × queries of request ``b``)
        against prefix-key split ``s`` of its cache row; the last split also
        attends the suffix keys. Q and the suffix K/V are summed from the QKV
        partials ``P`` (``(SPLITS, M, (heads + 2) · HEAD_DIM)``) and rotated here.
        Online softmax in float32 with the additive float mask ``MASK[b, query,
        key]``. Writes the split's unnormalized output, running max and sum."""
        b = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_s = tl.program_id(2)
        HALF: tl.constexpr = HEAD_DIM // 2
        N_TOTAL: tl.constexpr = (NUM_HEADS + 2) * HEAD_DIM
        n_rows = NUM_HEADS * q_len
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        row_mask = rows < n_rows
        head = rows // q_len
        t = rows % q_len
        prow = b * q_len + t
        dh = tl.arange(0, HALF)

        # Q of these rows: partial sums, bf16 rounding of the projection, RoPE.
        q_lo = tl.zeros((BLOCK_R, HALF), dtype=tl.float32)
        q_hi = tl.zeros((BLOCK_R, HALF), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            base = P + s * M * N_TOTAL + prow[:, None] * N_TOTAL + head[:, None] * HEAD_DIM + dh[None, :]
            q_lo += tl.load(base, mask=row_mask[:, None], other=0.0)
            q_hi += tl.load(base + HALF, mask=row_mask[:, None], other=0.0)
        q_lo = _bf16(q_lo)
        q_hi = _bf16(q_hi)
        rope = prow[:, None] * HEAD_DIM + dh[None, :]
        cos_lo = tl.load(COS + rope, mask=row_mask[:, None], other=0.0).to(tl.float32)
        cos_hi = tl.load(COS + rope + HALF, mask=row_mask[:, None], other=0.0).to(tl.float32)
        sin_lo = tl.load(SIN + rope, mask=row_mask[:, None], other=0.0).to(tl.float32)
        sin_hi = tl.load(SIN + rope + HALF, mask=row_mask[:, None], other=0.0).to(tl.float32)
        qr_lo = (q_lo * cos_lo - q_hi * sin_lo).to(tl.bfloat16)
        qr_hi = (q_hi * cos_hi + q_lo * sin_hi).to(tl.bfloat16)

        m_i = tl.full((BLOCK_R,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_R,), dtype=tl.float32)
        acc_lo = tl.zeros((BLOCK_R, HALF), dtype=tl.float32)
        acc_hi = tl.zeros((BLOCK_R, HALF), dtype=tl.float32)
        k_base = K + b * kv_batch_stride
        v_base = V + b * kv_batch_stride
        mask_base = MASK + b * mask_batch_stride + t[:, None] * mask_row_stride

        start = pid_s * keys_per_split
        end = tl.minimum(start + keys_per_split, prefix_len)
        for kb in range(start, end, BLOCK_N):
            kidx = kb + tl.arange(0, BLOCK_N)
            kmask = kidx < end
            k_lo = tl.load(k_base + kidx[None, :] * kv_row_stride + dh[:, None], mask=kmask[None, :], other=0.0)
            k_hi = tl.load(k_base + kidx[None, :] * kv_row_stride + HALF + dh[:, None], mask=kmask[None, :], other=0.0)
            sc = (tl.dot(qr_lo, k_lo) + tl.dot(qr_hi, k_hi)) * scale
            bias = tl.load(mask_base + kidx[None, :], mask=row_mask[:, None] & kmask[None, :], other=0.0)
            sc = tl.where(kmask[None, :], sc + bias.to(tl.float32), float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tldevice.exp(m_i - m_safe)
            p = tldevice.exp(sc - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            p16 = p.to(tl.bfloat16)
            v_lo = tl.load(v_base + kidx[:, None] * kv_row_stride + dh[None, :], mask=kmask[:, None], other=0.0)
            v_hi = tl.load(v_base + kidx[:, None] * kv_row_stride + HALF + dh[None, :], mask=kmask[:, None], other=0.0)
            acc_lo = acc_lo * alpha[:, None] + tl.dot(p16, v_lo)
            acc_hi = acc_hi * alpha[:, None] + tl.dot(p16, v_hi)
            m_i = m_new

        if pid_s == KV_SPLITS - 1:
            # The suffix keys: this request's K/V rows from the partials, rotated.
            ts = tl.arange(0, BLOCK_S)
            smask = ts < q_len
            srow = b * q_len + ts
            k_lo_s = tl.zeros((BLOCK_S, HALF), dtype=tl.float32)
            k_hi_s = tl.zeros((BLOCK_S, HALF), dtype=tl.float32)
            v_lo_s = tl.zeros((BLOCK_S, HALF), dtype=tl.float32)
            v_hi_s = tl.zeros((BLOCK_S, HALF), dtype=tl.float32)
            for s in tl.static_range(SPLITS):
                base_s = P + s * M * N_TOTAL + srow[:, None] * N_TOTAL + NUM_HEADS * HEAD_DIM + dh[None, :]
                k_lo_s += tl.load(base_s, mask=smask[:, None], other=0.0)
                k_hi_s += tl.load(base_s + HALF, mask=smask[:, None], other=0.0)
                v_lo_s += tl.load(base_s + HEAD_DIM, mask=smask[:, None], other=0.0)
                v_hi_s += tl.load(base_s + HEAD_DIM + HALF, mask=smask[:, None], other=0.0)
            k_lo_s = _bf16(k_lo_s)
            k_hi_s = _bf16(k_hi_s)
            srope = srow[:, None] * HEAD_DIM + dh[None, :]
            c_lo = tl.load(COS + srope, mask=smask[:, None], other=0.0).to(tl.float32)
            c_hi = tl.load(COS + srope + HALF, mask=smask[:, None], other=0.0).to(tl.float32)
            s_lo = tl.load(SIN + srope, mask=smask[:, None], other=0.0).to(tl.float32)
            s_hi = tl.load(SIN + srope + HALF, mask=smask[:, None], other=0.0).to(tl.float32)
            kr_lo = (k_lo_s * c_lo - k_hi_s * s_lo).to(tl.bfloat16)
            kr_hi = (k_hi_s * c_hi + k_lo_s * s_hi).to(tl.bfloat16)
            sc_s = (tl.dot(qr_lo, tl.trans(kr_lo)) + tl.dot(qr_hi, tl.trans(kr_hi))) * scale
            bias_s = tl.load(
                mask_base + prefix_len + ts[None, :], mask=row_mask[:, None] & smask[None, :], other=0.0
            )
            sc_s = tl.where(smask[None, :], sc_s + bias_s.to(tl.float32), float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc_s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tldevice.exp(m_i - m_safe)
            p_s = tldevice.exp(sc_s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p_s, axis=1)
            p16_s = p_s.to(tl.bfloat16)
            acc_lo = acc_lo * alpha[:, None] + tl.dot(p16_s, v_lo_s.to(tl.bfloat16))
            acc_hi = acc_hi * alpha[:, None] + tl.dot(p16_s, v_hi_s.to(tl.bfloat16))
            m_i = m_new

        part = (b * KV_SPLITS + pid_s) * n_rows + rows
        tl.store(PART_O + part[:, None] * HEAD_DIM + dh[None, :], acc_lo, mask=row_mask[:, None])
        tl.store(PART_O + part[:, None] * HEAD_DIM + HALF + dh[None, :], acc_hi, mask=row_mask[:, None])
        tl.store(PART_M + part, m_i, mask=row_mask)
        tl.store(PART_L + part, l_i, mask=row_mask)

    @triton.jit
    def _attn_combine_kernel(
        PART_O,
        PART_M,
        PART_L,
        OUT,
        q_len,
        out_stride,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KV_SPLITS: tl.constexpr,
    ):
        """Row ``(b, head, t)``: merge the key splits and store ``(b · q_len + t, head · HEAD_DIM)`` bf16."""
        b = tl.program_id(0)
        r = tl.program_id(1)
        n_rows = NUM_HEADS * q_len
        d = tl.arange(0, HEAD_DIM)
        m = tl.full((), float("-inf"), dtype=tl.float32)
        for s in tl.static_range(KV_SPLITS):
            m = tl.maximum(m, tl.load(PART_M + (b * KV_SPLITS + s) * n_rows + r))
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        l = tl.zeros((), dtype=tl.float32)
        for s in tl.static_range(KV_SPLITS):
            part = (b * KV_SPLITS + s) * n_rows + r
            w = tldevice.exp(tl.load(PART_M + part) - m)
            l += w * tl.load(PART_L + part)
            acc += w * tl.load(PART_O + part * HEAD_DIM + d)
        head = r // q_len
        t = r % q_len
        out = acc / l
        tl.store(OUT + (b * q_len + t) * out_stride + head * HEAD_DIM + d, out.to(tl.bfloat16))

    @triton.jit
    def _resid_norm_kernel(
        P,
        RES,
        GATE,
        MOD,
        XN_OUT,
        M,
        eps,
        SPLITS: tl.constexpr,
        N: tl.constexpr,
    ):
        """Row ``r``: ``RES[r] += bf16(gate · bf16(Σ P[s, r]))`` (float32, in place), then its
        AdaRMS norm under ``MOD`` into ``XN_OUT`` (bf16). ``GATE`` is the gate
        third of the modulation that produced this sublayer's input norm."""
        row = tl.program_id(0)
        cols = tl.arange(0, N)
        acc = tl.zeros((N,), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            acc += tl.load(P + (s * M + row) * N + cols)
        gate = tl.load(GATE + cols).to(tl.float32)
        res = tl.load(RES + row * N + cols).to(tl.float32)
        h = res + _bf16(gate * _bf16(acc))
        tl.store(RES + row * N + cols, h)
        var = tl.sum(h * h, axis=0) / N
        normed = h * tldevice.rsqrt(var + eps)
        scale = tl.load(MOD + cols).to(tl.float32)
        shift = tl.load(MOD + N + cols).to(tl.float32)
        tl.store(XN_OUT + row * N + cols, (normed * (1.0 + scale) + shift).to(tl.bfloat16))

    @triton.jit
    def _gelu_mul_kernel(
        P,
        OUT,
        M,
        SPLITS: tl.constexpr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``OUT[r, c] = gelu_tanh(Σ P[s, r, c]) · Σ P[s, r, N + c]`` (bf16)."""
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        g = tl.zeros((BLOCK,), dtype=tl.float32)
        u = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            base = P + (s * M + row) * (2 * N)
            g += tl.load(base + cols)
            u += tl.load(base + N + cols)
        out = _bf16(_gelu_tanh(_bf16(g))) * _bf16(u)
        tl.store(OUT + row * N + cols, out.to(tl.bfloat16))

    @triton.jit
    def _final_kernel(
        P,
        RES,
        GATE,
        MOD,
        W_OUT,
        B_OUT,
        XT,
        M,
        eps,
        dt,
        SPLITS: tl.constexpr,
        N: tl.constexpr,
        A: tl.constexpr,
    ):
        """Row ``r``: the last gated residual, the final AdaRMS norm, ``action_out_proj``
        and the Euler update ``XT[r] += dt · v`` in place (float32)."""
        row = tl.program_id(0)
        cols = tl.arange(0, N)
        acc = tl.zeros((N,), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            acc += tl.load(P + (s * M + row) * N + cols)
        gate = tl.load(GATE + cols).to(tl.float32)
        res = tl.load(RES + row * N + cols).to(tl.float32)
        h = res + _bf16(gate * _bf16(acc))
        var = tl.sum(h * h, axis=0) / N
        normed = h * tldevice.rsqrt(var + eps)
        scale = tl.load(MOD + cols).to(tl.float32)
        shift = tl.load(MOD + N + cols).to(tl.float32)
        y = normed * (1.0 + scale) + shift
        outs = tl.arange(0, A)
        w = tl.load(W_OUT + outs[:, None] * N + cols[None, :]).to(tl.float32)
        v = tl.sum(y[None, :] * w, axis=1) + tl.load(B_OUT + outs).to(tl.float32)
        x = tl.load(XT + row * A + outs)
        tl.store(XT + row * A + outs, x + dt * v)


@dataclass(frozen=True)
class GemmTiles:
    block_n: int
    block_k: int
    splits: int
    num_warps: int = 4
    num_stages: int = 3


@dataclass(frozen=True)
class AttnTiles:
    block_r: int = 16
    block_n: int = 64
    kv_splits: int = 8
    num_warps: int = 4
    num_stages: int = 2


# Tiles per GEMM, from an in-situ sweep on an A100 at 10 rows (results/iscadev/tune_gemm.py).
DEFAULT_TILES: dict[str, GemmTiles] = {
    "qkv": GemmTiles(64, 128, 2, 4, 4),
    "o": GemmTiles(64, 64, 4, 4, 4),
    "gateup": GemmTiles(64, 128, 1, 4, 3),
    "down": GemmTiles(64, 128, 4, 4, 4),
}
DEFAULT_ATTN = AttnTiles()
BLOCK_M = 16


class FastDenoiser:
    """The denoising loop of one ``Pi05ForActionPrediction`` on the kernels above.

    Built after the weights are loaded and the AdaRMS schedule is prepared;
    serves batches up to ``max_batch`` over scratch buffers of its own. The
    Q/K/V and gate/up weights of every layer are concatenated once (the
    ``nn.Linear`` modules keep views into the concatenation, so the eager path
    reads the same storage).
    """

    def __init__(
        self,
        model,
        max_batch: int = 1,
        tiles: dict[str, GemmTiles] | None = None,
        attn: AttnTiles | None = None,
    ):
        if not HAS_TRITON:
            raise RuntimeError("The π0.5 fast denoising loop needs Triton.")
        expert = model.paligemma_with_expert.gemma_expert.model
        layer0 = expert.layers[0]
        self.model = model
        self.tiles = dict(DEFAULT_TILES, **(tiles or {}))
        self.attn = attn or DEFAULT_ATTN
        self.width = layer0.self_attn.q_proj.in_features
        self.num_heads = layer0.self_attn.config.num_attention_heads
        self.head_dim = layer0.self_attn.head_dim
        self.mlp_dim = layer0.mlp.gate_proj.out_features
        self.eps = layer0.input_layernorm.eps
        self.horizon = model.action_horizon
        self.action_dim = model.action_dim
        self.num_layers = len(expert.layers)
        self.max_batch = max_batch
        self.dtype = layer0.self_attn.q_proj.weight.dtype
        if self.dtype != torch.bfloat16:
            raise RuntimeError(f"The fast denoising loop runs bfloat16 experts, not {self.dtype}.")
        if layer0.self_attn.config.num_key_value_heads != 1:
            raise RuntimeError("The fast denoising loop needs a single KV head.")
        if self.action_dim & (self.action_dim - 1):
            raise RuntimeError(f"The fast denoising loop needs a power-of-two action_dim, got {self.action_dim}.")
        device = layer0.self_attn.q_proj.weight.device
        self.device = device
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Concatenated weights; the Linear modules become views into them.
        self.w_qkv, self.w_o, self.w_gu, self.w_down = [], [], [], []
        with torch.no_grad():
            for layer in expert.layers:
                attn_mod, mlp = layer.self_attn, layer.mlp
                w_qkv = torch.cat([attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight]).contiguous()
                n_q = attn_mod.q_proj.weight.shape[0]
                n_kv = attn_mod.k_proj.weight.shape[0]
                attn_mod.q_proj.weight = nn.Parameter(w_qkv[:n_q], requires_grad=False)
                attn_mod.k_proj.weight = nn.Parameter(w_qkv[n_q : n_q + n_kv], requires_grad=False)
                attn_mod.v_proj.weight = nn.Parameter(w_qkv[n_q + n_kv :], requires_grad=False)
                w_gu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]).contiguous()
                mlp.gate_proj.weight = nn.Parameter(w_gu[: self.mlp_dim], requires_grad=False)
                mlp.up_proj.weight = nn.Parameter(w_gu[self.mlp_dim :], requires_grad=False)
                self.w_qkv.append(w_qkv)
                self.w_o.append(attn_mod.o_proj.weight)
                self.w_gu.append(w_gu)
                self.w_down.append(mlp.down_proj.weight)
        self.w_in_t = model.action_in_proj.weight.detach().t().contiguous()
        self.b_in = model.action_in_proj.bias
        self.w_out = model.action_out_proj.weight
        self.b_out = model.action_out_proj.bias
        if model.action_in_proj.bias is None or self.b_out is None:
            raise RuntimeError("The fast denoising loop expects biased action projections.")

        # The AdaRMS schedule: (steps, 2 · layers + 1, 3 · width).
        rows = getattr(model, "_runtime_steps", None)
        if not rows:
            raise RuntimeError("Prepare the AdaRMS schedule (pi05_precompute_adarms) before building the fast loop.")
        self.num_steps = len(rows)
        self.modulations = torch.stack([row.modulations[:, 0] for row in rows]).contiguous()
        assert self.modulations.shape == (self.num_steps, 2 * self.num_layers + 1, 3 * self.width)

        # Scratch for the largest batch.
        m = max_batch * self.horizon
        w, nh, hd = self.width, self.num_heads, self.head_dim
        n_qkv = (nh + 2) * hd
        f32, bf16 = torch.float32, torch.bfloat16
        t = self.tiles
        parts = max_batch * self.attn.kv_splits * nh * self.horizon
        self.res = torch.empty(m, w, dtype=f32, device=device)
        self.xn = torch.empty(m, w, dtype=bf16, device=device)
        self.p_qkv = torch.empty(t["qkv"].splits, m, n_qkv, dtype=f32, device=device)
        self.part_o = torch.empty(parts, hd, dtype=f32, device=device)
        self.part_m = torch.empty(parts, dtype=f32, device=device)
        self.part_l = torch.empty_like(self.part_m)
        self.att = torch.empty(m, nh * hd, dtype=bf16, device=device)
        self.p_o = torch.empty(t["o"].splits, m, w, dtype=f32, device=device)
        self.p_gu = torch.empty(t["gateup"].splits, m, 2 * self.mlp_dim, dtype=f32, device=device)
        self.hid = torch.empty(m, self.mlp_dim, dtype=bf16, device=device)
        self.p_down = torch.empty(t["down"].splits, m, w, dtype=f32, device=device)

    # ── launches ─────────────────────────────────────────────────────
    def _gemm(self, x, w, p, name: str, dual: bool = False, out=None) -> None:
        tiles = self.tiles[name]
        m, k = x.shape
        n = w.shape[0] // (2 if dual else 1)
        gelu_out = out is not None
        if gelu_out and (not dual or tiles.splits != 1):
            raise ValueError("The GELU epilogue needs the dual gate/up GEMM with one split.")
        grid = (triton.cdiv(n, tiles.block_n), tiles.splits, triton.cdiv(m, BLOCK_M))
        _skinny_gemm_kernel[grid](
            x,
            w,
            p,
            out if gelu_out else p,
            m,
            n,
            k,
            x.stride(0),
            w.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_N=tiles.block_n,
            BLOCK_K=tiles.block_k,
            SPLITS=tiles.splits,
            DUAL=dual,
            GELU_OUT=gelu_out,
            num_warps=tiles.num_warps,
            num_stages=tiles.num_stages,
        )

    def run(self, x_t: torch.Tensor, kv_cache, context, num_steps: int | None = None) -> torch.Tensor:
        """Denoise ``x_t`` (``(B, horizon, action_dim)`` float32, updated in place)
        over the prepared schedule. ``context`` holds the batch's additive
        attention mask ``(B, 1, horizon, n_keys)`` and RoPE tables ``(B, horizon,
        head_dim)``. Returns ``x_t``."""
        if num_steps is None:
            num_steps = self.num_steps
        if num_steps != self.num_steps:
            raise ValueError(f"The fast loop runs the prepared {self.num_steps}-step schedule, not {num_steps}.")
        batch, horizon, action_dim = x_t.shape
        if horizon != self.horizon or action_dim != self.action_dim or batch > self.max_batch:
            raise ValueError("x_t does not fit the fast denoising loop's buffers.")
        if not x_t.is_contiguous() or x_t.dtype != torch.float32:
            raise ValueError("x_t must be a contiguous float32 tensor.")
        m = batch * horizon
        w, nh, hd = self.width, self.num_heads, self.head_dim
        prefix_len, n_keys = kv_cache.prefix_len, kv_cache.prefix_len + kv_cache.suffix_len
        key, value = kv_cache.key, kv_cache.value  # (layers, B, 1, n_keys, hd)
        if key.shape[1] < batch or key.shape[2] != 1 or key.shape[3] != n_keys or kv_cache.suffix_len != horizon:
            raise ValueError("The KV cache does not match the batch.")
        mask = context.attention_mask  # (B, 1, horizon, n_keys) float32
        if mask.shape != (batch, 1, horizon, n_keys) or mask.dtype != torch.float32:
            raise ValueError(f"Unexpected attention mask {tuple(mask.shape)} {mask.dtype}.")
        mask = mask[:, 0]
        cos, sin = context.cos.reshape(m, hd), context.sin.reshape(m, hd)
        if cos.dtype != torch.bfloat16:
            cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        res, xn, att, hid = self.res[:m], self.xn[:m], self.att[:m], self.hid[:m]
        p_qkv, p_o, p_gu, p_down = self.p_qkv[:, :m], self.p_o[:, :m], self.p_gu[:, :m], self.p_down[:, :m]
        n_rows = nh * horizon
        a = self.attn
        keys_per_split = triton.cdiv(triton.cdiv(prefix_len, a.kv_splits), a.block_n) * a.block_n
        t = self.tiles
        dt = -1.0 / num_steps
        mods = self.modulations
        gelu_fused = t["gateup"].splits == 1

        for step in range(num_steps):
            mod = mods[step]
            _entry_kernel[(m,)](x_t, self.w_in_t, self.b_in, mod[0], res, xn, self.eps, A=action_dim, N=w, num_warps=4)
            for layer in range(self.num_layers):
                k_l, v_l = key[layer, :, 0], value[layer, :, 0]  # (B, n_keys, hd)
                mod_in, mod_post = mod[2 * layer], mod[2 * layer + 1]
                self._gemm(xn, self.w_qkv[layer], p_qkv, "qkv")
                _attn_kernel[(batch, triton.cdiv(n_rows, a.block_r), a.kv_splits)](
                    p_qkv,
                    cos,
                    sin,
                    k_l,
                    v_l,
                    mask,
                    self.part_o,
                    self.part_m,
                    self.part_l,
                    m,
                    horizon,
                    prefix_len,
                    keys_per_split,
                    k_l.stride(1),
                    k_l.stride(0),
                    mask.stride(1),
                    mask.stride(0),
                    self.scale,
                    SPLITS=t["qkv"].splits,
                    NUM_HEADS=nh,
                    HEAD_DIM=hd,
                    BLOCK_R=a.block_r,
                    BLOCK_N=a.block_n,
                    BLOCK_S=max(16, triton.next_power_of_2(horizon)),
                    KV_SPLITS=a.kv_splits,
                    num_warps=a.num_warps,
                    num_stages=a.num_stages,
                )
                _attn_combine_kernel[(batch, n_rows)](
                    self.part_o,
                    self.part_m,
                    self.part_l,
                    att,
                    horizon,
                    att.stride(0),
                    NUM_HEADS=nh,
                    HEAD_DIM=hd,
                    KV_SPLITS=a.kv_splits,
                    num_warps=4,
                )
                self._gemm(att, self.w_o[layer], p_o, "o")
                _resid_norm_kernel[(m,)](
                    p_o, res, mod_in[2 * w :], mod_post, xn, m, self.eps, SPLITS=t["o"].splits, N=w, num_warps=4
                )
                if gelu_fused:
                    self._gemm(xn, self.w_gu[layer], p_gu, "gateup", dual=True, out=hid)
                else:
                    self._gemm(xn, self.w_gu[layer], p_gu, "gateup", dual=True)
                    _gelu_mul_kernel[(m, triton.cdiv(self.mlp_dim, 1024))](
                        p_gu, hid, m, SPLITS=t["gateup"].splits, N=self.mlp_dim, BLOCK=1024, num_warps=4
                    )
                self._gemm(hid, self.w_down[layer], p_down, "down")
                if layer + 1 < self.num_layers:
                    _resid_norm_kernel[(m,)](
                        p_down,
                        res,
                        mod_post[2 * w :],
                        mod[2 * layer + 2],
                        xn,
                        m,
                        self.eps,
                        SPLITS=t["down"].splits,
                        N=w,
                        num_warps=4,
                    )
                else:
                    _final_kernel[(m,)](
                        p_down,
                        res,
                        mod_post[2 * w :],
                        mod[-1],
                        self.w_out,
                        self.b_out,
                        x_t,
                        m,
                        self.eps,
                        dt,
                        SPLITS=t["down"].splits,
                        N=w,
                        A=action_dim,
                        num_warps=8,
                    )
        return x_t
