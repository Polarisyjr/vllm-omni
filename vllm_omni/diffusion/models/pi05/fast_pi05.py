# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""The π0.5 fast path (``pi05_fast_path``): bfloat16 kernels for short action chunks.

Three drop-in replacements for parts of ``sample_actions``, installed by
``Pi05Pipeline`` and captured by ``Pi05CUDAGraphs`` (one graph per request):

* ``FastDenoiser``: the whole denoising loop of the action expert.
* ``FastPrefix``: the prefix forward of the PaliGemma language model.
* ``FastVision``: the SigLIP encoder.

**Denoising loop.** The action expert runs on ``chunk_size`` rows per request
(10-50), so for one request every GEMM of a denoising step is a weight stream:
the time of a step is set by how fast 622 MB of expert weights (bf16) and the
KV cache move through the GPU and by the number of kernels launched, not by
FLOPs. The kernels are shaped for that, and their tiles widen with the launch
as formed batches turn the GEMMs into compute (``LOOP_TILES``):

* ``_skinny_gemm_kernel``: a split-K GEMM over a tile of rows that writes
  float32 partial sums, so that enough CTAs stream the weight in parallel to
  reach a useful fraction of HBM bandwidth. The reduction over the splits is
  folded into the kernel that consumes the result (attention, gated residual
  + AdaRMS norm, output head). The gate/up GEMM runs unsplit with
  ``gelu(gate) · up`` as its epilogue.
* ``_attn_kernel`` / ``_attn_combine_kernel``: one flash-decoding style
  kernel for the single-KV-head attention, split over the prefix keys of the
  cache row. It reads Q and the suffix K/V straight from the QKV partials
  (applying RoPE itself), so the suffix K/V is never written to the cache.
* The Euler update ``x_t += dt · v_t`` is applied by the output-head kernel,
  so a whole schedule of steps runs with no host work in between and the
  loop is captured as one CUDA graph.

Eight kernels per layer, against the nine of the fused path. Numerics:
bfloat16 weights and activations, float32 residual stream, accumulation and
softmax, as the eager bfloat16 path; the reduction orders differ from it.
Every row is computed independently of the other rows of its launch. The
tiles follow the size of the launch (``LOOP_TILES``, 1 to 128 requests of a
10-token chunk), so the loop serves any formed batch; a request's bits depend
on the size of the launch it is in, never on the other requests.

**Prefix forward and vision tower.** Compute-bound GEMM chains at 512-612 rows,
where cuBLAS is the fastest GEMM; the kernels fuse what surrounds the GEMMs
(see the second half of this module).

Tested against the kernels it replaces in
``tests/diffusion/models/pi05/test_pi05_fast_path.py``.
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
            bias_s = tl.load(mask_base + prefix_len + ts[None, :], mask=row_mask[:, None] & smask[None, :], other=0.0)
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
        total = tl.zeros((), dtype=tl.float32)
        for s in tl.static_range(KV_SPLITS):
            part = (b * KV_SPLITS + s) * n_rows + r
            w = tldevice.exp(tl.load(PART_M + part) - m)
            total += w * tl.load(PART_L + part)
            acc += w * tl.load(PART_O + part * HEAD_DIM + d)
        head = r // q_len
        t = r % q_len
        out = acc / total
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
    """Tiles of one split-K GEMM: output columns and K per program, K splits."""

    block_n: int
    block_k: int
    splits: int
    num_warps: int = 4
    num_stages: int = 3


@dataclass(frozen=True)
class AttnTiles:
    """Tiles of the denoising attention.

    ``kv_splits`` programs share the keys of a ``block_r`` tile of query rows.
    A split takes a whole number of ``block_n`` key blocks, so the prefix keys
    fill the first ``ceil(prefix_len / keys_per_split)`` splits and the last
    split always adds the suffix keys (at a 612-key prefix and 6 splits: 5
    prefix splits and one with the suffix keys alone).
    """

    block_r: int = 16
    block_n: int = 64
    kv_splits: int = 6
    num_warps: int = 4
    num_stages: int = 2


@dataclass(frozen=True)
class LoopTiles:
    """The tiles of every kernel of the denoising loop at one launch size."""

    block_m: int  # rows per GEMM program
    qkv: GemmTiles
    o: GemmTiles
    gateup: GemmTiles  # one split fuses the GELU epilogue into the GEMM
    down: GemmTiles
    attn: AttnTiles


def _loop_tiles(block_m, qkv, o, gateup, down, attn) -> LoopTiles:
    return LoopTiles(block_m, GemmTiles(*qkv), GemmTiles(*o), GemmTiles(*gateup), GemmTiles(*down), AttnTiles(*attn))


# Tiles by launch size, as (rows from, tiles): a launch over ``rows`` action
# tokens (requests · chunk_size) runs the last entry that starts at or below
# it. Swept in the loop itself on an A100 at its default clocks, at 10 to 1280
# rows of 10-token chunks, i.e. 1 to 128 requests
# (benchmarks/pi05/tune_fast_loop_pi05.py); each entry starts halfway down to
# the size before it. Small launches are weight streams: narrow row tiles, the
# projections split over K, the keys split over programs. Large ones are
# compute: wide row tiles, one split.
# Fixed rather than autotuned, so a restart cannot pick other tiles. The
# splits set the order of the reductions, so a request's bits depend on the
# entry its launch falls in (not on the other requests of the launch).
# fmt: off
LOOP_TILES: tuple[tuple[int, LoopTiles], ...] = (
    # (rows from, _loop_tiles(block_m, qkv, o, gate/up, down,
    #                         attention)),
    (1, _loop_tiles(16, (64, 128, 1, 4, 4), (32, 128, 2, 4, 4), (64, 128, 1, 8, 3), (32, 128, 2, 4, 4),
                    (16, 64, 6, 4, 2))),
    (16, _loop_tiles(32, (32, 128, 1, 4, 4), (32, 128, 2, 4, 4), (64, 128, 1, 8, 4), (32, 128, 2, 4, 4),
                    (16, 64, 6, 4, 2))),
    (31, _loop_tiles(64, (32, 128, 1, 4, 4), (32, 128, 2, 4, 4), (32, 128, 1, 4, 3), (32, 128, 2, 4, 4),
                    (16, 64, 4, 4, 2))),
    (61, _loop_tiles(32, (128, 128, 1, 4, 4), (32, 128, 2, 4, 4), (64, 128, 1, 4, 3), (32, 128, 2, 4, 4),
                    (16, 64, 4, 4, 2))),
    (121, _loop_tiles(64, (32, 64, 1, 4, 4), (64, 64, 2, 4, 4), (32, 64, 1, 4, 3), (64, 64, 2, 4, 4),
                    (16, 64, 2, 4, 2))),
    (241, _loop_tiles(64, (64, 64, 1, 4, 4), (64, 64, 2, 4, 4), (64, 64, 1, 4, 3), (64, 64, 2, 4, 4),
                    (16, 64, 1, 4, 2))),
    (481, _loop_tiles(128, (64, 64, 1, 4, 4), (64, 64, 1, 4, 4), (32, 64, 1, 4, 3), (64, 64, 1, 4, 4),
                    (16, 64, 2, 4, 2))),
    (961, _loop_tiles(128, (64, 64, 1, 4, 4), (64, 64, 1, 4, 4), (32, 64, 1, 4, 3), (64, 64, 1, 4, 4),
                    (16, 64, 1, 4, 2))),
)
# fmt: on


# The attention tiles above belong to short chunks: the table was swept at 10
# tokens. On it a 50-token chunk runs the loop at 0.96x to 1.05x of the fused
# per-step kernels (1 to 8 requests), so ``Pi05Pipeline`` gives the fast loop
# chunks up to this length only and longer ones keep the per-step kernels
# behind the fast prefix forward and vision tower.
LOOP_MAX_CHUNK = 16


def loop_tiles(rows: int) -> LoopTiles:
    """The tiles of a launch over ``rows`` action tokens (requests · chunk_size)."""
    tiles = LOOP_TILES[0][1]
    for rows_from, entry in LOOP_TILES:
        if rows >= rows_from:
            tiles = entry
    return tiles


class FastDenoiser:
    """The denoising loop of one ``Pi05ForActionPrediction`` on the kernels above.

    Built after the weights are loaded and the AdaRMS schedule is prepared;
    serves batches up to ``max_batch`` over scratch buffers of its own, each
    launch on the tiles of its size (``loop_tiles``), or every launch on
    ``tiles`` when given (the tile sweep, which may then assign other
    ``tiles`` that split no further). The Q/K/V and gate/up weights of every
    layer are concatenated once (the ``nn.Linear`` modules keep views into the
    concatenation, so the eager path reads the same storage).
    """

    def __init__(self, model, max_batch: int = 1, tiles: LoopTiles | None = None):
        if not HAS_TRITON:
            raise RuntimeError("The π0.5 fast denoising loop needs Triton.")
        expert = model.paligemma_with_expert.gemma_expert.model
        layer0 = expert.layers[0]
        self.model = model
        self.tiles = tiles
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

        # Scratch for the largest batch and the most splits any launch uses. A
        # launch packs its ``(splits, rows, columns)`` partials at the start
        # of a buffer, whatever the buffer's own shape.
        m = max_batch * self.horizon
        w, nh, hd = self.width, self.num_heads, self.head_dim
        n_qkv = (nh + 2) * hd
        f32, bf16 = torch.float32, torch.bfloat16
        every = [tiles] if tiles is not None else [entry for _, entry in LOOP_TILES]
        parts = max_batch * max(t.attn.kv_splits for t in every) * nh * self.horizon
        self.res = torch.empty(m, w, dtype=f32, device=device)
        self.xn = torch.empty(m, w, dtype=bf16, device=device)
        self.p_qkv = torch.empty(max(t.qkv.splits for t in every), m, n_qkv, dtype=f32, device=device)
        self.part_o = torch.empty(parts, hd, dtype=f32, device=device)
        self.part_m = torch.empty(parts, dtype=f32, device=device)
        self.part_l = torch.empty_like(self.part_m)
        self.att = torch.empty(m, nh * hd, dtype=bf16, device=device)
        self.p_o = torch.empty(max(t.o.splits for t in every), m, w, dtype=f32, device=device)
        self.p_gu = torch.empty(max(t.gateup.splits for t in every), m, 2 * self.mlp_dim, dtype=f32, device=device)
        self.hid = torch.empty(m, self.mlp_dim, dtype=bf16, device=device)
        self.p_down = torch.empty(max(t.down.splits for t in every), m, w, dtype=f32, device=device)

    # ── launches ─────────────────────────────────────────────────────
    def _gemm(self, x, w, p, tiles: GemmTiles, block_m: int, dual: bool = False, out=None) -> None:
        m, k = x.shape
        n = w.shape[0] // (2 if dual else 1)
        gelu_out = out is not None
        if gelu_out and (not dual or tiles.splits != 1):
            raise ValueError("The GELU epilogue needs the dual gate/up GEMM with one split.")
        grid = (triton.cdiv(n, tiles.block_n), tiles.splits, triton.cdiv(m, block_m))
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
            BLOCK_M=block_m,
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
        p_qkv, p_o, p_gu, p_down = self.p_qkv, self.p_o, self.p_gu, self.p_down
        n_rows = nh * horizon
        t = self.tiles if self.tiles is not None else loop_tiles(m)
        a, block_m = t.attn, t.block_m
        if (
            t.qkv.splits > self.p_qkv.shape[0]
            or t.o.splits > self.p_o.shape[0]
            or t.gateup.splits > self.p_gu.shape[0]
            or t.down.splits > self.p_down.shape[0]
            or batch * a.kv_splits * n_rows > self.part_m.shape[0]
        ):
            raise ValueError("These tiles split further than the scratch buffers were sized for.")
        keys_per_split = triton.cdiv(triton.cdiv(prefix_len, a.kv_splits), a.block_n) * a.block_n
        dt = -1.0 / num_steps
        mods = self.modulations
        gelu_fused = t.gateup.splits == 1

        for step in range(num_steps):
            mod = mods[step]
            _entry_kernel[(m,)](x_t, self.w_in_t, self.b_in, mod[0], res, xn, self.eps, A=action_dim, N=w, num_warps=4)
            for layer in range(self.num_layers):
                k_l, v_l = key[layer, :, 0], value[layer, :, 0]  # (B, n_keys, hd)
                mod_in, mod_post = mod[2 * layer], mod[2 * layer + 1]
                self._gemm(xn, self.w_qkv[layer], p_qkv, t.qkv, block_m)
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
                    SPLITS=t.qkv.splits,
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
                self._gemm(att, self.w_o[layer], p_o, t.o, block_m)
                _resid_norm_kernel[(m,)](
                    p_o, res, mod_in[2 * w :], mod_post, xn, m, self.eps, SPLITS=t.o.splits, N=w, num_warps=4
                )
                if gelu_fused:
                    self._gemm(xn, self.w_gu[layer], p_gu, t.gateup, block_m, dual=True, out=hid)
                else:
                    self._gemm(xn, self.w_gu[layer], p_gu, t.gateup, block_m, dual=True)
                    _gelu_mul_kernel[(m, triton.cdiv(self.mlp_dim, 1024))](
                        p_gu, hid, m, SPLITS=t.gateup.splits, N=self.mlp_dim, BLOCK=1024, num_warps=4
                    )
                self._gemm(hid, self.w_down[layer], p_down, t.down, block_m)
                if layer + 1 < self.num_layers:
                    _resid_norm_kernel[(m,)](
                        p_down,
                        res,
                        mod_post[2 * w :],
                        mod[2 * layer + 2],
                        xn,
                        m,
                        self.eps,
                        SPLITS=t.down.splits,
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
                        SPLITS=t.down.splits,
                        N=w,
                        A=action_dim,
                        num_warps=8,
                    )
        return x_t


# ──────────────────────────────────────────────────────────────────────
# Prefix forward (PaliGemma LM over the prefix) and SigLIP vision tower
# ──────────────────────────────────────────────────────────────────────
# Both are compute-bound GEMM chains at 512-612 rows, where cuBLAS is the
# fastest GEMM on an A100. What is fused here is everything around the GEMMs:
# the Q/K/V projections run as one GEMM (three GEMMs of which two had 256
# output columns), the residual add is folded into the norm that follows it,
# RoPE writes K/V straight into the cache, the prefix attention is one
# flash-attention kernel with the additive mask, and SigLIP's LayerNorms take
# their residual add. The GEMM inputs and outputs are bf16, the prefix residual
# stream float32 and SigLIP's bf16, as in the eager path.

if HAS_TRITON:

    @triton.jit
    def _prefix_rope_kernel(
        QKV,
        COS,
        SIN,
        Q_OUT,
        K_CACHE,
        V_CACHE,
        n_rows,
        seq_len,
        qkv_stride,
        kv_row_stride,
        kv_batch_stride,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_Q: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Rows ``rm`` of the fused QKV projection ``(n_rows, (heads + 2) · HEAD_DIM)``
        (``(2 · HEAD_DIM)`` without ``HAS_Q``), head ``program_id(1)``: Gemma's
        half-split RoPE with eager's bf16 rounding points on Q and K; Q stored
        ``(requests, heads, seq_len, HEAD_DIM)``, K/V into the cache's prefix slots."""
        HALF: tl.constexpr = HEAD_DIM // 2
        rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rm[:, None] < n_rows
        if HAS_Q:
            head = tl.program_id(1)
            col0 = head * HEAD_DIM
        else:
            head = tl.program_id(1) + NUM_HEADS
            col0 = tl.program_id(1) * HEAD_DIM
        b = rm // seq_len
        t = rm % seq_len
        cols = tl.arange(0, HALF)
        src = QKV + rm[:, None] * qkv_stride + col0 + cols[None, :]
        lo = tl.load(src, mask=row_mask, other=0.0).to(tl.float32)
        hi = tl.load(src + HALF, mask=row_mask, other=0.0).to(tl.float32)
        if head <= NUM_HEADS:
            rope = rm[:, None] * HEAD_DIM + cols[None, :]
            cos_lo = tl.load(COS + rope, mask=row_mask, other=0.0).to(tl.float32)
            cos_hi = tl.load(COS + rope + HALF, mask=row_mask, other=0.0).to(tl.float32)
            sin_lo = tl.load(SIN + rope, mask=row_mask, other=0.0).to(tl.float32)
            sin_hi = tl.load(SIN + rope + HALF, mask=row_mask, other=0.0).to(tl.float32)
            # ``q · cos + rotate_half(q) · sin``, each product and the sum rounded as eager's bf16 ops.
            out_lo = _bf16(_bf16(lo * cos_lo) + _bf16(-hi * sin_lo))
            out_hi = _bf16(_bf16(hi * cos_hi) + _bf16(lo * sin_hi))
            if head < NUM_HEADS:
                dst = Q_OUT + ((b * NUM_HEADS + head) * seq_len + t)[:, None] * HEAD_DIM + cols[None, :]
            else:
                dst = K_CACHE + b[:, None] * kv_batch_stride + t[:, None] * kv_row_stride + cols[None, :]
            tl.store(dst, out_lo.to(tl.bfloat16), mask=row_mask)
            tl.store(dst + HALF, out_hi.to(tl.bfloat16), mask=row_mask)
        else:
            dst = V_CACHE + b[:, None] * kv_batch_stride + t[:, None] * kv_row_stride + cols[None, :]
            tl.store(dst, lo.to(tl.bfloat16), mask=row_mask)
            tl.store(dst + HALF, hi.to(tl.bfloat16), mask=row_mask)

    @triton.jit
    def _add_rms_norm_kernel(
        X,
        ADD,
        RES_OUT,
        WEIGHT,
        OUT,
        eps,
        N: tl.constexpr,
        HAS_ADD: tl.constexpr,
    ):
        """Row ``r``: ``RES_OUT = ADD + X`` (float32; ``ADD`` may be ``RES_OUT``) and Gemma's
        ``RMSNorm(RES_OUT) · (1 + weight)`` rounded to bf16 into ``OUT``."""
        row = tl.program_id(0)
        cols = tl.arange(0, N)
        x = tl.load(X + row * N + cols).to(tl.float32)
        if HAS_ADD:
            x = tl.load(ADD + row * N + cols).to(tl.float32) + x
            tl.store(RES_OUT + row * N + cols, x)
        var = tl.sum(x * x, axis=0) / N
        normed = x * tldevice.rsqrt(var + eps)
        weight = tl.load(WEIGHT + cols).to(tl.float32)
        tl.store(OUT + row * N + cols, (normed * (1.0 + weight)).to(tl.bfloat16))

    @triton.jit
    def _add_layer_norm_kernel(
        X,
        ADD,
        RES_OUT,
        WEIGHT,
        BIAS,
        OUT,
        eps,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        HAS_ADD: tl.constexpr,
    ):
        """Row ``r``: ``RES_OUT = bf16(ADD + X)`` (SigLIP's bf16 residual stream) and
        ``LayerNorm(RES_OUT)`` in float32 rounded to bf16 into ``OUT``."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_ADD:
            x = _bf16(tl.load(ADD + row * N + cols, mask=mask, other=0.0).to(tl.float32) + x)
            tl.store(RES_OUT + row * N + cols, x.to(tl.bfloat16), mask=mask)
        mean = tl.sum(x, axis=0) / N
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / N
        y = xc * tldevice.rsqrt(var + eps)
        w = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + row * N + cols, (y * w + bias).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _prefix_attn_kernel(
        Q,
        K,
        V,
        MASK,
        OUT,
        seq_len,
        n_keys,
        q_head_stride,
        q_batch_stride,
        kv_row_stride,
        kv_batch_stride,
        mask_row_stride,
        mask_batch_stride,
        out_stride,
        scale,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Flash-attention forward of query rows ``rm`` of head ``program_id(1)`` of
        request ``program_id(2)`` over the single KV head: float32 online softmax
        of ``Q Kᵀ · scale + MASK[b, query, key]``, bf16 probabilities against V,
        output stored as ``(b · seq_len + row, head · HEAD_DIM)`` for ``o_proj``."""
        pid_m = tl.program_id(0)
        head = tl.program_id(1)
        b = tl.program_id(2)
        HALF: tl.constexpr = HEAD_DIM // 2
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rm < seq_len
        dh = tl.arange(0, HALF)
        qb = Q + b * q_batch_stride + head * q_head_stride + rm[:, None] * HEAD_DIM
        q_lo = tl.load(qb + dh[None, :], mask=rmask[:, None], other=0.0)
        q_hi = tl.load(qb + HALF + dh[None, :], mask=rmask[:, None], other=0.0)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_lo = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
        acc_hi = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
        kb = K + b * kv_batch_stride
        vb = V + b * kv_batch_stride
        mb = MASK + b * mask_batch_stride + rm[:, None] * mask_row_stride
        for k0 in range(0, n_keys, BLOCK_N):
            kidx = k0 + tl.arange(0, BLOCK_N)
            kmask = kidx < n_keys
            k_lo = tl.load(kb + kidx[None, :] * kv_row_stride + dh[:, None], mask=kmask[None, :], other=0.0)
            k_hi = tl.load(kb + kidx[None, :] * kv_row_stride + HALF + dh[:, None], mask=kmask[None, :], other=0.0)
            sc = (tl.dot(q_lo, k_lo) + tl.dot(q_hi, k_hi)) * scale
            bias = tl.load(mb + kidx[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
            sc = tl.where(kmask[None, :], sc + bias.to(tl.float32), float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tldevice.exp(m_i - m_safe)
            p = tldevice.exp(sc - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            p16 = p.to(tl.bfloat16)
            v_lo = tl.load(vb + kidx[:, None] * kv_row_stride + dh[None, :], mask=kmask[:, None], other=0.0)
            v_hi = tl.load(vb + kidx[:, None] * kv_row_stride + HALF + dh[None, :], mask=kmask[:, None], other=0.0)
            acc_lo = acc_lo * alpha[:, None] + tl.dot(p16, v_lo)
            acc_hi = acc_hi * alpha[:, None] + tl.dot(p16, v_hi)
            m_i = m_new
        ob = OUT + (b * seq_len + rm)[:, None] * out_stride + head * HEAD_DIM
        tl.store(ob + dh[None, :], (acc_lo / l_i[:, None]).to(tl.bfloat16), mask=rmask[:, None])
        tl.store(ob + HALF + dh[None, :], (acc_hi / l_i[:, None]).to(tl.bfloat16), mask=rmask[:, None])


class FastPrefix:
    """The prefix forward of ``PaliGemmaWithActionExpertPi05`` with fused Q/K/V
    weights, add+norm kernels, RoPE into the cache and a flash-attention
    kernel; cuBLAS GEMMs."""

    def __init__(self, model):
        if not HAS_TRITON:
            raise RuntimeError("The π0.5 fast prefix forward needs Triton.")
        lm = model.paligemma_with_expert.paligemma.model.language_model
        self.lm = lm
        attn0 = lm.layers[0].self_attn
        self.dtype = attn0.q_proj.weight.dtype
        if self.dtype != torch.bfloat16:
            raise RuntimeError(f"The fast prefix forward runs bfloat16 GEMMs, not {self.dtype}.")
        if attn0.config.num_key_value_heads != 1:
            raise RuntimeError("The fast prefix forward needs a single KV head.")
        self.num_heads = attn0.config.num_attention_heads
        self.head_dim = attn0.head_dim
        self.width = attn0.q_proj.in_features
        self.n_q = self.num_heads * self.head_dim
        self.layers = []
        with torch.no_grad():
            for layer in lm.layers:
                attn, mlp = layer.self_attn, layer.mlp
                w_qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]).contiguous()
                attn.q_proj.weight = nn.Parameter(w_qkv[: self.n_q], requires_grad=False)
                attn.k_proj.weight = nn.Parameter(w_qkv[self.n_q : self.n_q + self.head_dim], requires_grad=False)
                attn.v_proj.weight = nn.Parameter(w_qkv[self.n_q + self.head_dim :], requires_grad=False)
                self.layers.append(
                    (
                        layer.input_layernorm,
                        w_qkv,
                        attn.o_proj.weight,
                        layer.post_attention_layernorm,
                        mlp.gate_proj.weight,
                        mlp.up_proj.weight,
                        mlp.down_proj.weight,
                    )
                )
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _norm(self, x, norm, out, add=None, res_out=None):
        n_rows, n = x.shape
        _add_rms_norm_kernel[(n_rows,)](
            x,
            add if add is not None else x,
            res_out if res_out is not None else x,
            norm.weight,
            out,
            norm.eps,
            N=n,
            HAS_ADD=add is not None,
            num_warps=8,
        )

    def run(self, inputs_embeds, attention_mask, position_ids, kv_cache, *, prefix_kv_only=False, rotary_emb=None):
        """``PaliGemmaWithActionExpertPi05.forward``'s prefix pass: writes every
        layer's prefix K/V into ``kv_cache``; returns the final hidden state, or
        None with ``prefix_kv_only``."""
        from vllm_omni.diffusion.models.pi05.modeling_pi05 import _fused_gelu_mul

        lm = self.lm
        batch, seq_len, width = inputs_embeds.shape
        n_rows = batch * seq_len
        device = inputs_embeds.device
        nh, hd = self.num_heads, self.head_dim
        if rotary_emb is None:
            rotary_emb = lm.rotary_emb(inputs_embeds.new_empty((), dtype=self.dtype), position_ids)
        cos, sin = rotary_emb
        cos, sin = cos.reshape(n_rows, hd), sin.reshape(n_rows, hd)
        if cos.dtype != torch.bfloat16:
            cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        residual = inputs_embeds.reshape(n_rows, width).to(torch.float32).contiguous()
        if residual.data_ptr() == inputs_embeds.data_ptr():
            residual = residual.clone()
        x_normed = torch.empty(n_rows, width, dtype=torch.bfloat16, device=device)
        q = torch.empty(batch, nh, seq_len, hd, dtype=torch.bfloat16, device=device)
        att = torch.empty(n_rows, nh * hd, dtype=torch.bfloat16, device=device)
        if attention_mask.shape != (batch, 1, seq_len, seq_len) or attention_mask.dtype != torch.float32:
            raise ValueError(f"Unexpected prefix attention mask {tuple(attention_mask.shape)} {attention_mask.dtype}.")
        mask = attention_mask[:, 0]
        last = len(self.layers) - 1
        mlp_out = None
        for idx, (in_norm, w_qkv, w_o, post_norm, w_gate, w_up, w_down) in enumerate(self.layers):
            if mlp_out is None:
                self._norm(residual, in_norm, x_normed)
            else:
                self._norm(mlp_out, in_norm, x_normed, add=residual, res_out=residual)
            key, value = kv_cache.key[idx, :, 0], kv_cache.value[idx, :, 0]  # (B, prefix + suffix, hd)
            kv_only = prefix_kv_only and idx == last
            qkv = torch.nn.functional.linear(x_normed, w_qkv[self.n_q :] if kv_only else w_qkv)
            _prefix_rope_kernel[(triton.cdiv(n_rows, 64), 2 if kv_only else nh + 2)](
                qkv,
                cos,
                sin,
                q,
                key,
                value,
                n_rows,
                seq_len,
                qkv.stride(0),
                key.stride(1),
                key.stride(0),
                NUM_HEADS=nh,
                HEAD_DIM=hd,
                HAS_Q=not kv_only,
                BLOCK_M=64,
                num_warps=4,
            )
            if kv_only:
                return None
            _prefix_attn_kernel[(triton.cdiv(seq_len, 64), nh, batch)](
                q,
                key,
                value,
                mask,
                att,
                seq_len,
                seq_len,
                q.stride(1),
                q.stride(0),
                key.stride(1),
                key.stride(0),
                mask.stride(1),
                mask.stride(0),
                att.stride(0),
                self.scale,
                HEAD_DIM=hd,
                BLOCK_M=64,
                BLOCK_N=128,
                num_warps=8,
                num_stages=1,
            )
            attn_out = torch.nn.functional.linear(att, w_o)
            self._norm(attn_out, post_norm, x_normed, add=residual, res_out=residual)
            gate = torch.nn.functional.linear(x_normed, w_gate)
            _fused_gelu_mul(gate, torch.nn.functional.linear(x_normed, w_up), gate)
            mlp_out = torch.nn.functional.linear(gate, w_down)
        hidden = residual + mlp_out.to(torch.float32)
        return lm.norm(hidden.view(batch, seq_len, width))


class FastVision:
    """SigLIP's encoder with fused Q/K/V weights, add+LayerNorm kernels and
    flash attention; the embeddings (float32 patch GEMM + positions) and the
    projector are the module's own."""

    def __init__(self, model):
        if not HAS_TRITON:
            raise RuntimeError("The π0.5 fast vision tower needs Triton.")
        tower = model.paligemma_with_expert.paligemma.model.vision_tower
        self.vision = getattr(tower, "vision_model", tower)
        encoder = self.vision.encoder
        layer0 = encoder.layers[0]
        if layer0.self_attn.q_proj.weight.dtype != torch.bfloat16:
            raise RuntimeError("The fast vision tower runs a bfloat16 encoder (pi05_vision_precision selective_bf16).")
        self.num_heads = layer0.self_attn.num_heads
        self.head_dim = layer0.self_attn.head_dim
        self.width = layer0.self_attn.embed_dim
        self.layers = []
        with torch.no_grad():
            for layer in encoder.layers:
                attn, mlp = layer.self_attn, layer.mlp
                w_qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]).contiguous()
                b_qkv = torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias]).contiguous()
                w = self.width
                attn.q_proj.weight = nn.Parameter(w_qkv[:w], requires_grad=False)
                attn.k_proj.weight = nn.Parameter(w_qkv[w : 2 * w], requires_grad=False)
                attn.v_proj.weight = nn.Parameter(w_qkv[2 * w :], requires_grad=False)
                attn.q_proj.bias = nn.Parameter(b_qkv[:w], requires_grad=False)
                attn.k_proj.bias = nn.Parameter(b_qkv[w : 2 * w], requires_grad=False)
                attn.v_proj.bias = nn.Parameter(b_qkv[2 * w :], requires_grad=False)
                self.layers.append(
                    (layer.layer_norm1, w_qkv, b_qkv, attn.out_proj, layer.layer_norm2, mlp.fc1, mlp.fc2)
                )
        self.post_norm = self.vision.post_layernorm
        self.block = triton.next_power_of_2(self.width)

    def _norm(self, x, norm, out, add=None, res_out=None):
        n_rows, n = x.shape
        _add_layer_norm_kernel[(n_rows,)](
            x,
            add if add is not None else x,
            res_out if res_out is not None else x,
            norm.weight,
            norm.bias,
            out,
            norm.eps,
            N=n,
            BLOCK=self.block,
            HAS_ADD=add is not None,
            num_warps=4,
        )

    def run(self, pixel_values):
        """``vision_tower(pixel_values).last_hidden_state``."""
        h = self.vision.embeddings(pixel_values)  # (B, tokens, width), bf16 after the hook
        batch, tokens, width = h.shape
        n_rows = batch * tokens
        nh, hd = self.num_heads, self.head_dim
        residual = h.reshape(n_rows, width).contiguous()
        if residual.data_ptr() == h.data_ptr():
            residual = residual.clone()
        x = torch.empty_like(residual)
        mlp_out = None
        for ln1, w_qkv, b_qkv, out_proj, ln2, fc1, fc2 in self.layers:
            if mlp_out is None:
                self._norm(residual, ln1, x)
            else:
                self._norm(mlp_out, ln1, x, add=residual, res_out=residual)
            qkv = torch.nn.functional.linear(x, w_qkv, b_qkv).view(batch, tokens, 3, nh, hd)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            att = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            att = att.transpose(1, 2).reshape(n_rows, width)
            o = out_proj(att)
            self._norm(o, ln2, x, add=residual, res_out=residual)
            f = torch.nn.functional.gelu(fc1(x), approximate="tanh")
            mlp_out = fc2(f)
        out = torch.empty_like(residual)
        self._norm(mlp_out, self.post_norm, out, add=residual, res_out=residual)
        return out.view(batch, tokens, width)
