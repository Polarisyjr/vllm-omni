# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Time the π0.5 fast denoising loop per batch size and sweep its tiles (``fast_pi05.LOOP_TILES``).

For each batch size the script fills a KV cache with the real prefix forward
and times the whole denoising schedule as one CUDA graph, replayed until the
GPU clock has ramped up: the fused per-step loop (what serving runs without
``pi05_fast_path``), the fast loop on the tiles ``LOOP_TILES`` has for that
launch size, and with ``--sweep`` the fast loop over a coordinate sweep of its
tiles (row tile, each GEMM's column tile, K tile, splits, warps and stages,
the attention's row tile, key tile and key splits), starting from the table's
entry. It prints the best tiles as a ``LOOP_TILES`` row.

    # compare the loops at the deployment's chunk length
    python benchmarks/pi05/tune_fast_loop_pi05.py --batches 1 2 4 8 16 32 64 128
    # retune for another GPU or chunk length: paste the printed rows into LOOP_TILES
    python benchmarks/pi05/tune_fast_loop_pi05.py --batches 1 2 4 8 16 32 64 128 --sweep

A sweep takes about 10 minutes per batch size. Run one batch size per GPU to
shorten it; do not share the GPU, and check it carries no clock limit.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import time
from pathlib import Path

import torch
from bench_batch_pi05 import seeded_inputs
from bench_pi05 import build_model, resolve_checkpoint

import vllm_omni
from vllm_omni.diffusion.models.pi05.fast_pi05 import AttnTiles, FastDenoiser, GemmTiles, LoopTiles, loop_tiles
from vllm_omni.diffusion.models.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

GEMMS = ("qkv", "o", "down", "gateup")
# The most splits a candidate may use: the scratch buffers are sized for them once.
MAX_SPLITS, MAX_KV_SPLITS = 8, 8


def timed_ms(fn, warm_s: float, time_s: float) -> float:
    """Milliseconds per call of ``fn`` replayed as one CUDA graph, after ``warm_s`` of replays."""
    for _ in range(2):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start = time.perf_counter()
    while time.perf_counter() - start < warm_s:
        graph.replay()
        torch.accelerator.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    replays, start = 0, time.perf_counter()
    begin.record()
    while time.perf_counter() - start < time_s or replays < 3:
        graph.replay()
        replays += 1
        if replays % 8 == 0:
            torch.accelerator.synchronize()
    end.record()
    torch.accelerator.synchronize()
    return begin.elapsed_time(end) / replays


def table_row(rows: int, tiles: LoopTiles) -> str:
    """``tiles`` as the ``LOOP_TILES`` entry of a launch over ``rows`` rows."""
    parts = [str(tiles.block_m)]
    parts += [str(dataclasses.astuple(getattr(tiles, name))) for name in ("qkv", "o", "gateup", "down", "attn")]
    return f"({rows}, _loop_tiles({', '.join(parts)})),"


def candidates(tiles: LoopTiles, width: int, mlp_dim: int, heads_dim: int):
    """``(label, tiles)`` one coordinate away from ``tiles``."""
    for block_m in (16, 32, 64, 128):
        yield f"block_m={block_m}", dataclasses.replace(tiles, block_m=block_m)
    in_features = {"qkv": width, "o": heads_dim, "gateup": width, "down": mlp_dim}
    for name in GEMMS:
        current = getattr(tiles, name)
        k = in_features[name]
        for splits in (1,) if name == "gateup" else (1, 2, 4, 8):  # the GELU epilogue needs gate/up unsplit
            for block_n, block_k in ((32, 64), (32, 128), (64, 64), (64, 128), (128, 64), (128, 128)):
                if k % splits == 0 and (k // splits) % block_k == 0:
                    gemm = dataclasses.replace(current, block_n=block_n, block_k=block_k, splits=splits)
                    yield f"{name} {dataclasses.astuple(gemm)}", dataclasses.replace(tiles, **{name: gemm})
        for num_warps, num_stages in ((4, 2), (4, 3), (4, 4), (8, 3), (8, 4)):
            gemm = dataclasses.replace(current, num_warps=num_warps, num_stages=num_stages)
            yield f"{name} {dataclasses.astuple(gemm)}", dataclasses.replace(tiles, **{name: gemm})
    for block_r, block_n, kv_splits, num_warps in itertools.product((16, 32, 64), (64, 128), (1, 2, 4, 6, 8), (4, 8)):
        attn = dataclasses.replace(
            tiles.attn, block_r=block_r, block_n=block_n, kv_splits=kv_splits, num_warps=num_warps
        )
        yield f"attn {dataclasses.astuple(attn)}", dataclasses.replace(tiles, attn=attn)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="lerobot/pi05_base")
    parser.add_argument(
        "--deploy-config", default=str(Path(vllm_omni.__file__).parent / "deploy" / "pi05_low_latency.yaml")
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--sweep", action="store_true", help="Sweep the fast loop's tiles at each batch size")
    parser.add_argument("--rounds", type=int, default=2, help="Passes of the coordinate sweep")
    parser.add_argument("--warm-seconds", type=float, default=0.5, help="Replays before each timing")
    parser.add_argument("--seconds", type=float, default=0.5, help="Length of each timing")
    parser.add_argument("--min-gain", type=float, default=0.005, help="Relative gain a candidate needs to be kept")
    args = parser.parse_args()

    # The fused kernels without graphs or the fast path: this script builds its own loops and graphs.
    model = build_model(
        resolve_checkpoint(args.model),
        args.deploy_config,
        "bfloat16",
        False,
        "selective_bf16",
        True,
        precompute_adarms=True,
        hoist_denoise_invariants=True,
        prefix_kv_only=True,
        fast_path=False,
        execution_mode="fused",
        max_num_seqs=max(args.batches),
    )
    layer = model.paligemma_with_expert.gemma_expert.model.layers[0]
    width, mlp_dim = layer.self_attn.q_proj.in_features, layer.mlp.gate_proj.out_features
    heads_dim = layer.self_attn.o_proj.in_features
    # Sized once for every candidate; each run assigns the tiles it times.
    unsplit, most = GemmTiles(64, 64, 1), GemmTiles(64, 64, MAX_SPLITS)
    sizing = LoopTiles(64, most, most, unsplit, most, AttnTiles(kv_splits=MAX_KV_SPLITS))
    denoiser = FastDenoiser(model, max_batch=max(args.batches), tiles=sizing)
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, {model.action_horizon}-token chunks")

    def timed(fn) -> float:
        return timed_ms(fn, args.warm_seconds, args.seconds)

    rows_table = []
    with torch.no_grad():
        for batch in args.batches:
            inputs = seeded_inputs(model, 0, batch)
            inputs["lang_masks"][:, inputs["lang_masks"].shape[1] * 3 // 5 :] = False  # masked keys, as real prompts
            embs, pad_masks, att_masks = model.embed_prefix(
                inputs["images"], inputs["image_masks"], inputs["lang_tokens"], inputs["lang_masks"]
            )
            mask = prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            cache = model.new_kv_cache(batch, embs.shape[1])
            model.paligemma_with_expert.forward(
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                inputs_embeds=[embs, None],
                use_cache=True,
                prefix_kv_only=True,
                rotary_emb=model.prepare_prefix_rotary(embs, position_ids),
            )
            context = model.prepare_denoise_context(pad_masks)
            steps = model.runtime_step_table(model.num_inference_steps, batch)
            noise, x_t = inputs["noise"], inputs["noise"].clone()

            def fused(pad_masks=pad_masks, cache=cache, context=context, steps=steps, noise=noise):
                x, dt = noise.clone(), -1.0 / len(steps)
                for row in steps:
                    v = model.denoise_step(
                        prefix_pad_masks=pad_masks,
                        past_key_values=cache,
                        x_t=x,
                        timestep=row.timestep,
                        context=context,
                        step_data=row,
                    )
                    x = x + dt * v
                return x

            def fast_ms(tiles: LoopTiles, x_t=x_t, cache=cache, context=context) -> float:
                denoiser.tiles = tiles
                return timed(lambda: denoiser.run(x_t, cache, context))

            rows = batch * model.action_horizon
            reference = fused()
            tiles = loop_tiles(rows)
            denoiser.tiles = tiles
            drift = (denoiser.run(noise.clone(), cache, context) - reference).abs().max().item()
            fused_ms = min(timed(fused), timed(fused))
            best = min(fast_ms(tiles), fast_ms(tiles))
            line = f"B={batch} ({rows} rows): fused loop {fused_ms:.2f} ms, fast loop {best:.2f} ms"
            print(f"{line} ({fused_ms / best:.2f}x, max |diff| {drift:.1e})", flush=True)
            if not args.sweep:
                continue
            for _ in range(args.rounds):
                kept = 0
                for label, candidate in candidates(tiles, width, mlp_dim, heads_dim):
                    if candidate == tiles:
                        continue
                    try:
                        ms = fast_ms(candidate)
                    except Exception as exc:  # tiles the device's shared memory cannot hold
                        print(f"   {label}: {type(exc).__name__}", flush=True)
                        continue
                    if ms < best * (1.0 - args.min_gain):
                        best, tiles, kept = ms, candidate, kept + 1
                        print(f"   {label}: {ms:.2f} ms", flush=True)
                if not kept:
                    break
            rows_table.append((batch, rows, fused_ms, best, tiles))
            print(f"   swept: {best:.2f} ms  {table_row(rows, tiles)}", flush=True)

    if rows_table:
        print("\nLOOP_TILES rows (start each entry halfway down to the size before it):")
        for batch, rows, fused_ms, best, tiles in rows_table:
            print(f"    {table_row(rows, tiles)}  # B={batch}: {best:.2f} ms, fused {fused_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
