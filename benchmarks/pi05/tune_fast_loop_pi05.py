# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tile sweep for the GEMMs of the π0.5 fast denoising loop (``fast_pi05.DEFAULT_TILES``).

Times ``_skinny_gemm_kernel`` at each of the action expert's four GEMM shapes
over a grid of tiles, the way the loop runs it: 18 launches per CUDA graph,
each over another layer's weight, so every launch streams its weight from
memory. Needs a GPU only: the weights are random, the shapes are those of the
``gemma_300m`` expert.

    python benchmarks/pi05/tune_fast_loop_pi05.py --rows 10

``--rows`` is the number of action tokens of one launch (``chunk_size`` times
the requests of a formed batch). Put the best tiles of each GEMM in
``DEFAULT_TILES``; ``gateup`` must keep one split for its fused GELU epilogue
(the ``best splits=1`` line).
"""

from __future__ import annotations

import argparse
import itertools

import torch
import triton

from vllm_omni.diffusion.models.pi05.fast_pi05 import _block_m, _skinny_gemm_kernel

# name: (output columns, input columns, two stacked weights)
SHAPES = {
    "qkv": (2560, 1024, False),
    "o": (1024, 2048, False),
    "gateup": (4096, 1024, True),
    "down": (1024, 4096, False),
}
LAYERS = 18
GRID = {
    "block_n": (32, 64, 128),
    "block_k": (64, 128, 256),
    "splits": (1, 2, 4, 8),
    "num_warps": (4, 8),
    "num_stages": (2, 3, 4),
}


def time_graph_us(launch, replays: int = 5) -> float:
    """Microseconds per launch of ``LAYERS`` launches captured as one CUDA graph."""
    launch(0)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(LAYERS):
            launch(layer)
    graph.replay()
    torch.accelerator.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / replays / LAYERS * 1e3


def sweep(name: str, rows: int, top: int) -> None:
    n, k, dual = SHAPES[name]
    n_out = n * (2 if dual else 1)
    block_m = _block_m(rows)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(n_out, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(LAYERS)]
    megabytes = n_out * k * 2 / 1e6
    results, rejected = [], 0
    for block_n, block_k, splits, num_warps, num_stages in itertools.product(*GRID.values()):
        if k % splits or (k // splits) % block_k:
            continue
        partials = torch.empty(splits, rows, n_out, device="cuda", dtype=torch.float32)
        grid = (triton.cdiv(n, block_n), splits, triton.cdiv(rows, block_m))

        def launch(layer: int) -> None:
            w = weights[layer]
            _skinny_gemm_kernel[grid](
                x,
                w,
                partials,
                partials,
                rows,
                n,
                k,
                x.stride(0),
                w.stride(0),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                SPLITS=splits,
                DUAL=dual,
                GELU_OUT=False,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        try:
            micros = time_graph_us(launch)
        except Exception:  # a tile that does not fit the device's shared memory
            rejected += 1
            continue
        results.append((micros, block_n, block_k, splits, num_warps, num_stages))
    results.sort()
    best = results[0][0]
    print(
        f"== {name}: {megabytes:.1f} MB per launch, {rows} rows, {len(results)} tiles timed ({rejected} rejected); "
        f"best {best:.1f} us ({megabytes / best * 1e3:.0f} GB/s)"
    )
    for micros, *tiles in results[:top]:
        print(f"   {micros:6.1f} us  GemmTiles({', '.join(str(tile) for tile in tiles)})")
    one_split = [row for row in results if row[3] == 1]
    if one_split:
        micros, *tiles = one_split[0]
        print(f"   best splits=1: {micros:.1f} us  GemmTiles({', '.join(str(tile) for tile in tiles)})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=10, help="Action tokens per launch (chunk_size x batch)")
    parser.add_argument("--gemms", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    parser.add_argument("--top", type=int, default=6, help="Tiles to print per GEMM")
    args = parser.parse_args()
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, triton {triton.__version__}")
    print("GemmTiles(block_n, block_k, splits, num_warps, num_stages)")
    for name in args.gemms:
        sweep(name, args.rows, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
