# π0.5 Benchmarks

Latency benchmarks for π0.5 (`lerobot/pi05_base`) serving, used for the measurements in [`recipes/lerobot/Pi05.md`](../../recipes/lerobot/Pi05.md).

| Script | Measures | Needs |
| --- | --- | --- |
| `e2e_latency_pi05.py` | Client E2E round trip over the OpenPI websocket: msgpack pack, send, engine, denoise, unpack. Reports p50/p95/min/max and the device memory peak sampled from `nvidia-smi`. | A running server |
| `bench_pi05.py` | Model-level `sample_actions` latency and output shape over a batch size × camera-view sweep, with no serving stack. Writes `pi05_sweep.csv` and `pi05_sweep.md`. | A GPU and the checkpoint |
| `bench_batch_pi05.py` | Stage profile of `sample_actions` in rounds of B requests: time per stage (prefix embedding, prefill, denoising) and the DCGM activity of each stage looped alone. Writes one JSON per chain and a Markdown report over them. | A GPU and the checkpoint; `dcgmi` for the activity columns |
| `bench_engine_overhead_pi05.py` | BS=1 latency of the layers in between: model-only and in-process pipeline in one clean process, then `Omni.generate` with the `uni` or `mp` executor. With `--trace-dir`, the per-request marks of `vllm_omni.utils.latency_trace`. | A GPU and the checkpoint |
| `run_engine_overhead_matrix.sh`, `report_engine_overhead_pi05.py` | The full matrix for one dtype (independent runs of every layer plus traced and untraced client E2E, for one or two source trees) and its tables: the four layers, their gaps on the same requests, and p50/p95 per engine segment. | A GPU and the checkpoint |
| `tune_fast_loop_pi05.py` | The denoising loop alone per batch size: the fused per-step loop against the fast loop, and with `--sweep` a sweep of the fast loop's tiles that prints rows for `fast_pi05.LOOP_TILES`, to retune for another GPU. | A GPU and the checkpoint |

## Client E2E latency

Start a server with the recipe's command, then run the client per camera count.
It serves the optimized path in the deploy config's `float32`; pass `--enforce-eager` to the server for the eager baseline and `--dtype bfloat16` for bfloat16.

```bash
# optimized path (CUDA graphs + fused kernels)
vllm serve lerobot/pi05_base --omni --port 8000 --deploy-config vllm_omni/deploy/pi05.yaml --disable-log-stats
# eager baseline
vllm serve lerobot/pi05_base --omni --port 8000 --deploy-config vllm_omni/deploy/pi05.yaml --enforce-eager --disable-log-stats

python benchmarks/pi05/e2e_latency_pi05.py --host 127.0.0.1 --port 8000 --views 3 --warmup 3 --iters 30 --out e2e.json
```

## Model-level latency

The script builds the model through `Pi05Pipeline`, as the server does, from the deploy config's `model_config`.
It runs the optimized path by default; pass `--enforce-eager` for the eager baseline and `--dtype bfloat16` for bfloat16.

```bash
# optimized path (CUDA graphs + fused kernels)
python benchmarks/pi05/bench_pi05.py --bs 1 --views 3 --warmup 5 --iters 30 --out pi05_bench/
# eager baseline
python benchmarks/pi05/bench_pi05.py --enforce-eager --bs 1 --views 3 --warmup 5 --iters 30 --out pi05_bench_eager/
```

## Experimental P0 options

The benchmark supports independent `--vision-precision` (`fp32`, `bf16_autocast`,
`selective_bf16`) and `--batch-cameras` / `--no-batch-cameras` options to the model
benchmark. The corresponding stage `custom_pipeline_args` keys are
`pi05_vision_precision` and `pi05_batch_cameras`; CLI values override them.
Defaults preserve R0's FP32 vision/projector and sequential cameras. Both BF16
experiments restore FP32 image features before the prefix LM, preserving R0's
prefix residual dtype. Selective BF16 retains FP32 patch/position embeddings;
autocast retains all FP32 vision/projector weights. Changing either precision or
camera batching can change rounding; action parity and task quality need to be
evaluated separately from latency.
The 1/2-camera cases retain three image slots with false masks for missing views.

Three independent runtime flags are also available, all disabled by default:
`--precompute-adarms` prepares the default denoising schedule's timestep/conditioning
projections after checkpoint load; `--hoist-denoise-invariants` computes mask,
position IDs, and RoPE once per request; `--prefix-kv-only` omits the final prefix
hidden-state work when action generation only consumes KV. Their stage
`custom_pipeline_args` keys are `pi05_precompute_adarms`,
`pi05_hoist_denoise_invariants`, and `pi05_prefix_kv_only`. Each flag also has a
`--no-...` form for ablation. General prefix forward keeps its full hidden output.

`--execution-mode eager|fused|graph` selects the implementation independently from
these numerical/algorithm settings. `--enforce-eager` remains an alias for eager;
combining it with fused/graph is an error. Default execution remains graph.

```bash
python benchmarks/pi05/bench_pi05.py --dtype bfloat16 --bs 1 --views 1 2 3 \
  --vision-precision selective_bf16 --batch-cameras --warmup 20 --iters 30 \
  --out pi05_p0_selective_batched/

# Full P0 candidate; use --no-<flag> for individual ablations, and compare the
# same options under --execution-mode eager / fused / graph.
python benchmarks/pi05/bench_pi05.py --dtype bfloat16 --bs 1 --views 1 2 3 \
  --vision-precision selective_bf16 --batch-cameras --precompute-adarms \
  --hoist-denoise-invariants --prefix-kv-only --execution-mode graph \
  --warmup 20 --iters 30 --out pi05_p0_all_on/
```

These are latency/finite-output checks, not a fixed-noise numerical parity gate.
Run each policy in a fresh process; correctness, capture/replay,
stage profiling, and task-level quality require separate validation. Compare
latency only at the same model/serving boundary, with matching inputs, denoising
steps, warmup, GPU, and precision policy.

## Stage profile

`bench_batch_pi05.py` times the three calls `sample_actions` makes, in place, for rounds of B requests, then
replays each stage alone so that the DCGM samples of a window belong to one stage. One run measures one chain
(`bfloat16` unless `--dtype` says otherwise); the report covers every chain file in the directory.

```bash
python benchmarks/pi05/bench_batch_pi05.py --mode eager --out pi05_stages/
python benchmarks/pi05/bench_batch_pi05.py --mode graph --out pi05_stages/
python benchmarks/pi05/bench_batch_pi05.py --mode graph --vision-precision selective_bf16 --all4 --out pi05_stages/
python benchmarks/pi05/bench_batch_pi05.py --report pi05_stages/ --out-md pi05_stages/report.md
```

`--all4` turns on the four opt-in runtime switches above. The optimized path captures the three regions for
batch sizes 1, the powers of two below the largest `--batches` value, and that value; a batch size between two
of them runs its prefix regions without a graph and its denoising step on the next larger graph. A batch size
the optimized path has no CUDA graph for is reported as such rather than timed on its eager fallback.

The report also sets every batch against the same requests run one at a time (B times the chain's batch-1
p50): req/s, speedup, scaling efficiency, the run's round-to-round spread and peak memory, per chain and
batch size. The matching correctness check is `tests/diffusion/models/pi05/test_pi05_batch_parity.py`: each
row of a formed batch against its request alone, and no cross-request contamination.

## Fast path

`pi05_fast_path` (stage `custom_pipeline_args`, with `pi05_precompute_adarms` and `pi05_hoist_denoise_invariants`; bfloat16 only) replaces three parts of a request with kernels shaped for short action chunks (`vllm_omni/diffusion/models/pi05/fast_pi05.py`):

- the denoising loop: split-K GEMMs that stream the expert weights, one attention kernel over the KV cache, the reductions folded into the residual + AdaRMS kernels, all steps as one CUDA graph;
- the prefix forward: one Q/K/V GEMM per layer, add + RMSNorm kernels, RoPE written straight into the KV cache, a flash-attention kernel with the mask;
- the SigLIP encoder (with `pi05_vision_precision: selective_bf16`): one Q/K/V GEMM per layer, add + LayerNorm kernels, flash attention.

A request of a captured batch size then replays as a single graph. The denoising loop picks its tiles by the size of the launch (`fast_pi05.LOOP_TILES`, swept at 1 to 128 requests of 10 action tokens), so it serves every formed batch. The table is for short chunks: chunks over 16 action tokens keep the fast prefix forward and vision tower and denoise on the fused per-step kernels.

[`vllm_omni/deploy/pi05_low_latency.yaml`](../../vllm_omni/deploy/pi05_low_latency.yaml) is the deployment it was built for: bfloat16, 2 camera slots, 100 text tokens, 10 action tokens, every P0 switch and the fast path on. The checkpoint must be trained for 10-token chunks.

```bash
# serve
vllm serve <checkpoint> --omni --port 8000 --deploy-config vllm_omni/deploy/pi05_low_latency.yaml --disable-log-stats

# stage profile: the deployment as configured, then its baseline without the fast path
python benchmarks/pi05/bench_batch_pi05.py --deploy-config vllm_omni/deploy/pi05_low_latency.yaml \
  --batches 1 2 4 8 16 32 64 128 --name fast --out pi05_fast/
python benchmarks/pi05/bench_batch_pi05.py --deploy-config vllm_omni/deploy/pi05_low_latency.yaml \
  --no-fast --batches 1 2 4 8 16 32 64 128 --name baseline --out pi05_fast/
python benchmarks/pi05/bench_batch_pi05.py --report pi05_fast/ --out-md pi05_fast/report.md

# correctness: each fast component, whole requests and formed batches against the kernels they replace
python -m pytest tests/diffusion/models/pi05/test_pi05_fast_path.py -v -s
PI05_BATCH_PARITY_DTYPES=bfloat16 PI05_BATCH_PARITY_SIZES=1,2,4,8,16,32,64,128 \
  PI05_BATCH_PARITY_DEPLOY=vllm_omni/deploy/pi05_low_latency.yaml \
  python -m pytest tests/diffusion/models/pi05/test_pi05_batch_parity.py -v -s
```

A100-SXM4-80GB at its default clocks (1410 MHz at batch 1; check that `nvidia-smi -q` shows no clock limit left by `nvidia-smi -lgc`, a 990 MHz cap makes every number here 1.3x larger), bfloat16, P0 switches on, 10 denoising steps, p50 of whole `sample_actions` calls (the report's `sample_actions` rows, not the sum of the stages timed one by one):

| Deployment | B | P0 baseline | fast path | speedup | fast path req/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 cameras / 100 text tokens / 10 action tokens (`pi05_low_latency.yaml`) | 1 | 33.8 ms | **26.4 ms** | 1.28x | 37.9 |
| | 2 | 49.7 ms | 40.3 ms | 1.23x | 49.6 |
| | 4 | 83.3 ms | 69.2 ms | 1.20x | 57.8 |
| | 8 | 149.1 ms | 125.5 ms | 1.19x | 63.7 |
| | 16 | 279.9 ms | 235.3 ms | 1.19x | 68.0 |
| | 32 | 530.7 ms | 454.5 ms | 1.17x | 70.4 |
| | 64 | 1033.9 ms | 890.5 ms | 1.16x | 71.9 |
| | 128 | 2055.3 ms | 1776.7 ms | 1.16x | 72.0 |
| 3 cameras / 200 text tokens / 50 action tokens (`pi05.yaml` in bfloat16, `--vision-precision selective_bf16 --all4 --fast`) | 1 | 45.9 ms | 42.5 ms | 1.08x | 23.5 |

At batch 1 of the low-latency deployment the stages, timed one by one, go from 4.5 to 3.8 ms (prefix embedding), 15.1 to 12.8 ms (prefill) and 14.2 to 9.7 ms (denoising); replaying the request as one graph instead of one per region saves another 0.35 ms. Served over the OpenPI websocket (`e2e_latency_pi05.py --views 2`, 224-px frames, `mp` executor) the round trip is 30.5 ms p50 against 38.0 ms.

Batching buys little throughput: the prefill is compute-bound from one request on (12.8 ms at batch 1, 1379 ms at batch 128, 78% of that request), so only the denoising loop gains from a larger launch (9.7 ms at batch 1, 97.6 ms at 128). The loop alone, fused per-step kernels against the fast loop (`tune_fast_loop_pi05.py`): 13.6 / 19.9 / 41.9 / 138.0 ms against 9.7 / 14.5 / 28.2 / 95.7 ms at batch 1 / 8 / 32 / 128.

Numerics: the same dtypes as the bfloat16 path (bfloat16 GEMMs and K/V, float32 residual stream, softmax and accumulation) with other rounding points and reduction orders. A chunk moves by up to 1.6% (relative L2) from the baseline's on the test's requests, where bfloat16 itself moves it by up to 6% from float32. LIBERO closed-loop success with OpenPI's `pi05_libero` checkpoint (10-token chunks) served through the OpenPI endpoint, 4 suites × 10 tasks × 10 trials per arm and seed:

| seed | P0 baseline | fast path | identical episodes | only baseline : only fast path |
| --- | ---: | ---: | ---: | ---: |
| 7 | 95.50% (382/400) | 96.00% (384/400) | 384/400 | 7 : 9 |
| 8 | 97.25% (389/400) | 96.75% (387/400) | 384/400 | 9 : 7 |
| both | 96.38% (771/800) | 96.38% (771/800) | 768/800 | 16 : 16 |

Two seeds of one path differ on 23-27 of 400 episodes, so the 16 the two paths differ on per seed are within that spread.

The loop's tiles were swept in the loop itself on an A100, per batch size; `tune_fast_loop_pi05.py --sweep` repeats the sweep on another GPU and prints the rows of `LOOP_TILES`.

## Engine overhead

The time between the model and the client is attributed with an opt-in trace: with `VLLM_OMNI_LATENCY_TRACE_DIR` set, every process stamps each request at about 40 points on one clock, and the report sums consecutive stamps into segments.

```bash
# profile of the current tree: 3 runs per layer, 20 warmup + 200 timed requests per camera count
VARIANTS=after RUNS=3 WARMUP=20 ITERS=200 REQUESTS=200 bash benchmarks/pi05/run_engine_overhead_matrix.sh bf16 results/
# against a reference checkout as well
BEFORE_TREE=/path/to/reference/vllm-omni RUNS=3 WARMUP=20 ITERS=200 REQUESTS=200 bash benchmarks/pi05/run_engine_overhead_matrix.sh bf16 results/
python benchmarks/pi05/report_engine_overhead_pi05.py results/ --out-md results/report.md --out-json results/report.json
```

## Reference results

### Client E2E latency

RTX 5080 16 GB, batch size 1, 10 denoising steps, p50/p95 of 30 round trips after 3 warmup, untimed server:

| dtype | path | p50 ms, 1 / 2 / 3 cameras | p95 ms, 1 / 2 / 3 cameras | device peak (GiB) |
| --- | --- | ---: | ---: | ---: |
| float32 | eager | 263.3 / 264.2 / 264.8 | 272.5 / 272.2 / 267.3 | 14.67 |
| float32 | **CUDA graphs + fused kernels** | **240.5 / 241.9 / 242.4** | 243.1 / 245.9 / 243.8 | 14.95 |
| bfloat16 | eager | 166.4 / 166.7 / 167.7 | 167.4 / 168.2 / 190.6 | 8.71 |
| bfloat16 | **CUDA graphs + fused kernels** | **118.0 / 118.1 / 118.2** | 118.9 / 120.1 / 119.6 | 9.07 |

### Model-level latency

RTX 5080 16 GB, batch size 1, 3 cameras, 10 steps, 5 warmup:

| dtype | PR 6950 (eager) | eager path | optimized path (CUDA graph + fused Triton kernels) |
| --- | ---: | ---: | ---: |
| float32 | 264.0 ms | 260.4 ms | 237.7 ms |
| bfloat16 | 173.0 ms | 168.3 ms | 114.0 ms |
