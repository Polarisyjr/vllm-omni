# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5's fast path (``pi05_fast_path``, ``fast_pi05``) against the path it replaces.

Loads the real checkpoint once in bfloat16 through ``Pi05Pipeline`` under the
low-latency deployment (``vllm_omni/deploy/pi05_low_latency.yaml``: 2 camera
slots, 100 text tokens, 10-token chunks, the P0 switches and the fast path),
with the graphs captured for 1, 2, 4 and 8 requests. One model serves every path:
the fast hooks and the graphs are taken off it to run the fused kernels the
fast path replaces (the baseline), which read the same weights.

* Each fast component against the baseline's on the same inputs: the vision
  tower, the prefix forward (K/V cache and hidden state) and the denoising
  loop over one cache.
* A whole request: the served request graph is its kernels run eagerly bit for
  bit, and stays within bfloat16 rounding of the baseline.
* Formed batches on the fast loop: each row against its request alone, no
  cross-request contamination, and the loop's rows bit for bit independent of
  the rest of the launch at every entry of its tile table (10 to 1280 rows).
* The calls the fast loop does not serve (another step count) still run.

Needs a CUDA GPU and the real checkpoint::

    python -m pytest tests/diffusion/models/pi05/test_pi05_fast_path.py -v -s

``PI05_PARITY_MODEL_PATH`` points at a local checkpoint to skip the HF download.
"""

from __future__ import annotations

import dataclasses
import gc
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.fast_pi05 import LOOP_TILES, FastDenoiser, loop_tiles
from vllm_omni.diffusion.models.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "H100"}),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The fast path needs a CUDA GPU."),
]

MODEL_PATH = os.environ.get("PI05_PARITY_MODEL_PATH", "lerobot/pi05_base")
DEPLOY_CONFIG = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "pi05_low_latency.yaml"
# The graphs are captured for 1, 2, 4 and 8 requests; 3 and 6 run on the next larger ones.
MAX_BATCH = 8
CAPTURED = (1, 2, 4, 8)
BATCHES = (2, 3, 4, 6, 8)
# One more than a batch holds: the last one replaces a row in the contamination test.
NUM_REQUESTS = MAX_BATCH + 1
# Launches that reach the rest of the fast loop's tile table (requests of 10 tokens).
LARGE_LAUNCHES = (16, 32, 64, 128)

PROMPTS = (
    "turn on the stove",
    "open the middle drawer of the cabinet",
    "pick up the red block and place it in the bin",
    "put both the alphabet soup and the tomato sauce in the basket",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
)

# How far a fast result may be from the baseline's on the same inputs: both
# are bfloat16 and differ in rounding points and reduction order only. About
# twice the worst measured on an A100 (in the comments), far below what a
# wrong mask, row or weight gives.
# Relative L2 of one component's output. The image features and the K/V are
# bfloat16 values up to 146 and 84, a few units in the last place apart at
# single elements, so they have no absolute bound.
VISION_TOLERANCE = 1e-2  # image features: 4.7e-3
PREFIX_TOLERANCE = 4e-2  # K/V of all 18 layers: 9.8e-3 / 1.9e-2; final hidden state: 1.6e-2
# (relative L2, largest absolute difference) of an action chunk.
LOOP_TOLERANCE = (5e-3, 1.5e-2)  # the denoising loop over one cache: 2.1e-3, 6.6e-3
REQUEST_TOLERANCE = (
    3e-2,
    6e-2,
)  # a whole request: 1.6e-2, 3.2e-2; a batch row against its request alone: 1.3e-2, 2.5e-2


def _resolve_checkpoint_dir() -> str:
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_PATH, repo_type="model")


def _robot_obs(config, index: int) -> dict:
    """Request ``index``: 2 or 1 cameras, its own frames, state and instruction."""
    rng = np.random.default_rng(index)
    num_views = len(config.image_feature_keys) - index % 2
    images = {
        key: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for key in config.image_feature_keys[:num_views]
    }
    state = rng.uniform(-1.0, 1.0, size=config.state_dim).astype(np.float32)
    return {"images": images, "state": state, "prompt": PROMPTS[index % len(PROMPTS)]}


class _Request:
    """One request's model inputs and the fixed noise it denoises from."""

    def __init__(self, pipeline, index: int):
        model = pipeline.model
        self.index = index
        self.inputs = pipeline.processor.build_model_inputs(_robot_obs(pipeline.config, index))
        device = self.inputs[2].device
        self.noise = torch.randn(
            1,
            model.action_horizon,
            model.action_dim,
            dtype=torch.float32,
            device=device,
            generator=torch.Generator(device=device).manual_seed(1000 + index),
        )


def _batch(requests: list[_Request]) -> dict:
    """One ``sample_actions`` call over the rows of ``requests``."""
    parts = [request.inputs for request in requests]
    return dict(
        images=[torch.cat([part[0][cam] for part in parts]) for cam in range(len(parts[0][0]))],
        image_masks=[torch.cat([part[1][cam] for part in parts]) for cam in range(len(parts[0][1]))],
        lang_tokens=torch.cat([part[2] for part in parts]),
        lang_masks=torch.cat([part[3] for part in parts]),
        noise=torch.cat([request.noise for request in requests]),
    )


def _sample(model, inputs: dict, **kwargs) -> torch.Tensor:
    with torch.inference_mode():
        return model.sample_actions(**inputs, **kwargs).cpu()


def _distance(mine: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    """``(relative L2, largest absolute difference)`` of ``mine`` from ``other``."""
    mine, other = mine.double(), other.double()
    return ((mine - other).norm() / other.norm()).item(), (mine - other).abs().max().item()


def _assert_within(mine: torch.Tensor, other: torch.Tensor, tolerance: float | tuple[float, float], label: str) -> None:
    """``mine`` is within ``tolerance`` of ``other``: a relative L2, or that and a largest absolute difference."""
    assert torch.isfinite(mine).all(), f"{label}: not finite"
    rel, abs_diff = _distance(mine, other)
    print(f"\n{label}: rel L2 {rel:.3e}, max |diff| {abs_diff:.3e} (|reference| max {other.abs().max().item():.3f})")
    max_rel, max_abs = tolerance if isinstance(tolerance, tuple) else (tolerance, float("inf"))
    assert rel <= max_rel and abs_diff <= max_abs, (
        f"{label}: rel L2 {rel:.3e}, max |diff| {abs_diff:.3e}; allowed {max_rel:.1e}, {max_abs:.1e}"
    )


@pytest.fixture(scope="module")
def pipeline():
    (stage,) = yaml.safe_load(DEPLOY_CONFIG.read_text())["stages"]
    assert stage["custom_pipeline_args"]["pi05_fast_path"] is True
    od_config = OmniDiffusionConfig(
        model=_resolve_checkpoint_dir(),
        dtype="bfloat16",
        model_config=stage["model_config"],
        enforce_eager=False,
        max_num_seqs=MAX_BATCH,
        custom_pipeline_args=stage["custom_pipeline_args"],
    )
    # Constructed the way DiffusersLoader does it: under the serving dtype as
    # torch's default dtype, which is also when the CUDA graphs are captured.
    with set_default_torch_dtype(od_config.dtype):
        pipe = Pi05Pipeline(od_config=od_config)
    yield pipe
    del pipe.model
    gc.collect()
    torch.accelerator.empty_cache()


@contextmanager
def _path(model, *, fast: bool, graphs: bool = False):
    """Run ``model`` without its graphs and, with ``fast`` unset, without the
    fast hooks (the fused kernels the fast path replaces); restore both after."""
    backbone = model.paligemma_with_expert
    served = (model.cuda_graphs, model.fast_denoiser, backbone.fast_prefix, backbone.fast_vision)
    if not graphs:
        model.cuda_graphs = None
    if not fast:
        model.fast_denoiser = backbone.fast_prefix = backbone.fast_vision = None
    try:
        yield
    finally:
        model.cuda_graphs, model.fast_denoiser, backbone.fast_prefix, backbone.fast_vision = served


@pytest.fixture(scope="module")
def requests(pipeline) -> list[_Request]:
    return [_Request(pipeline, index) for index in range(NUM_REQUESTS)]


@pytest.fixture(scope="module")
def alone(pipeline, requests) -> dict[int, torch.Tensor]:
    """Every request's chunk alone on the served fast path."""
    return {request.index: _sample(pipeline.model, _batch([request])) for request in requests}


@pytest.fixture(scope="module")
def baseline(pipeline, requests) -> dict[int, torch.Tensor]:
    """Every request's chunk alone on the fused kernels the fast path replaces."""
    with _path(pipeline.model, fast=False):
        return {request.index: _sample(pipeline.model, _batch([request])) for request in requests}


# ── Installation ─────────────────────────────────────────────────────


def test_fast_path_is_installed(pipeline):
    model = pipeline.model
    backbone = model.paligemma_with_expert
    assert model.cuda_graphs is not None
    assert backbone.fast_vision is not None and backbone.fast_prefix is not None
    assert model.fast_denoiser is not None and model.fast_denoiser.max_batch == MAX_BATCH
    assert model.fast_denoiser.tiles is None, "serving picks the tiles of each launch from LOOP_TILES"


# ── Components ───────────────────────────────────────────────────────


@pytest.mark.parametrize("batch", [1, MAX_BATCH], ids=lambda batch: f"batch={batch}")
def test_fast_vision_matches_the_vision_tower(pipeline, requests, batch):
    model = pipeline.model
    backbone = model.paligemma_with_expert
    inputs = _batch(requests[:batch])
    pixels = torch.cat(inputs["images"])
    with torch.inference_mode():
        fast = backbone.embed_image(pixels)
        with _path(model, fast=False):
            reference = backbone.embed_image(pixels)
    assert fast.shape == reference.shape and fast.dtype == reference.dtype
    _assert_within(fast.cpu(), reference.cpu(), VISION_TOLERANCE, f"image features, batch={batch}")


def _prefix_inputs(model, inputs: dict):
    """The prefix forward's arguments for ``inputs``, embedded on the baseline."""
    with _path(model, fast=False):
        embs, pad_masks, att_masks = model.embed_prefix(
            inputs["images"], inputs["image_masks"], inputs["lang_tokens"], inputs["lang_masks"]
        )
    mask = prepare_attention_masks_4d(make_att_2d_masks(pad_masks, att_masks))
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    rotary = model.prepare_prefix_rotary(embs, position_ids)
    return embs, pad_masks, mask, position_ids, rotary


def _prefix_forward(model, embs, mask, position_ids, rotary, *, kv_only: bool):
    cache = model.new_kv_cache(embs.shape[0], embs.shape[1])
    (hidden, _), _ = model.paligemma_with_expert.forward(
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=cache,
        inputs_embeds=[embs, None],
        use_cache=True,
        prefix_kv_only=kv_only,
        rotary_emb=rotary,
    )
    return hidden, cache


@pytest.mark.parametrize("kv_only", [True, False], ids=["kv_only", "hidden"])
@pytest.mark.parametrize("batch", [1, MAX_BATCH], ids=lambda batch: f"batch={batch}")
def test_fast_prefix_matches_the_fused_prefix_forward(pipeline, requests, batch, kv_only):
    """The K/V every layer leaves in the cache, with rows of 1 and 2 cameras
    and of different instruction lengths (masked keys); the hidden state when
    the caller asks for it."""
    model = pipeline.model
    with torch.inference_mode():
        embs, _, mask, position_ids, rotary = _prefix_inputs(model, _batch(requests[:batch]))
        with _path(model, fast=True):
            hidden, cache = _prefix_forward(model, embs, mask, position_ids, rotary, kv_only=kv_only)
        with _path(model, fast=False):
            ref_hidden, ref_cache = _prefix_forward(model, embs, mask, position_ids, rotary, kv_only=kv_only)
    prefix = slice(0, embs.shape[1])
    label = f"batch={batch}, {'kv only' if kv_only else 'with hidden state'}"
    for name, mine, other in (("K", cache.key, ref_cache.key), ("V", cache.value, ref_cache.value)):
        mine, other = mine[:, :, :, prefix].float().cpu(), other[:, :, :, prefix].float().cpu()
        _assert_within(mine, other, PREFIX_TOLERANCE, f"prefix {name}, {label}")
    if kv_only:
        assert hidden is None and ref_hidden is None
    else:
        assert hidden.shape == ref_hidden.shape and hidden.dtype == ref_hidden.dtype
        _assert_within(hidden.float().cpu(), ref_hidden.float().cpu(), PREFIX_TOLERANCE, f"hidden, {label}")


def _fused_loop(model, pad_masks, cache, context, noise) -> torch.Tensor:
    """The denoising loop the fast loop replaces: one fused step per Euler step."""
    table = model.runtime_step_table(model.num_inference_steps, noise.shape[0])
    x_t, dt = noise.clone(), -1.0 / len(table)
    for row in table:
        v_t = model.denoise_step(
            prefix_pad_masks=pad_masks,
            past_key_values=cache,
            x_t=x_t,
            timestep=row.timestep,
            context=context,
            step_data=row,
        )
        x_t = x_t + dt * v_t
    return x_t


def _reversed(model, cache, context, noise):
    """The same launch with its requests in reverse order."""
    batch = noise.shape[0]
    order = torch.arange(batch - 1, -1, -1, device=noise.device)
    flipped = model.new_kv_cache(batch, cache.prefix_len)
    flipped.key.copy_(cache.key[:, order])
    flipped.value.copy_(cache.value[:, order])
    fields = {field.name: getattr(context, field.name) for field in dataclasses.fields(context)}
    assert all(value.shape[0] == batch for value in fields.values())
    return flipped, dataclasses.replace(context, **{name: value[order] for name, value in fields.items()}), noise[order]


@pytest.mark.parametrize("batch", [1, *BATCHES], ids=lambda batch: f"batch={batch}")
def test_fast_loop_matches_the_fused_loop_over_one_cache(pipeline, requests, batch):
    """Both loops denoise the same noise over the same prefix K/V. A row of the
    fast loop does not depend on the other rows of its launch: the requests in
    reverse order give the same chunks bit for bit."""
    model = pipeline.model
    inputs = _batch(requests[:batch])
    with torch.inference_mode(), _path(model, fast=False):
        embs, pad_masks, mask, position_ids, rotary = _prefix_inputs(model, inputs)
        _, cache = _prefix_forward(model, embs, mask, position_ids, rotary, kv_only=True)
        context = model.prepare_denoise_context(pad_masks)
        reference = _fused_loop(model, pad_masks, cache, context, inputs["noise"])
    with torch.inference_mode(), _path(model, fast=True):
        fast = model.fast_denoise(inputs["noise"], cache, context)
        flipped_cache, flipped_context, flipped_noise = _reversed(model, cache, context, inputs["noise"])
        flipped = model.fast_denoise(flipped_noise, flipped_cache, flipped_context)
    assert fast.shape == reference.shape == (batch, model.action_horizon, model.action_dim)
    _assert_within(fast.cpu(), reference.cpu(), LOOP_TOLERANCE, f"denoising loop, batch={batch}")
    assert torch.equal(flipped.flip(0), fast), (
        f"batch={batch}: reversing the requests moved a chunk by up to "
        f"{(flipped.flip(0).double() - fast.double()).abs().max().item():.3e}"
    )


@pytest.fixture(scope="module")
def large_launches(pipeline, requests):
    """A fast loop for launches up to the largest in ``LARGE_LAUNCHES``, and the
    prefix K/V, context and masks of ``MAX_BATCH`` requests to repeat into them."""
    model = pipeline.model
    with torch.inference_mode(), _path(model, fast=False):
        embs, pad_masks, mask, position_ids, rotary = _prefix_inputs(model, _batch(requests[:MAX_BATCH]))
        _, cache = _prefix_forward(model, embs, mask, position_ids, rotary, kv_only=True)
    denoiser = FastDenoiser(model, max_batch=max(LARGE_LAUNCHES))
    yield denoiser, cache, pad_masks
    del denoiser
    gc.collect()
    torch.accelerator.empty_cache()


@pytest.mark.parametrize("batch", LARGE_LAUNCHES, ids=lambda batch: f"batch={batch}")
def test_fast_loop_serves_large_launches(pipeline, requests, large_launches, batch):
    """Launches of 16 to 128 requests run the remaining entries of the tile
    table. The launch repeats the same ``MAX_BATCH`` requests: against the
    fused loop within ``LOOP_TOLERANCE``, and every repeat of a request is its
    first copy bit for bit."""
    model = pipeline.model
    denoiser, cache, pad_masks = large_launches
    repeats = batch // MAX_BATCH
    assert loop_tiles(batch * model.action_horizon) is not loop_tiles(MAX_BATCH * model.action_horizon)
    with torch.inference_mode(), _path(model, fast=False):
        big = model.new_kv_cache(batch, cache.prefix_len)
        big.key.copy_(cache.key.repeat(1, repeats, 1, 1, 1))
        big.value.copy_(cache.value.repeat(1, repeats, 1, 1, 1))
        pad = pad_masks.repeat(repeats, 1)
        context = model.prepare_denoise_context(pad)
        noise = torch.cat([request.noise for request in requests[:MAX_BATCH]]).repeat(repeats, 1, 1)
        reference = _fused_loop(model, pad, big, context, noise)
        fast = denoiser.run(noise.clone(), big, context)
    _assert_within(fast.cpu(), reference.cpu(), LOOP_TOLERANCE, f"denoising loop, batch={batch}")
    assert torch.equal(fast, fast[:MAX_BATCH].repeat(repeats, 1, 1)), (
        f"batch={batch}: a repeated request differs from its first copy by up to "
        f"{(fast.double() - fast[:MAX_BATCH].repeat(repeats, 1, 1).double()).abs().max().item():.3e}"
    )


def test_tile_table_entries_are_all_run():
    """The launches above cover every entry of ``LOOP_TILES`` (10-token chunks)."""
    launches = [1, *BATCHES, *LARGE_LAUNCHES]
    covered = {id(loop_tiles(batch * 10)) for batch in launches}
    assert covered == {id(tiles) for _, tiles in LOOP_TILES}


# ── Whole requests ───────────────────────────────────────────────────


def test_request_graph_serves_requests_bit_exact(pipeline, requests, alone):
    """A request replays one graph, and capture and replay change no
    computation: the chunk is the one the fast kernels give eagerly."""
    model = pipeline.model
    replays_before = model.cuda_graphs.num_replays.copy()
    served = {request.index: _sample(model, _batch([request])) for request in requests}
    replays = model.cuda_graphs.num_replays - replays_before
    assert replays["request"] == len(requests), f"expected one request-graph replay per request, got {dict(replays)}"
    with _path(model, fast=True):
        eager = {request.index: _sample(model, _batch([request])) for request in requests}
    for request in requests:
        assert torch.equal(served[request.index], alone[request.index]), "the request graph is not deterministic"
        assert torch.equal(served[request.index], eager[request.index]), (
            f"request {request.index}: the request graph differs from its kernels run eagerly by up to "
            f"{(served[request.index].double() - eager[request.index].double()).abs().max().item():.3e}"
        )


def test_fast_path_matches_the_baseline(pipeline, requests, alone, baseline):
    """The fast kernels round and reduce in another order than the fused ones;
    the chunk stays within bfloat16's own noise of the baseline's."""
    assert alone[0].shape == (1, pipeline.config.chunk_size, pipeline.config.max_action_dim)
    chunks = torch.stack([alone[request.index] for request in requests])
    assert chunks.unique(dim=0).shape[0] == len(requests), "the requests should give different chunks"
    for request in requests:
        _assert_within(
            alone[request.index], baseline[request.index], REQUEST_TOLERANCE, f"request {request.index} vs baseline"
        )


def test_other_step_count_runs_off_the_fast_loop(pipeline, requests):
    """The fast loop runs the prepared schedule only: a per-request step count
    keeps the fast vision tower and prefix forward and denoises step by step."""
    model = pipeline.model
    inputs = _batch(requests[:1])
    replays_before = model.cuda_graphs.num_replays.copy()
    actual = _sample(model, inputs, num_steps=4)
    replays = model.cuda_graphs.num_replays - replays_before
    assert replays["request"] == 0 and replays["denoise_all"] == 0, dict(replays)
    with _path(model, fast=False):
        reference = _sample(model, inputs, num_steps=4)
    _assert_within(actual, reference, REQUEST_TOLERANCE, "4 steps vs baseline")


# ── Formed batches ───────────────────────────────────────────────────


@pytest.mark.parametrize("size", BATCHES, ids=lambda size: f"batch={size}")
def test_formed_batch_rows_match_the_requests_alone(pipeline, requests, alone, size):
    """A batch of a captured size replays its request graph; one between two
    sizes runs its loop on the next larger graph. Each row is its request's
    chunk within bfloat16 rounding (the prefix GEMMs and the loop's tiles of
    a batch round differently than a request's alone)."""
    model = pipeline.model
    rows = requests[:size]
    replays_before = model.cuda_graphs.num_replays.copy()
    actual = _sample(model, _batch(rows))
    replays = model.cuda_graphs.num_replays - replays_before
    assert replays["request" if size in CAPTURED else "denoise_all"] == 1, dict(replays)
    assert actual.shape == (size, pipeline.config.chunk_size, pipeline.config.max_action_dim)
    for position, request in enumerate(rows):
        _assert_within(
            actual[position : position + 1],
            alone[request.index],
            REQUEST_TOLERANCE,
            f"batch={size} row {position} (request {request.index}) vs alone",
        )


@pytest.mark.parametrize("size", BATCHES, ids=lambda size: f"batch={size}")
def test_replacing_one_row_leaves_the_others_bit_identical(pipeline, requests, alone, size):
    """No cross-request contamination on the fast path."""
    model = pipeline.model
    rows = requests[:size]
    before = _sample(model, _batch(rows))
    replaced = size // 2
    substitute = requests[-1]  # 1 camera; not among the rows of any batch under test
    rows_after = [substitute if position == replaced else request for position, request in enumerate(rows)]
    after = _sample(model, _batch(rows_after))
    others = [position for position in range(size) if position != replaced]
    assert torch.equal(after[others], before[others]), (
        f"batch={size}: replacing row {replaced} changed other rows by up to "
        f"{(after[others].double() - before[others].double()).abs().max().item():.3e}"
    )
    _assert_within(
        after[replaced : replaced + 1], alone[substitute.index], REQUEST_TOLERANCE, f"batch={size} new row vs alone"
    )
