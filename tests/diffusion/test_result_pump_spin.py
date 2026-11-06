# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The result pump's shm reader must sleep, not spin, while it waits for a reply."""

import time
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.executor import multiproc_executor
from vllm_omni.diffusion.executor.multiproc_executor import _relax_reader_spin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Condition:
    def __init__(self, *, is_reader=True, busy_loop_s=1.0):
        self.is_reader = is_reader
        self.busy_loop_s = busy_loop_s
        self.last_read = time.monotonic()
        self.idle_waits = []

    def wait(self, timeout_ms=None):
        self.idle_waits.append(timeout_ms)


def test_busy_window_sleeps_instead_of_delegating(monkeypatch):
    condition = _Condition()
    sleeps = []
    monkeypatch.setattr(multiproc_executor.time, "sleep", sleeps.append)

    assert _relax_reader_spin(SimpleNamespace(_spin_condition=condition), 50e-6) is True
    condition.wait(timeout_ms=1000)

    assert sleeps == [50e-6]
    assert condition.idle_waits == []


def test_idle_reader_keeps_the_original_wait(monkeypatch):
    condition = _Condition()
    condition.last_read = time.monotonic() - 10.0
    sleeps = []
    monkeypatch.setattr(multiproc_executor.time, "sleep", sleeps.append)

    assert _relax_reader_spin(SimpleNamespace(_spin_condition=condition), 50e-6) is True
    condition.wait(timeout_ms=1000)

    assert sleeps == []
    assert condition.idle_waits == [1000]


@pytest.mark.parametrize(
    "queue",
    [
        SimpleNamespace(),
        SimpleNamespace(_spin_condition=None),
        SimpleNamespace(_spin_condition=_Condition(is_reader=False)),
        SimpleNamespace(_spin_condition=SimpleNamespace(is_reader=True)),
    ],
)
def test_unknown_reader_layout_is_left_alone(queue):
    assert _relax_reader_spin(queue, 50e-6) is False


def test_zero_pause_keeps_the_busy_loop():
    condition = _Condition()
    original = condition.wait

    assert _relax_reader_spin(SimpleNamespace(_spin_condition=condition), 0.0) is False
    assert condition.wait == original
