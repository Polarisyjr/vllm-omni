# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent, initialization-only Pi0.5 runtime optimization switches."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pi05RuntimeOptions:
    precompute_adarms: bool = False
    hoist_denoise_invariants: bool = False
    prefix_kv_only: bool = False
    # The fast bfloat16 kernels (``fast_pi05``) for the vision tower, the prefix
    # forward and the denoising loop, a whole request captured as one CUDA graph
    # per batch size. Needs ``precompute_adarms`` and ``hoist_denoise_invariants``.
    fast_path: bool = False

    def __post_init__(self):
        for name in ("precompute_adarms", "hoist_denoise_invariants", "prefix_kv_only", "fast_path"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"pi05_{name} must be a boolean, got {value!r}.")
        if self.fast_path and not (self.precompute_adarms and self.hoist_denoise_invariants):
            raise ValueError("pi05_fast_path needs pi05_precompute_adarms and pi05_hoist_denoise_invariants.")

    @classmethod
    def from_pipeline_args(cls, args: dict | None) -> "Pi05RuntimeOptions":
        args = args or {}
        return cls(
            precompute_adarms=args.get("pi05_precompute_adarms", False),
            hoist_denoise_invariants=args.get("pi05_hoist_denoise_invariants", False),
            prefix_kv_only=args.get("pi05_prefix_kv_only", False),
            fast_path=args.get("pi05_fast_path", False),
        )
