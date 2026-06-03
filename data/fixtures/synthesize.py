"""Generate deterministic synthetic fixtures for schicluster-rs parity tests.

Run under any env with numpy + scipy. Output is bit-identical across envs
because every RNG call is seeded and every op is float32.

Usage:
    python data/fixtures/synthesize.py
"""
from __future__ import annotations

import pathlib

import numpy as np
from scipy.ndimage import convolve


FIXTURE_DIR = pathlib.Path(__file__).resolve().parent


def conv_small_fixture(seed: int = 42, n: int = 64, pad: int = 3, gap: int = 1) -> dict:
    """64x64 f32 input + 7x7 donut-shaped f32 kernel + scipy mirror-convolve reference.

    The kernel matches the shape of the "donut" mask used in loop_bkg.py
    so this fixture exercises the same code path the loop module will hit.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)).astype(np.float32)

    w = pad * 2 + 1
    k = np.ones((w, w), dtype=np.float32)
    k[(pad - gap):(pad + gap + 1), (pad - gap):(pad + gap + 1)] = 0.0
    k = k / k.sum()

    ref = convolve(a, k, mode="mirror").astype(np.float32)
    return {"input": a, "kernel": k, "convolved": ref}


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture = conv_small_fixture()
    np.savez(FIXTURE_DIR / "conv_small.npz", **fixture)
    print(f"wrote {FIXTURE_DIR / 'conv_small.npz'}")
    print(f"  input.shape    = {fixture['input'].shape}, dtype = {fixture['input'].dtype}")
    print(f"  kernel.shape   = {fixture['kernel'].shape}, sum  = {fixture['kernel'].sum():.6f}")
    print(f"  convolved.shape= {fixture['convolved'].shape}, mean = {fixture['convolved'].mean():.6e}")


if __name__ == "__main__":
    main()
