"""Numerical parity of the native WORLD analysis against the C reference.

`pyworld` is a dev-only dependency; the suite skips this module when it is
absent. CheapTrick and D4C tolerances are deliberately loose: WORLD dithers its
windowed waveform and its smoothed power spectrum with a serial xorshift128
PRNG that a frame-vectorized port cannot reproduce. The residual is the
algorithm's own sensitivity to that dither (two different seeds differ from
each other by the same magnitude as each differs from `pyworld`), so tightening
these toward 1e-12 would pin the RNG stream and fail on any torch version whose
`torch.rand` sequence differs.
"""

import numpy as np
import pytest
import torch

pyworld = pytest.importorskip("pyworld")

from benchmarks.bench_analysis import synth_speech  # noqa: E402
from diff_world.analysis import world_analyze_batch  # noqa: E402
from diff_world.world import cheaptrick, d4c, dio, stonemask  # noqa: E402
from diff_world.world._common import clamp_gather  # noqa: E402
from diff_world.world.d4c import _residual_energy_db  # noqa: E402

FFT_SIZE = 1024
FS = 22050


def reference(x, fs):
    f0, t = pyworld.dio(x, fs)
    f0 = pyworld.stonemask(x, f0, t, fs)
    sp = pyworld.cheaptrick(x, f0, t, fs, fft_size=FFT_SIZE)
    ap = pyworld.d4c(x, f0, t, fs, fft_size=FFT_SIZE)
    return f0, t, sp, ap


def assert_sp_close(sp, ref):
    assert sp.shape == ref.shape
    db = np.abs(10 * np.log10(np.maximum(sp, 1e-300) / np.maximum(ref, 1e-300)))
    assert db.max() < 0.05  # measured worst case: 0.0014 dB


def assert_ap_close(ap, ref):
    assert ap.shape == ref.shape
    # Love-train gating is a discrete decision and must agree exactly.
    gated = np.all(ap == 1.0 - 1e-12, axis=1)
    ref_gated = np.all(ref == 1.0 - 1e-12, axis=1)
    assert np.array_equal(gated, ref_gated)
    assert np.abs(ap - ref).max() < 0.15  # measured worst case: 0.066


def test_dio_matches_reference(audio, device):
    x, fs = audio
    f0, t = dio(torch.as_tensor(x, device=device), fs)
    f0, t = f0.cpu().numpy(), t.cpu().numpy()
    ref_f0, ref_t = pyworld.dio(x, fs)

    assert np.array_equal(t, ref_t)
    assert np.array_equal(f0 > 0, ref_f0 > 0)
    assert np.abs(f0 - ref_f0).max() < 1e-9


def test_stonemask_matches_reference(audio, device):
    x, fs = audio
    xt = torch.as_tensor(x, device=device)
    f0, t = dio(xt, fs)
    ref_f0, ref_t = pyworld.dio(x, fs)

    refined = stonemask(xt, f0, t, fs).cpu().numpy()
    ref = pyworld.stonemask(x, ref_f0, ref_t, fs)
    assert np.abs(refined - ref).max() < 1e-9


def test_cheaptrick_matches_reference(audio, device):
    x, fs = audio
    xt = torch.as_tensor(x, device=device)
    f0, t = dio(xt, fs)
    f0 = stonemask(xt, f0, t, fs)
    ref_f0, ref_t = pyworld.dio(x, fs)
    ref_f0 = pyworld.stonemask(x, ref_f0, ref_t, fs)

    sp = cheaptrick(xt, f0, t, fs, fft_size=FFT_SIZE).cpu().numpy()
    assert_sp_close(sp, pyworld.cheaptrick(x, ref_f0, ref_t, fs, fft_size=FFT_SIZE))


def test_d4c_matches_reference(audio, device):
    x, fs = audio
    xt = torch.as_tensor(x, device=device)
    f0, t = dio(xt, fs)
    f0 = stonemask(xt, f0, t, fs)
    ref_f0, ref_t = pyworld.dio(x, fs)
    ref_f0 = pyworld.stonemask(x, ref_f0, ref_t, fs)

    ap = d4c(xt, f0, t, fs, fft_size=FFT_SIZE).cpu().numpy()
    assert_ap_close(ap, pyworld.d4c(x, ref_f0, ref_t, fs, fft_size=FFT_SIZE))


# Lengths straddle DIO's FFT-size buckets: 1.0 s -> 2^15, 2.5 s and 2.9 s ->
# 2^16 (two ragged rows in one bucket), 3.1 s -> 2^17; 0.3 s is shorter than
# the contour fixer's minimum and must come back all-unvoiced.
RAGGED = [(1.0, 10), (2.5, 11), (2.9, 12), (3.1, 13), (0.3, 14)]


def test_ragged_batch_matches_reference(device):
    xs = [synth_speech(s, seed) for s, seed in RAGGED]
    out = world_analyze_batch(xs, FS, FFT_SIZE, device=device)
    assert len(out) == len(xs)
    for x, o in zip(xs, out):
        o = {k: v.cpu().numpy() for k, v in o.items()}
        f0, t, sp, ap = reference(x, FS)
        assert np.array_equal(o["t"], t)
        assert np.array_equal(o["f0"] > 0, f0 > 0)
        assert np.abs(o["f0"] - f0).max() < 1e-9
        assert_sp_close(o["sp"], sp)
        assert_ap_close(o["ap"], ap)


def test_batched_dio_matches_per_utterance(device):
    xs = [synth_speech(s, seed) for s, seed in RAGGED]
    lengths = [len(x) for x in xs]
    padded = torch.zeros(len(xs), max(lengths), dtype=torch.float64, device=device)
    for i, x in enumerate(xs):
        padded[i, : len(x)] = torch.as_tensor(x)
        padded[i, len(x) :] = 1.0  # garbage past `lengths` must be ignored
    f0, t = dio(padded, FS, lengths=lengths)
    for i, x in enumerate(xs):
        f1, t1 = dio(torch.as_tensor(x, device=device), FS)
        n = f1.numel()
        assert torch.equal(t[:n], t1)
        assert torch.equal(f0[i, :n] > 0, f1 > 0)
        assert (f0[i, :n] - f1).abs().max() < 1e-9
        assert not f0[i, n:].any()


def test_equal_length_tensor_batch(device):
    x = torch.stack([torch.as_tensor(synth_speech(2.0, s)) for s in (20, 21, 22)]).to(device)
    from_tensor = world_analyze_batch(x, FS, FFT_SIZE)
    from_list = world_analyze_batch(list(x), FS, FFT_SIZE)
    for a, b in zip(from_tensor, from_list):
        for k in a:
            assert torch.equal(a[k], b[k])


def test_clamp_gather_replicates_each_utterance_edge(device):
    x = torch.arange(10.0, device=device)  # utterances [0..3] and [4..9]
    idx = torch.tensor([[-2, 0, 3, 5], [-1, 0, 5, 9]], device=device)
    bounds = (torch.tensor([0, 4], device=device), torch.tensor([4, 6], device=device))
    got = clamp_gather(x, idx, bounds)
    assert got.tolist() == [[0, 0, 3, 3], [4, 4, 9, 9]]


def test_residual_energy_matches_sorted_cumsum():
    pp = torch.rand(64, 1025, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) ** 8
    b = 29
    cs = torch.cumsum(torch.sort(pp, dim=-1).values, dim=-1)
    upstream = 10 * torch.log10(cs[:, 1024 - b - 1] / cs[:, 1024])
    assert (upstream - _residual_energy_db(pp, b)).abs().max() < 1e-9
