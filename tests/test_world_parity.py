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

import pathlib

import numpy as np
import pytest
import soundfile as sf
import torch

pyworld = pytest.importorskip("pyworld")

from diff_world.world import cheaptrick, d4c, dio, stonemask  # noqa: E402

FIXTURES = ["test_tone.wav", "test_complex.wav"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
FFT_SIZE = 1024


@pytest.fixture(params=FIXTURES)
def audio(request):
    path = ROOT / request.param
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    return np.ascontiguousarray(x, dtype=np.float64), int(fs)


def test_dio_matches_reference(audio):
    x, fs = audio
    f0, t = dio(torch.as_tensor(x), fs)
    ref_f0, ref_t = pyworld.dio(x, fs)

    assert np.array_equal(t.numpy(), ref_t)
    assert np.array_equal(f0.numpy() > 0, ref_f0 > 0)
    assert np.abs(f0.numpy() - ref_f0).max() < 1e-9


def test_stonemask_matches_reference(audio):
    x, fs = audio
    xt = torch.as_tensor(x)
    f0, t = dio(xt, fs)
    ref_f0, ref_t = pyworld.dio(x, fs)

    refined = stonemask(xt, f0, t, fs).numpy()
    ref = pyworld.stonemask(x, ref_f0, ref_t, fs)
    assert np.abs(refined - ref).max() < 1e-9


def test_cheaptrick_matches_reference(audio):
    x, fs = audio
    xt = torch.as_tensor(x)
    f0, t = dio(xt, fs)
    f0 = stonemask(xt, f0, t, fs)
    ref_f0, ref_t = pyworld.dio(x, fs)
    ref_f0 = pyworld.stonemask(x, ref_f0, ref_t, fs)

    sp = cheaptrick(xt, f0, t, fs, fft_size=FFT_SIZE).numpy()
    ref = pyworld.cheaptrick(x, ref_f0, ref_t, fs, fft_size=FFT_SIZE)
    assert sp.shape == ref.shape

    db = np.abs(10 * np.log10(np.maximum(sp, 1e-300) / np.maximum(ref, 1e-300)))
    assert db.max() < 0.05  # measured worst case: 0.0014 dB


def test_d4c_matches_reference(audio):
    x, fs = audio
    xt = torch.as_tensor(x)
    f0, t = dio(xt, fs)
    f0 = stonemask(xt, f0, t, fs)
    ref_f0, ref_t = pyworld.dio(x, fs)
    ref_f0 = pyworld.stonemask(x, ref_f0, ref_t, fs)

    ap = d4c(xt, f0, t, fs, fft_size=FFT_SIZE).numpy()
    ref = pyworld.d4c(x, ref_f0, ref_t, fs, fft_size=FFT_SIZE)
    assert ap.shape == ref.shape

    # Love-train gating is a discrete decision and must agree exactly.
    gated = np.all(ap == 1.0 - 1e-12, axis=1)
    ref_gated = np.all(ref == 1.0 - 1e-12, axis=1)
    assert np.array_equal(gated, ref_gated)

    assert np.abs(ap - ref).max() < 0.15  # measured worst case: 0.066
