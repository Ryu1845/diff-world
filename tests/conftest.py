import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.bench_analysis import synth_speech  # noqa: E402

FIXTURES = ["test_tone.wav", "test_complex.wav"]
# Synthetic stand-ins when the (gitignored) wav fixtures are absent. Kept at
# <= 3.1 s: DIO's 1e-9 parity bound is FFT round-off, which grows with length.
SYNTHETIC = {"synth_1.0s": (1.0, 0), "synth_3.1s": (3.1, 3)}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def load(name: str) -> tuple[np.ndarray, int]:
    if name in SYNTHETIC:
        seconds, seed = SYNTHETIC[name]
        return synth_speech(seconds, seed), 22050
    x, fs = sf.read(ROOT / name)
    if x.ndim > 1:
        x = x[:, 0]
    return np.ascontiguousarray(x, dtype=np.float64), int(fs)


@pytest.fixture(params=[f for f in FIXTURES if (ROOT / f).exists()] + list(SYNTHETIC))
def audio(request):
    return load(request.param)


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)
