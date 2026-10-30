"""Throughput benchmark for the native WORLD analysis.

Usage:
    uv run python benchmarks/bench_analysis.py [--device cuda] [--wav-dir DIR]

Reports audio-seconds analyzed per wall-second for a per-utterance loop over
the stage functions and, when available, for `world_analyze_batch`. It also
reports per-utterance counts of profiled aten ops (a device-independent proxy
for CUDA kernel launches) and of host-syncing ops (`_local_scalar_dense`, i.e.
.item()/bool(); `nonzero`, i.e. boolean indexing; `unique`). `.tolist()` on a
CUDA tensor also syncs but is invisible to the CPU profiler.

On CPU, large batches are dominated by page faults from glibc mmap'ing every
large temporary; compare with
    MALLOC_MMAP_THRESHOLD_=34359738368 MALLOC_TRIM_THRESHOLD_=34359738368
"""

import argparse
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from diff_world.world import cheaptrick, d4c, dio, stonemask  # noqa: E402

try:
    from diff_world.analysis import world_analyze_batch
except ImportError:  # pre-batching revisions
    world_analyze_batch = None

FS = 22050
SYNC_OPS = ("aten::_local_scalar_dense", "aten::nonzero", "aten::_unique2", "aten::unique_dim")


def synth_speech(seconds: float, seed: int, fs: int = FS) -> np.ndarray:
    """Speech-like test signal: voiced harmonic segments with vibrato and a
    formant envelope, shaped-noise bursts, and pauses, over a -80 dB noise
    floor. Without the floor, DIO's band signals in digital silence are pure
    FFT round-off and any two FFT implementations disagree there."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        kind = rng.choice(["voiced", "noise", "silence"], p=[0.6, 0.25, 0.15])
        seg = min(int(rng.uniform(0.08, 0.5) * fs), n - pos)
        tt = np.arange(seg) / fs
        if kind == "voiced":
            base = rng.uniform(90.0, 320.0)
            f0 = base * (1 + 0.03 * np.sin(2 * np.pi * 5.5 * tt)) * (1 + rng.uniform(-0.2, 0.2) * tt)
            phi = 2 * np.pi * np.cumsum(f0) / fs
            formants = rng.uniform([500, 1200, 2500], [900, 2200, 3500])
            y = np.zeros(seg)
            for k in range(1, int(fs / 2 / f0.max())):
                fk = k * f0
                amp = sum(np.exp(-0.5 * ((fk - fm) / 150.0) ** 2) for fm in formants) + 0.05
                y += amp / k * np.sin(k * phi)
            y *= np.hanning(seg)
        elif kind == "noise":
            spec = np.fft.rfft(rng.standard_normal(seg))
            freqs = np.fft.rfftfreq(seg, 1 / fs)
            y = np.fft.irfft(spec * (freqs > 2000) * np.exp(-freqs / 6000), n=seg) * np.hanning(seg)
        else:
            y = np.zeros(seg)
        out[pos : pos + seg] = y
        pos += seg
    peak = np.abs(out).max()
    out = out / peak * 0.5 if peak > 0 else out
    return out + 1e-4 * rng.standard_normal(n)


def load_corpus(wav_dir: str | None, n: int, seed: int) -> list[np.ndarray]:
    if wav_dir:
        import soundfile as sf

        xs = []
        for p in sorted(pathlib.Path(wav_dir).glob("*.wav"))[:n]:
            x, fs = sf.read(p)
            assert fs == FS, f"{p}: expected {FS} Hz"
            xs.append(np.ascontiguousarray(x if x.ndim == 1 else x[:, 0], dtype=np.float64))
        return xs
    rng = np.random.default_rng(seed)
    return [synth_speech(rng.uniform(1.0, 8.0), seed + i) for i in range(n)]


def clips(n: int, seconds: float, seed: int) -> list[np.ndarray]:
    src = synth_speech(n * seconds, seed)
    m = int(seconds * FS)
    return [src[i * m : (i + 1) * m] for i in range(n)]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_loop(xs, device, stage_times=None):
    gen = torch.Generator(device=device).manual_seed(0)
    for a in xs:
        x = torch.as_tensor(a, device=device)
        t0 = time.perf_counter()
        f0, t = dio(x, FS)
        sync(device)
        t1 = time.perf_counter()
        f0 = stonemask(x, f0, t, FS)
        sync(device)
        t2 = time.perf_counter()
        cheaptrick(x, f0, t, FS, fft_size=1024, generator=gen)
        sync(device)
        t3 = time.perf_counter()
        d4c(x, f0, t, FS, fft_size=1024, generator=gen)
        sync(device)
        t4 = time.perf_counter()
        if stage_times is not None:
            for k, v in zip(("dio", "stonemask", "cheaptrick", "d4c"), (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                stage_times[k] += v


def run_batch(xs, device, batch_size):
    for i in range(0, len(xs), batch_size):
        world_analyze_batch(xs[i : i + batch_size], fs=FS, device=device)
    sync(device)


def timed(fn, repeat):
    fn()  # warm-up: caches, allocator, cuFFT plans
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def op_counts(fn, n_utt):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    events = prof.events()
    names = Counter(e.name for e in events)
    leaf = sum(1 for e in events if e.name.startswith("aten::") and not e.cpu_children)
    syncs = sum(names[s] for s in SYNC_OPS)
    return leaf / n_utt, syncs / n_utt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-profile", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"device={device} torch={torch.__version__} threads={torch.get_num_threads()}")

    configs = [(f"clips B={b} x 2s", clips(b, 2.0, 1000 + b), b) for b in (1, 8, 32)]
    corpus = load_corpus(args.wav_dir, 32, 7)
    configs.append((f"ragged n={len(corpus)}", corpus, 8))

    header = f"{'config':<20}{'mode':<8}{'audio s':>9}{'wall ms':>10}{'x RT':>9}{'ops/utt':>10}{'syncs/utt':>11}"
    print(header)
    print("-" * len(header))
    for name, xs, bs in configs:
        audio_s = sum(len(a) for a in xs) / FS
        modes = [("loop", lambda xs=xs: run_loop(xs, device))]
        if world_analyze_batch is not None:
            modes.append((f"batch{bs}", lambda xs=xs, bs=bs: run_batch(xs, device, bs)))
        for mode, fn in modes:
            wall = timed(fn, args.repeat)
            ops, syncs = (float("nan"),) * 2 if args.no_profile else op_counts(fn, len(xs))
            print(f"{name:<20}{mode:<8}{audio_s:>9.1f}{wall * 1e3:>10.1f}{audio_s / wall:>9.1f}{ops:>10.0f}{syncs:>11.1f}")

    stage = Counter()
    run_loop(corpus, device)
    run_loop(corpus, device, stage)
    total = sum(stage.values())
    print("\nper-stage share, loop over ragged corpus: " + ", ".join(
        f"{k} {v * 1e3:.0f} ms ({100 * v / total:.0f}%)" for k, v in stage.items()
    ))


if __name__ == "__main__":
    main()
