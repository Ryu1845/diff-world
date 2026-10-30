"""CheapTrick spectral envelope estimation, ported from WORLD's cheaptrick.cpp
(BSD-3-Clause). Fully batched: one (F, fft_size) buffer, one rfft, one irfft,
and one host sync (the f0 maximum that sizes the spectral smoothing).
"""

import math

import torch

from ._common import (
    KEPS,
    KSG,
    clamp_gather,
    dc_correction,
    dither,
    frame_windows,
    linear_smoothing,
    matlab_round,
    round_int,
)

KDEFAULT_F0 = 500.0


def get_fft_size(fs: int, f0_floor: float = 71.0) -> int:
    """GetFFTSizeForCheapTrick."""
    return int(2.0 ** (1.0 + int(math.log(3.0 * fs / f0_floor + 1) / 0.69314718055994529)))


def get_f0_floor(fs: int, fft_size: int) -> float:
    """GetF0FloorForCheapTrick."""
    return 3.0 * fs / (fft_size - 3.0)


def cheaptrick(
    x: torch.Tensor,
    f0: torch.Tensor,
    t: torch.Tensor,
    fs: int,
    fft_size: int = 1024,
    q1: float = -0.15,
    generator: torch.Generator | None = None,
    *,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Estimate spectral envelopes. Returns (F, fft_size//2+1) power spectra.

    bounds: per-frame (start, length) of each frame's utterance in x, when
    frames of several utterances share one flat x (see ``clamp_gather``).
    """
    dev = x.device
    x = x.to(torch.float64)
    f0 = f0.to(torch.float64)
    t = t.to(torch.float64)
    n_frames = f0.shape[0]

    # The floor is the RECOMPUTED 3*fs/(fft_size-3), not the caller's f0_floor.
    # In practice it fires exactly on f0 == 0 frames; there is no other
    # unvoiced special case, so unvoiced frames yield a real envelope.
    floor = get_f0_floor(fs, fft_size)
    cf0 = torch.where(f0 <= floor, torch.full_like(f0, KDEFAULT_F0), f0)

    # F0-adaptive windowing. Ragged windows are LEFT-ALIGNED at buffer index 0,
    # exactly as the C writes them. cf0 >= min(floor, KDEFAULT_F0) bounds the
    # half window statically (+1 for a 1-ulp device/host disagreement), so no
    # sync; the excess width is masked.
    hwl = matlab_round(1.5 * fs / cf0)
    n, valid = frame_windows(hwl, round_int(1.5 * fs / min(floor, KDEFAULT_F0)) + 1)
    origin = matlab_round(t * fs + 0.001).unsqueeze(1)

    w = 0.5 * torch.cos(math.pi * (n / 1.5 / fs) * cf0.unsqueeze(1)) + 0.5
    w = torch.where(valid, w, torch.zeros_like(w))
    w = w / torch.sqrt((w * w).sum(1, keepdim=True))  # normalize the window

    noise = dither(w.shape, generator, dev) * KSG
    samples = clamp_gather(x, (origin + n).long(), bounds)
    seg = torch.where(valid, samples * w + noise, torch.zeros_like(w))
    # Window-weighted mean removal; w == 0 outside `valid` keeps the tail clean.
    seg = seg - w * (seg.sum(1, keepdim=True) / w.sum(1, keepdim=True))

    buf = torch.zeros(n_frames, fft_size, dtype=torch.float64, device=dev)
    width = min(seg.shape[1], fft_size)
    buf[:, :width] = seg[:, :width]
    spec = torch.fft.rfft(buf, dim=1)
    ps = spec.real**2 + spec.imag**2

    f0_max = cf0.max().item()
    ps = dc_correction(ps, cf0, fs, fft_size, f0_max)
    ps = linear_smoothing(ps, cf0 * 2.0 / 3.0, fs, fft_size, f0_max * 2.0 / 3.0)
    # Infinitesimal noise: this is what guards the log() below against zeros.
    ps = ps + dither(ps.shape, generator, dev).abs() * KEPS

    # Smoothing (log axis) and spectral recovery on the cepstrum domain.
    ia = torch.arange(fft_size // 2 + 1, device=dev, dtype=torch.float64).unsqueeze(0)
    quef = ia / fs
    z = cf0.unsqueeze(1) * quef
    sinc = torch.where(z == 0, torch.ones_like(z), torch.sin(math.pi * z) / (math.pi * z))
    lifter = sinc * (
        (1.0 - 2.0 * q1) + 2.0 * q1 * torch.cos(2.0 * math.pi * quef * cf0.unsqueeze(1))
    )
    lps = torch.log(ps)
    full = torch.cat([lps, lps.flip(1)[:, 1:-1]], dim=1)  # even-symmetric
    ceps = torch.fft.rfft(full, dim=1).real
    # Upstream's /fft_size cancels against its unnormalized c2r; irfft is normalized.
    return torch.exp(torch.fft.irfft(ceps * lifter, n=fft_size, dim=1)[:, : fft_size // 2 + 1])
