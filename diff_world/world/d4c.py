"""D4C band-aperiodicity estimation, ported from WORLD's d4c.cpp (BSD-3-Clause).

Two independent FFT sizes are in play: the internals run at ``fft_size_d4c``
(2048 at fs=22050, derived from kFloorF0D4C = 47 Hz) while the output grid uses
the caller's ``fft_size``. They are never conflated.
"""

import math

import torch

from ._common import (
    KLOG2,
    KSG,
    dc_correction,
    dither,
    frame_windows,
    linear_smoothing,
    matlab_round,
    nuttall,
)

KFLOOR_F0_D4C = 47.0
KFREQUENCY_INTERVAL = 3000.0
KUPPER_LIMIT = 15000.0
KSAFE_GUARD_D4C = 1e-6


def _windowed_waveform(
    x: torch.Tensor,
    fs: int,
    f0: torch.Tensor,
    pos: torch.Tensor,
    window_type: str,
    ratio: float,
    n_fft: int,
    generator: torch.Generator | None,
):
    """GetWindowedWaveform, batched.

    Unlike CheapTrick's variant: ratio-parameterized half window, two window
    shapes, no window-energy normalization, noise amplitude 1e-6. The ragged
    windows MUST be left-aligned at buffer index 0 -- centering them on a
    shared max half-width breaks GetCentroid's ramp weighting.
    """
    dev = x.device
    hwl = matlab_round(ratio * fs / f0 / 2.0)
    n, valid = frame_windows(hwl, int(hwl.max().item()))
    origin = matlab_round(pos * fs + 0.001).unsqueeze(1)
    safe = (origin + n).long().clamp_(0, x.shape[0] - 1)

    a = math.pi * ((2.0 * n / ratio) / fs) * f0.unsqueeze(1)
    if window_type == "hann":
        w = 0.5 * torch.cos(a) + 0.5
    else:
        w = 0.42 + 0.5 * torch.cos(a) + 0.08 * torch.cos(a * 2)
    w = torch.where(valid, w, torch.zeros_like(w))

    noise = dither(w.shape, generator, dev) * KSAFE_GUARD_D4C
    seg = torch.where(valid, x[safe] * w + noise, torch.zeros_like(w))
    seg = seg - w * (seg.sum(1, keepdim=True) / w.sum(1, keepdim=True))
    seg = torch.where(valid, seg, torch.zeros_like(seg))  # w == 0 rows got the offset

    buf = torch.zeros(f0.shape[0], n_fft, dtype=torch.float64, device=dev)
    width = min(seg.shape[1], n_fft)
    buf[:, :width] = seg[:, :width]
    return buf, hwl


def _centroid(
    x: torch.Tensor,
    fs: int,
    f0: torch.Tensor,
    pos: torch.Tensor,
    n_fft: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """GetCentroid: energy centroid of the Blackman-windowed waveform."""
    buf, _ = _windowed_waveform(x, fs, f0, pos, "black", 4.0, n_fft, generator)
    m = (matlab_round(2.0 * fs / f0) * 2).long()
    mask = torch.arange(n_fft, device=x.device).unsqueeze(0) <= m.unsqueeze(1)
    power = (torch.where(mask, buf, torch.zeros_like(buf)) ** 2).sum(1, keepdim=True)
    # Normalizes the windowed SIGNAL after windowing (CheapTrick normalizes the
    # window before applying it -- genuinely different operations).
    buf = torch.where(mask, buf / torch.sqrt(power), buf)
    s1 = torch.fft.rfft(buf, dim=1)
    ramp = torch.arange(n_fft, device=x.device, dtype=torch.float64) + 1.0
    s2 = torch.fft.rfft(buf * ramp, dim=1)
    return s2.real * s1.real + s1.imag * s2.imag


def d4c(
    x: torch.Tensor,
    f0: torch.Tensor,
    t: torch.Tensor,
    fs: int,
    fft_size: int = 1024,
    threshold: float = 0.85,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estimate band aperiodicity. Returns (F, fft_size//2+1), linear scale."""
    dev = x.device
    x = x.to(torch.float64)
    f0 = f0.to(torch.float64)
    t = t.to(torch.float64)
    n_frames = f0.shape[0]

    # Unvoiced and love-train-gated frames keep 1 - 1e-12, never exactly 1.0.
    ap = torch.full(
        (n_frames, fft_size // 2 + 1), 1.0 - KSG, dtype=torch.float64, device=dev
    )

    fsd = int(2.0 ** (1.0 + int(math.log(4.0 * fs / KFLOOR_F0_D4C + 1) / KLOG2)))
    nap = int(min(KUPPER_LIMIT, fs / 2.0 - KFREQUENCY_INTERVAL) / KFREQUENCY_INTERVAL)
    window_length = int(KFREQUENCY_INTERVAL * fsd / fs) * 2 + 1
    window = nuttall(window_length, dev)

    # D4C Love Train: fraction of (100, 4000] Hz energy within (100, 7900] Hz.
    ap0 = torch.zeros(n_frames, dtype=torch.float64, device=dev)
    voiced = f0 != 0.0
    if bool(voiced.any()):
        n_lt = int(2.0 ** (1.0 + int(math.log(3.0 * fs / 40.0 + 1) / KLOG2)))
        b0 = int(math.ceil(100.0 * n_lt / fs))
        b1 = int(math.ceil(4000.0 * n_lt / fs))
        b2 = int(math.ceil(7900.0 * n_lt / fs))
        cf = f0[voiced].clamp_min(40.0)
        buf, _ = _windowed_waveform(x, fs, cf, t[voiced], "black", 3.0, n_lt, generator)
        wlen = (matlab_round(1.5 * fs / cf) * 2 + 1).long()
        buf = torch.where(
            torch.arange(n_lt, device=dev).unsqueeze(0) < wlen.unsqueeze(1),
            buf,
            torch.zeros_like(buf),
        )
        spec = torch.fft.rfft(buf, dim=1)
        ps = spec.real**2 + spec.imag**2
        ps[:, : b0 + 1] = 0.0
        cs = torch.cumsum(ps[:, b0 : b2 + 1], dim=1)
        ap0[voiced] = cs[:, b1 - b0] / cs[:, b2 - b0]

    sel = voiced & (ap0 > threshold)
    if not bool(sel.any()):
        return ap

    cf = f0[sel].clamp_min(KFLOOR_F0_D4C)
    pos = t[sel]

    # Static centroid. Note the negative origins on the first term.
    sc = _centroid(x, fs, cf, pos - 0.25 / cf, fsd, generator) + _centroid(
        x, fs, cf, pos + 0.25 / cf, fsd, generator
    )
    sc = dc_correction(sc, cf, fs, fsd)

    # Smoothed power spectrum.
    buf, _ = _windowed_waveform(x, fs, cf, pos, "hann", 4.0, fsd, generator)
    spec = torch.fft.rfft(buf, dim=1)
    sps = spec.real**2 + spec.imag**2
    sps = dc_correction(sps, cf, fs, fsd)
    sps = linear_smoothing(sps, cf, fs, fsd)  # width f0, NOT 2/3 f0

    # Static group delay. Order is load-bearing: the second smoothing consumes
    # the output of the first, not the raw ratio.
    sgd = sc / sps
    sgd = linear_smoothing(sgd, cf / 2.0, fs, fsd)
    sgd = sgd - linear_smoothing(sgd, cf, fs, fsd)

    # Coarse aperiodicity in multiples of 3 kHz.
    boundary = int(matlab_round(torch.tensor(fsd * 8.0 / window_length)).item())
    hwl = window_length // 2
    coarse = torch.empty(cf.shape[0], nap, dtype=torch.float64, device=dev)
    for q in range(nap):
        center = int(KFREQUENCY_INTERVAL * (q + 1) * fsd / fs)
        buf = torch.zeros(cf.shape[0], fsd, dtype=torch.float64, device=dev)
        buf[:, : 2 * hwl + 1] = sgd[:, center - hwl : center + hwl + 1] * window.unsqueeze(0)
        spec = torch.fft.rfft(buf, dim=1)
        pp = spec.real**2 + spec.imag**2
        smallest = torch.topk(pp, fsd // 2 + 1, dim=1, largest=False, sorted=True).values
        cs = torch.cumsum(smallest, dim=1)
        coarse[:, q] = 10 * torch.log10(cs[:, fsd // 2 - boundary - 1] / cs[:, fsd // 2])
    coarse = torch.minimum(
        torch.zeros_like(coarse), coarse + (cf.unsqueeze(1) - 100) / 50.0
    )

    # Linear (in Hz, on both axes) expansion onto the output grid.
    knots = torch.tensor(
        [i * KFREQUENCY_INTERVAL for i in range(nap + 1)] + [fs / 2.0],
        dtype=torch.float64,
        device=dev,
    )
    shape = (coarse.shape[0], 1)
    table = torch.cat(
        [
            torch.full(shape, -60.0, dtype=torch.float64, device=dev),
            coarse,
            torch.full(shape, -KSG, dtype=torch.float64, device=dev),
        ],
        dim=1,
    )
    fax = torch.arange(fft_size // 2 + 1, device=dev, dtype=torch.float64) * fs / fft_size
    k = torch.searchsorted(knots.contiguous(), fax.contiguous(), right=True).clamp(
        1, knots.shape[0] - 1
    )
    s = (fax - knots[k - 1]) / (knots[k] - knots[k - 1])
    lo = table[:, k - 1]
    ap[sel] = 10.0 ** ((lo + s.unsqueeze(0) * (table[:, k] - lo)) / 20.0)
    return ap
