"""Batched primitives shared by the native WORLD analysis stages.

Ported from the WORLD C++ reference (github.com/mmorise/World, BSD-3-Clause).
All math is float64 on the caller's device; every routine takes and returns
tensors with a leading frame dimension.

Frames from several utterances can share one call: the samples live in one
flat buffer and each frame carries ``bounds = (start, length)`` of its own
utterance, so edge replication stops at that utterance's ends.
"""

import math

import torch

KLOG2 = 0.69314718055994529
KEPS = 2.2204460492503131e-16  # kEps
KSG = 1e-12  # kMySafeGuardMinimum


def matlab_round(x: torch.Tensor) -> torch.Tensor:
    """Round half away from zero (WORLD's MyRoundInt / matlab round)."""
    return torch.where(x > 0, torch.trunc(x + 0.5), torch.trunc(x - 0.5))


def round_int(v: float) -> int:
    """Scalar matlab_round (half away from zero), as an int."""
    return int(v + 0.5) if v > 0 else int(v - 0.5)


def clamp_gather(
    x: torch.Tensor, idx: torch.Tensor, bounds: tuple[torch.Tensor, torch.Tensor] | None
) -> torch.Tensor:
    """x[idx] with edge replication at each frame's utterance boundaries.

    idx: (F, W) integer sample indices, local to the frame's utterance.
    bounds: None when x is a single utterance, else per-frame (start, length)
    of the frame's utterance within the flat buffer x.
    """
    if bounds is None:
        return x[idx.clamp(0, x.shape[0] - 1)]
    start, length = bounds
    return x[idx.clamp(min=0).minimum((length - 1).unsqueeze(1)) + start.unsqueeze(1)]


def suitable_fft_size(n: int) -> int:
    """GetSuitableFFTSize: next power of two strictly above n."""
    return int(2.0 ** (int(math.log(n) / KLOG2) + 1.0))


def nuttall(n: int, device: torch.device) -> torch.Tensor:
    """Symmetric 4-term Nuttall window (WORLD's NuttallWindow)."""
    t = torch.arange(n, device=device, dtype=torch.float64) / (n - 1.0)
    return (
        0.355768
        - 0.487396 * torch.cos(2 * math.pi * t)
        + 0.144232 * torch.cos(4 * math.pi * t)
        - 0.012604 * torch.cos(6 * math.pi * t)
    )


def dither(shape, generator, device: torch.device) -> torch.Tensor:
    """Gaussian dither for the analysis stages.

    WORLD draws from Irwin-Hall(12) -- the classic sum-of-12-uniforms
    approximation to the standard normal -- via a serial xorshift128 stream
    this port cannot reproduce anyway. Sampling `randn` directly is the same
    law for 25x less work, and float32 is ample for a term scaled by 1e-12
    before being added to a float64 signal.

    The dither is NOT cosmetic: without it, digitally silent or constant input
    drives log(0) in CheapTrick's cepstral liftering and the envelope becomes
    all-NaN.
    """
    return torch.randn(
        tuple(shape), generator=generator, device=device, dtype=torch.float32
    ).to(torch.float64)


def interp1q_rows(y: torch.Tensor, p: torch.Tensor, use_floor: bool = True) -> torch.Tensor:
    """Row-wise gather interpolation, the batched form of upstream's interp1Q.

    y: (F, L) one lookup table per frame.
    p: (F, M) query positions already in table-index coordinates.
    """
    b = torch.floor(p) if use_floor else torch.trunc(p)
    base = b.long().clamp(0, y.shape[1] - 1)
    frac = p - b
    dy = torch.zeros_like(y)
    dy[:, :-1] = y[:, 1:] - y[:, :-1]  # dy[:, -1] stays 0, as upstream requires
    return torch.gather(y, 1, base) + torch.gather(dy, 1, base) * frac


def dc_correction(
    ps: torch.Tensor,
    f0: torch.Tensor,
    fs: int,
    fft_size: int,
    f0_max: float | None = None,
) -> torch.Tensor:
    """Fold the spectrum about f0 into the low bins (WORLD's DCCorrection).

    ps: (F, fft_size//2+1) power spectra. f0: (F,). f0_max, when the caller
    already holds an upper bound of f0 on the host, spares a device sync; the
    extra columns it may add are masked, so the result is unchanged.
    """
    upper = (2 + torch.trunc(f0 * fft_size / fs)).long()  # (F,)
    if f0_max is None:
        m = int(upper.max().item())
    else:  # +1 absorbs a 1-ulp host/device disagreement in the trunc
        m = min(2 + int(f0_max * fft_size / fs) + 1, ps.shape[1])
    idx = torch.arange(m, device=ps.device)
    axis = idx.double() * fs / fft_size
    t = (axis.unsqueeze(0) - f0.unsqueeze(1)) / (-float(fs) / fft_size)
    rep = interp1q_rows(ps, t, use_floor=False)  # trunc: positions are >= 0
    valid = idx.unsqueeze(0) < (upper - 1).unsqueeze(1)  # n_rep = upper - 1
    out = ps.clone()
    out[:, :m] = out[:, :m] + torch.where(valid, rep, torch.zeros_like(rep))
    return out


def linear_smoothing(
    ps: torch.Tensor,
    width: torch.Tensor,
    fs: int,
    fft_size: int,
    width_max: float | None = None,
) -> torch.Tensor:
    """Boxcar smoothing of `width` Hz per frame (WORLD's LinearSmoothing).

    Upstream reflect-pads by a per-frame boundary and queries the cumulative
    array relative to it; the boundary is an additive constant that cancels in
    the high-low difference, so pad once by the batch maximum and query in
    original-array coordinates. That reparametrization drives the low query
    negative at i = 0, which is why this function uses floor, not trunc.
    width_max is an optional host-side upper bound of width (spares a sync).
    """
    if width_max is None:
        width_max = width.max().item()
    b = int(width_max * fft_size / fs) + 2  # +2 = margin; b >= w/2 + 1 suffices
    ms = torch.cat([ps[:, 1 : b + 1].flip(1), ps, ps[:, -b - 1 : -1].flip(1)], dim=1)
    seg = torch.cumsum(ms * (fs / fft_size), dim=1)  # scale THEN accumulate
    i = torch.arange(fft_size // 2 + 1, device=ps.device, dtype=torch.float64).unsqueeze(0)
    w = (width * fft_size / fs).unsqueeze(1)
    hi = interp1q_rows(seg, i + w / 2 + b - 0.5)
    lo = interp1q_rows(seg, i - w / 2 + b - 0.5)
    return (hi - lo) / width.unsqueeze(1)


def frame_windows(hwl: torch.Tensor, n_max: int):
    """Ragged per-frame window indices, LEFT-ALIGNED at buffer index 0.

    hwl: (F,) per-frame half-window lengths.
    Returns (offsets, valid), both (F, 2*n_max+1); offsets[f, j] = j - hwl[f]
    is the base sample index and valid masks the ragged tail.
    """
    j = torch.arange(2 * n_max + 1, device=hwl.device, dtype=torch.float64).unsqueeze(0)
    return j - hwl.unsqueeze(1), j <= (2 * hwl).unsqueeze(1)
