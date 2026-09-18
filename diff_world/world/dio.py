"""DIO F0 estimation, ported from WORLD's dio.cpp (BSD-3-Clause).

Only ``speed == 1`` is supported, so ``decimation_ratio`` is always 1 and
upstream's ``decimate()`` path is unreachable; it is deliberately not ported.
"""

import math

import numpy as np
import torch

from ._common import KLOG2, KSG, matlab_round, nuttall, suitable_fft_size

KCUTOFF = 50.0
KMAXIMUM_VALUE = 100000.0


def _round_int(v: float) -> int:
    """Scalar matlab_round (half away from zero), as an int."""
    return int(v + 0.5) if v > 0 else int(v - 0.5)


def _spectrum_for_estimation(x: torch.Tensor, y_length: int, fs: int, fft_size: int):
    """GetSpectrumForEstimation: DC-removed spectrum times the 50 Hz low-cut filter."""
    dev = x.device
    x_length = x.shape[0]
    y = torch.zeros(fft_size, dtype=torch.float64, device=dev)
    y[:x_length] = x
    # Biased mean: the sum runs over y_length = x_length + 1 samples, the last
    # of which is still zero, so y[x_length] ends up at -mean_y, not 0.
    y[:y_length] = y[:y_length] - y[:y_length].sum() / y_length
    spectrum = torch.fft.rfft(y)

    cutoff_in_sample = _round_int(fs / KCUTOFF)
    n = cutoff_in_sample * 2 + 1
    h = (n - 1) // 2
    k = torch.arange(1, n + 1, device=dev, dtype=torch.float64)
    w = 0.5 - 0.5 * torch.cos(k * 2.0 * math.pi / (n + 1))
    w = -w / w.sum()
    lcf = torch.zeros(fft_size, dtype=torch.float64, device=dev)
    lcf[: h + 1] = w[h:]
    lcf[fft_size - h :] = w[:h]
    lcf[0] += 1.0
    return spectrum * torch.fft.rfft(lcf)


def _filtered_signals(spectrum: torch.Tensor, half_lengths, fft_size: int, y_length: int):
    """GetFilteredSignal for every band at once.

    Replicates upstream's off-by-one mirror write, which clobbers bins
    ``fft_size/2 - 1`` and ``fft_size/2`` with ``Y[N/2] * (Y*H)[N/2-1]``.
    """
    dev = spectrum.device
    nb = len(half_lengths)
    lpf = torch.zeros(nb, fft_size, dtype=torch.float64, device=dev)
    for b, hl in enumerate(half_lengths):
        lpf[b, : hl * 4] = nuttall(hl * 4, dev)
    s = spectrum.unsqueeze(0) * torch.fft.rfft(lpf, dim=1)
    bug = spectrum[-1] * s[:, -2]
    s[:, -2] = bug
    s[:, -1] = bug.real.to(s.dtype)
    s[:, 0] = s[:, 0].real.to(s.dtype)
    full = torch.fft.irfft(s, n=fft_size, dim=1) * fft_size
    out = []
    for b, hl in enumerate(half_lengths):
        bias = hl * 2
        out.append(full[b, bias : bias + y_length])
    return out


def _zero_crossings(sig: torch.Tensor, n: int, fs: float):
    """ZeroCrossingEngine: locations (s) and intervals (Hz) of falling crossings."""
    edges = torch.nonzero((sig[: n - 1] > 0.0) & (sig[1:n] <= 0.0)).squeeze(1) + 1
    if edges.numel() < 2:
        return None
    prev = sig[edges - 1]
    fine = edges.double() - prev / (sig[edges] - prev)
    intervals = fs / (fine[1:] - fine[:-1])
    locations = (fine[:-1] + fine[1:]) / 2.0 / fs
    return locations, intervals


def _interp1(loc: torch.Tensor, val: torch.Tensor, ti: torch.Tensor) -> torch.Tensor:
    """Upstream's histc-based interp1: linear, extrapolating past both ends."""
    k = torch.searchsorted(loc.contiguous(), ti.contiguous(), right=True).clamp(
        1, loc.numel() - 1
    )
    lo = loc[k - 1]
    return val[k - 1] + (ti - lo) / (loc[k] - lo) * (val[k] - val[k - 1])


def _four_zero_crossings(sig: torch.Tensor, y_length: int, fs: float):
    """The four event sets: negative, positive, peak, dip."""
    d = sig[1:y_length] - sig[: y_length - 1]
    return (
        _zero_crossings(sig, y_length, fs),
        _zero_crossings(-sig, y_length, fs),
        _zero_crossings(d, y_length - 1, fs),
        _zero_crossings(-d, y_length - 1, fs),
    )


def _select_best_f0(current: float, past: float, cands: np.ndarray, allowed_range: float):
    reference = (current * 3.0 - past) / 2.0
    best = float(cands[int(np.argmin(np.abs(reference - cands)))])
    if reference == 0.0:
        return 0.0
    return 0.0 if abs(1.0 - best / reference) > allowed_range else best


def _fix_f0_contour(
    candidates: torch.Tensor,
    best: torch.Tensor,
    f0_length: int,
    frame_period: float,
    f0_floor: float,
    allowed_range: float,
) -> torch.Tensor:
    vrm = int(0.5 + 1000.0 / frame_period / f0_floor) * 2 + 1
    if f0_length <= vrm:
        # Upstream returns early and leaves the caller's buffer untouched;
        # pyworld zero-allocates it, so all-zero is the observable behavior.
        return torch.zeros(f0_length, dtype=torch.float64, device=best.device)

    # Step 1: drop unnatural frame-to-frame jumps.
    base = best.clone()
    base[:vrm] = 0.0
    base[f0_length - vrm :] = 0.0
    prev = torch.cat([base.new_zeros(1), base[:-1]])
    keep = ((base - prev) / (KSG + base)).abs() < allowed_range
    step1 = torch.where(keep, base, torch.zeros_like(base))
    step1[:vrm] = 0.0

    # Step 2: erase frames whose +/- center neighbourhood contains a zero.
    center = (vrm - 1) // 2
    step2 = step1.clone()
    if f0_length > 2 * center:
        windows = step1.unfold(0, 2 * center + 1, 1)
        zeroed = (windows == 0).any(dim=1)
        inner = step2[center : f0_length - center]
        step2[center : f0_length - center] = torch.where(
            zeroed, torch.zeros_like(inner), inner
        )

    voiced = step2 != 0
    neg = torch.nonzero(voiced[:-1] & ~voiced[1:]).squeeze(1).tolist()
    pos = (torch.nonzero(~voiced[:-1] & voiced[1:]).squeeze(1) + 1).tolist()

    # Steps 3 and 4 are genuinely recurrent (each write feeds the next
    # iteration and both break early), so they run serially on CPU.
    cand = candidates.detach().cpu().numpy()
    f0 = step2.detach().cpu().numpy().copy()
    for i, start in enumerate(neg):
        limit = f0_length - 1 if i == len(neg) - 1 else neg[i + 1]
        for j in range(start, limit):
            f0[j + 1] = _select_best_f0(f0[j], f0[j - 1], cand[:, j + 1], allowed_range)
            if f0[j + 1] == 0.0:
                break
    for i in range(len(pos) - 1, -1, -1):
        limit = 1 if i == 0 else pos[i - 1]
        j = pos[i]
        while j > limit:
            f0[j - 1] = _select_best_f0(f0[j], f0[j + 1], cand[:, j - 1], allowed_range)
            if f0[j - 1] == 0.0:
                break
            j -= 1
    return torch.as_tensor(f0, dtype=torch.float64, device=best.device)


def dio(
    x: torch.Tensor,
    fs: int,
    f0_floor: float = 71.0,
    f0_ceil: float = 800.0,
    channels_in_octave: float = 2.0,
    frame_period: float = 5.0,
    allowed_range: float = 0.1,
):
    """Estimate F0 with DIO. Returns ``(f0, temporal_positions)``, both (F,)."""
    dev = x.device
    x = x.to(torch.float64)
    x_length = x.shape[0]

    nb = 1 + int(math.log(f0_ceil / f0_floor) / KLOG2 * channels_in_octave)
    boundary = [f0_floor * 2.0 ** ((i + 1) / channels_in_octave) for i in range(nb)]

    y_length = x_length + 1
    fft_size = suitable_fft_size(
        y_length
        + _round_int(fs / KCUTOFF) * 2
        + 1
        + 4 * int(1.0 + fs / boundary[0] / 2.0)
    )

    spectrum = _spectrum_for_estimation(x, y_length, fs, fft_size)

    f0_length = int(1000.0 * x_length / fs / frame_period) + 1
    # Bit-identical to upstream's i * frame_period / 1000.0 (order matters).
    # The divisor MUST be a tensor: CUDA lowers float64 `/ scalar` to a
    # reciprocal-multiply (`* 0.001`), which is off by 1 ulp from true IEEE
    # division on 74/601 frames here. That ulp is not cosmetic -- t * fs lands
    # exactly on .5 rounding boundaries for ~20% of frames, so matlab_round
    # flips the window's sample index and StoneMask's F0 moves by whole Hz.
    t = torch.arange(f0_length, device=dev, dtype=torch.float64) * frame_period / torch.tensor(
        1000.0, device=dev, dtype=torch.float64
    )

    half_lengths = [_round_int(fs / b / 2.0) for b in boundary]
    signals = _filtered_signals(spectrum, half_lengths, fft_size, y_length)

    candidates = torch.zeros(nb, f0_length, dtype=torch.float64, device=dev)
    scores = torch.empty(nb, f0_length, dtype=torch.float64, device=dev)
    for b in range(nb):
        events = _four_zero_crossings(signals[b], y_length, float(fs))
        if any(e is None or e[1].numel() - 2 <= 0 for e in events):
            scores[b] = KMAXIMUM_VALUE / KSG
            continue
        vals = torch.stack([_interp1(loc, iv, t) for loc, iv in events])
        c = vals.mean(dim=0)
        sd = torch.sqrt(((vals - c) ** 2).sum(dim=0) / 3.0)
        bad = (c > boundary[b]) | (c < boundary[b] / 2.0) | (c > f0_ceil) | (c < f0_floor)
        cand = torch.where(bad, torch.zeros_like(c), c)
        candidates[b] = cand
        scores[b] = torch.where(bad, torch.full_like(sd, KMAXIMUM_VALUE), sd) / (cand + KSG)

    # First minimum wins ties, matching upstream's strict `>`.
    best = candidates.gather(0, scores.argmin(dim=0, keepdim=True)).squeeze(0)
    f0 = _fix_f0_contour(
        candidates, best, f0_length, frame_period, f0_floor, allowed_range
    )
    return f0, t
