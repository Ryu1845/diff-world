"""DIO F0 estimation, ported from WORLD's dio.cpp (BSD-3-Clause).

Only ``speed == 1`` is supported, so ``decimation_ratio`` is always 1 and
upstream's ``decimate()`` path is unreachable; it is deliberately not ported.
"""

import functools
import math

import torch

from ._common import KLOG2, KSG, nuttall, round_int, suitable_fft_size

KCUTOFF = 50.0
KMAXIMUM_VALUE = 100000.0


def num_frames(x_length: int, fs: int, frame_period: float = 5.0) -> int:
    """Number of DIO analysis frames for x_length samples."""
    return int(1000.0 * x_length / fs / frame_period) + 1


@functools.lru_cache(maxsize=8)
def _filter_spectra(fs: int, fft_size: int, half_lengths: tuple, device: torch.device):
    """Signal-independent spectra: the 50 Hz low-cut filter, and every band's
    Nuttall low-pass stacked as (nb, fft_size//2+1). Callers only read them."""
    cutoff_in_sample = round_int(fs / KCUTOFF)
    n = cutoff_in_sample * 2 + 1
    h = (n - 1) // 2
    k = torch.arange(1, n + 1, device=device, dtype=torch.float64)
    w = 0.5 - 0.5 * torch.cos(k * 2.0 * math.pi / (n + 1))
    w = -w / w.sum()
    lcf = torch.zeros(fft_size, dtype=torch.float64, device=device)
    lcf[: h + 1] = w[h:]
    lcf[fft_size - h :] = w[:h]
    lcf[0] += 1.0

    lpf = torch.zeros(len(half_lengths), fft_size, dtype=torch.float64, device=device)
    for b, hl in enumerate(half_lengths):
        lpf[b, : hl * 4] = nuttall(hl * 4, device)
    return torch.fft.rfft(lcf), torch.fft.rfft(lpf, dim=1)


def _spectrum_for_estimation(
    x: torch.Tensor, y_length: torch.Tensor, fft_size: int, lcf: torch.Tensor
) -> torch.Tensor:
    """GetSpectrumForEstimation per row: DC-removed spectrum times the low-cut filter.

    x: (B, L) rows, zero past their own lengths. y_length: (B,) float64.
    """
    y = x.new_zeros(x.shape[0], fft_size)
    y[:, : x.shape[1]] = x
    # Biased mean: the sum runs over y_length = x_length + 1 samples, the last
    # of which is still zero, so y[x_length] ends up at -mean_y, not 0.
    inside = torch.arange(fft_size, device=x.device).unsqueeze(0) < y_length.unsqueeze(1)
    mean = y.sum(1, keepdim=True) / y_length.unsqueeze(1)
    y = torch.where(inside, y - mean, y)
    return torch.fft.rfft(y, dim=1) * lcf


def _filtered_signals(
    spectrum: torch.Tensor, lpf: torch.Tensor, half_lengths, fft_size: int, width: int
) -> torch.Tensor:
    """GetFilteredSignal for every row and band at once: (B, nb, width).

    Replicates upstream's off-by-one mirror write, which clobbers bins
    ``fft_size/2 - 1`` and ``fft_size/2`` with ``Y[N/2] * (Y*H)[N/2-1]``.
    Samples past a row's own y_length are garbage; callers mask them.
    """
    dev = spectrum.device
    s = spectrum.unsqueeze(1) * lpf
    bug = spectrum[:, -1:] * s[..., -2]
    s[..., -2] = bug
    s[..., -1] = bug.real.to(s.dtype)
    s[..., 0] = s[..., 0].real.to(s.dtype)
    full = torch.fft.irfft(s, n=fft_size, dim=-1) * fft_size
    bias = torch.tensor([hl * 2 for hl in half_lengths], device=dev).unsqueeze(1)
    idx = (bias + torch.arange(width, device=dev)).clamp_(max=fft_size - 1)
    return full.gather(2, idx.expand(s.shape[0], -1, -1))


def _band_candidates(
    sig: torch.Tensor,
    y_length: torch.Tensor,
    t: torch.Tensor,
    fs: float,
    boundary: list[float],
    f0_floor: float,
    f0_ceil: float,
):
    """ZeroCrossingEngine + per-band F0 candidates, for all rows and bands at once.

    sig: (B, nb, L) band signals. y_length: (B,) int64. t: (F,) frame times.
    Returns candidates and scores, both (B, nb, F).

    Upstream walks each of the four event sets (negative, positive, peak, dip)
    of each band separately. Here every (row, band, event) set is one row of a
    (R, L) mask; its events are compacted left into an inf-padded table, so
    one batched searchsorted serves them all. Per-element arithmetic is
    verbatim upstream's.
    """
    B, nb, L = sig.shape
    dev = sig.device
    d = sig[..., 1:] - sig[..., :-1]
    d = torch.cat([d, d.new_zeros(B, nb, 1)], dim=-1)
    rows = torch.stack([sig, -sig, d, -d], dim=2).reshape(-1, L)
    n = torch.stack([y_length, y_length, y_length - 1, y_length - 1], dim=1)  # (B, 4)
    n = n.unsqueeze(1).expand(B, nb, 4).reshape(-1)

    prev, cur = rows[:, :-1], rows[:, 1:]
    j = torch.arange(L - 1, device=dev)
    mask = (prev > 0.0) & (cur <= 0.0) & (j.unsqueeze(0) < (n - 1).unsqueeze(1))
    fine_all = (j + 1).double() - prev / torch.where(mask, cur - prev, -1.0)
    counts = mask.sum(1)
    width = max(int(counts.max().item()), 3)  # the one sync of this stage
    slot = torch.where(mask, mask.cumsum(1) - 1, width)
    fine = rows.new_full((rows.shape[0], width + 1), math.inf)
    fine = fine.scatter_(1, slot, fine_all)[:, :width]

    intervals = fs / (fine[:, 1:] - fine[:, :-1])
    locations = (fine[:, :-1] + fine[:, 1:]) / 2.0 / fs  # inf-padded, still sorted
    n_loc = (counts - 1).clamp(min=2)

    # Upstream's histc-based interp1: linear, extrapolating past both ends.
    ti = t.unsqueeze(0).expand(rows.shape[0], -1).contiguous()
    k = torch.searchsorted(locations.contiguous(), ti, right=True)
    k = k.clamp(min=1).minimum((n_loc - 1).unsqueeze(1))
    lo = locations.gather(1, k - 1)
    v_lo = intervals.gather(1, k - 1)
    vals = v_lo + (ti - lo) / (locations.gather(1, k) - lo) * (intervals.gather(1, k) - v_lo)

    vals = vals.reshape(B, nb, 4, -1)
    c = vals.mean(dim=2)
    sd = torch.sqrt(((vals - c.unsqueeze(2)) ** 2).sum(dim=2) / 3.0)
    upper = torch.tensor(boundary, dtype=torch.float64, device=dev).unsqueeze(1)
    lower = torch.tensor([b / 2.0 for b in boundary], dtype=torch.float64, device=dev).unsqueeze(1)
    bad = (c > upper) | (c < lower) | (c > f0_ceil) | (c < f0_floor)
    cand = torch.where(bad, torch.zeros_like(c), c)
    scores = torch.where(bad, torch.full_like(sd, KMAXIMUM_VALUE), sd) / (cand + KSG)

    # A band with too few events in any set (upstream: fewer than 2 edges, or
    # intervals.size() - 2 <= 0) yields no candidates at all.
    dead = (counts.reshape(B, nb, 4) <= 3).any(dim=2).unsqueeze(2)
    cand = torch.where(dead, torch.zeros_like(cand), cand)
    scores = torch.where(dead, torch.full_like(scores, KMAXIMUM_VALUE / KSG), scores)
    return cand, scores


def _select_best_f0(current: float, past: float, cands: list, allowed_range: float):
    """Pure-Python floats: the same IEEE-754 double ops as numpy scalars, at a
    fraction of the per-call overhead. Strict `<` keeps np.argmin's first-minimum
    tie-break."""
    reference = (current * 3.0 - past) / 2.0
    best = cands[0]
    best_err = abs(reference - best)
    for c in cands[1:]:
        err = abs(reference - c)
        if err < best_err:
            best, best_err = c, err
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
    # iteration and both break early), so they run serially on the host.
    cand = candidates.T.tolist()
    f0 = step2.tolist()
    for i, start in enumerate(neg):
        limit = f0_length - 1 if i == len(neg) - 1 else neg[i + 1]
        for j in range(start, limit):
            f0[j + 1] = _select_best_f0(f0[j], f0[j - 1], cand[j + 1], allowed_range)
            if f0[j + 1] == 0.0:
                break
    for i in range(len(pos) - 1, -1, -1):
        limit = 1 if i == 0 else pos[i - 1]
        j = pos[i]
        while j > limit:
            f0[j - 1] = _select_best_f0(f0[j], f0[j + 1], cand[j - 1], allowed_range)
            if f0[j - 1] == 0.0:
                break
            j -= 1
    return torch.tensor(f0, dtype=torch.float64, device=best.device)


def dio(
    x: torch.Tensor,
    fs: int,
    f0_floor: float = 71.0,
    f0_ceil: float = 800.0,
    channels_in_octave: float = 2.0,
    frame_period: float = 5.0,
    allowed_range: float = 0.1,
    lengths: list[int] | None = None,
):
    """Estimate F0 with DIO.

    x: (L,) one utterance, or (B, L) rows zero-padded past ``lengths``
    (default: every row is L long).
    Returns ``(f0, temporal_positions)``: (F,) and (F,) for 1-D input;
    (B, F_max) and (F_max,) for batched input, each row's f0 zero past its own
    ``num_frames(lengths[b])``.

    Rows are grouped by their own DIO FFT size, so every row is filtered
    exactly as it would be alone.
    """
    single = x.dim() == 1
    x = x.to(torch.float64)
    if single:
        x = x.unsqueeze(0)
    dev = x.device
    B, L = x.shape
    lengths = [L] * B if lengths is None else [int(n) for n in lengths]

    nb = 1 + int(math.log(f0_ceil / f0_floor) / KLOG2 * channels_in_octave)
    boundary = [f0_floor * 2.0 ** ((i + 1) / channels_in_octave) for i in range(nb)]
    half_lengths = tuple(round_int(fs / b / 2.0) for b in boundary)
    pad = round_int(fs / KCUTOFF) * 2 + 1 + 4 * int(1.0 + fs / boundary[0] / 2.0)

    frames = [num_frames(n, fs, frame_period) for n in lengths]
    f_max = max(frames)
    # Bit-identical to upstream's i * frame_period / 1000.0 (order matters).
    # The divisor MUST be a tensor: CUDA lowers float64 `/ scalar` to a
    # reciprocal-multiply (`* 0.001`), which is off by 1 ulp from true IEEE
    # division on 74/601 frames here. That ulp is not cosmetic -- t * fs lands
    # exactly on .5 rounding boundaries for ~20% of frames, so matlab_round
    # flips the window's sample index and StoneMask's F0 moves by whole Hz.
    t = torch.arange(f_max, device=dev, dtype=torch.float64) * frame_period / torch.tensor(
        1000.0, device=dev, dtype=torch.float64
    )

    buckets: dict[int, list[int]] = {}
    for b, n in enumerate(lengths):
        buckets.setdefault(suitable_fft_size(n + 1 + pad), []).append(b)

    candidates = x.new_zeros(B, nb, f_max)
    best = x.new_zeros(B, f_max)
    for fft_size, rows in buckets.items():
        n = [lengths[b] for b in rows]
        width = max(n) + 1
        f_k = max(frames[b] for b in rows)
        idx = torch.tensor(rows, device=dev)
        n_t = torch.tensor(n, device=dev)
        xb = x.index_select(0, idx)[:, : max(n)]
        xb = torch.where(torch.arange(max(n), device=dev) < n_t.unsqueeze(1), xb, 0.0)
        lcf, lpf = _filter_spectra(fs, fft_size, half_lengths, dev)
        spectrum = _spectrum_for_estimation(xb, (n_t + 1).double(), fft_size, lcf)
        signals = _filtered_signals(spectrum, lpf, half_lengths, fft_size, width)
        cand, scores = _band_candidates(
            signals, n_t + 1, t[:f_k], float(fs), boundary, f0_floor, f0_ceil
        )
        # First minimum wins ties, matching upstream's strict `>`.
        candidates[idx, :, :f_k] = cand
        best[idx, :f_k] = cand.gather(1, scores.argmin(dim=1, keepdim=True)).squeeze(1)

    candidates, best = candidates.cpu(), best.cpu()
    f0 = torch.zeros(B, f_max, dtype=torch.float64)
    for b in range(B):
        f = frames[b]
        f0[b, :f] = _fix_f0_contour(
            candidates[b, :, :f], best[b, :f], f, frame_period, f0_floor, allowed_range
        )
    f0 = f0.to(dev)
    return (f0[0], t) if single else (f0, t)
