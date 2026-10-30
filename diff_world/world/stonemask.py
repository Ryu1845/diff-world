"""StoneMask F0 refinement, ported from WORLD's stonemask.cpp (BSD-3-Clause).

Each frame's internal FFT size depends on its own f0, so frames are bucketed by
identical fft_size and each bucket runs as one batched FFT. Bucketing costs one
host sync per call, however many utterances the frames come from.
"""

import math

import torch

from ._common import KLOG2, KSG, clamp_gather, matlab_round

KFLOOR_F0_STONEMASK = 40.0


def _fix_f0(
    power: torch.Tensor,
    num_i: torch.Tensor,
    fft_size: int,
    fs: int,
    init_f0: torch.Tensor,
    n_harm: int,
) -> torch.Tensor:
    """Batched FixF0: amplitude-weighted instantaneous frequency over n_harm harmonics."""
    k = torch.arange(1, n_harm + 1, device=power.device, dtype=torch.float64)
    idx = torch.minimum(
        matlab_round(init_f0.unsqueeze(1) * fft_size / fs * k),
        torch.tensor(float(fft_size // 2), device=power.device, dtype=torch.float64),
    ).long()
    p = torch.gather(power, 1, idx)
    inst = torch.where(
        p == 0.0,
        torch.zeros_like(p),
        idx.double() * fs / fft_size
        + torch.gather(num_i, 1, idx) / p.clamp_min(1e-300) * fs / 2.0 / math.pi,
    )
    amp = torch.sqrt(p)
    return (amp * inst).sum(1) / ((amp * k).sum(1) + KSG)


def stonemask(
    x: torch.Tensor,
    f0: torch.Tensor,
    t: torch.Tensor,
    fs: int,
    *,
    bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Refine a DIO F0 contour via instantaneous frequency. Returns (F,).

    bounds: per-frame (start, length) of each frame's utterance in x, when
    frames of several utterances share one flat x (see ``clamp_gather``).
    """
    dev = x.device
    x = x.to(torch.float64)
    f0 = f0.to(torch.float64)
    t = t.to(torch.float64)

    out = torch.zeros_like(f0)
    active = (f0 > KFLOOR_F0_STONEMASK) & (f0 <= fs / 12.0)

    hwl = torch.trunc(1.5 * fs / f0.clamp_min(KSG) + 1.0)  # truncation, not rounding
    length = 2.0 * hwl + 1.0
    # Integer shift, NOT torch.pow(2.0, e).long(): CUDA's float64 pow returns
    # 2047.999... for 2**11, which truncates to 2047 and silently corrupts the
    # per-frame FFT size. Exponent is exact and small, so 1 << e is safe.
    exponent = (2.0 + torch.trunc(torch.log(length) / KLOG2)).long()

    # Tensor divisor, not the Python float: CUDA turns float64 `/ scalar` into a
    # reciprocal-multiply that is 1 ulp off true division, which perturbs the
    # window phase and moves the refined F0 by ~0.07 Hz. See dio()'s `t`.
    fs_t = torch.tensor(float(fs), device=dev, dtype=torch.float64)

    # Sort active frames by exponent (inactive ones to key 0) so each bucket
    # is a contiguous slice; the counts are the only host round trip.
    key = torch.where(active, exponent, torch.zeros_like(exponent))
    order = torch.argsort(key, stable=True)
    counts = torch.bincount(key).tolist()
    offset = counts[0]
    for e in range(1, len(counts)):
        if counts[e] == 0:
            continue
        sel = order[offset : offset + counts[e]]
        offset += counts[e]
        size = 1 << e
        # 2 + trunc(log2(2h + 1)) == e implies 2h + 1 < size / 2, so this static
        # width covers every frame in the bucket; the excess is masked below.
        n_max = size // 4 - 1
        h = hwl[sel]
        j = torch.arange(2 * n_max + 1, device=dev, dtype=torch.float64).unsqueeze(0)
        n = j - h.unsqueeze(1)
        valid = j <= (2 * h).unsqueeze(1)

        pos = t[sel].unsqueeze(1)
        base_time = n / fs_t
        index_raw = matlab_round((pos + base_time) * fs)
        tmp = (index_raw - 1.0) / fs_t - pos
        wlit = (2.0 * h + 1.0).unsqueeze(1) / fs_t
        mw = (
            0.42
            + 0.5 * torch.cos(2.0 * math.pi * tmp / wlit)
            + 0.08 * torch.cos(4.0 * math.pi * tmp / wlit)
        )
        mw = torch.where(valid, mw, torch.zeros_like(mw))

        # Negated central difference. With mw masked to zero past each frame's
        # own length, the interior formula reproduces upstream's last-sample
        # special case automatically.
        dw = torch.zeros_like(mw)
        dw[:, 0] = -mw[:, 1] / 2.0
        dw[:, 1:-1] = -(mw[:, 2:] - mw[:, :-2]) / 2.0
        dw[:, -1] = mw[:, -2] / 2.0
        dw = torch.where(valid, dw, torch.zeros_like(dw))

        sub = None if bounds is None else (bounds[0][sel], bounds[1][sel])
        seg = clamp_gather(x, (index_raw - 1).long(), sub)  # edge replication
        buf = torch.zeros(2, sel.numel(), size, dtype=torch.float64, device=dev)
        buf[0, :, : mw.shape[1]] = seg * mw
        buf[1, :, : dw.shape[1]] = seg * dw
        ms, ds = torch.fft.rfft(buf, dim=-1)

        num_i = ms.real * ds.imag - ms.imag * ds.real
        power = ms.real**2 + ms.imag**2

        init = f0[sel]
        tent = _fix_f0(power, num_i, size, fs, init, 2)
        # Evaluate the 6-harmonic pass unconditionally and select; clamp its
        # input so the discarded branch cannot produce NaN.
        refined = _fix_f0(power, num_i, size, fs, tent.clamp_min(KSG), 6)
        mean_f0 = torch.where(
            (tent <= 0.0) | (tent > init * 2), torch.zeros_like(tent), refined
        )
        # Overlarge correction (>20%) falls back to the initial F0, not to 0.
        out[sel] = torch.where((mean_f0 - init).abs() > init * 0.2, init, mean_f0)
    return out
