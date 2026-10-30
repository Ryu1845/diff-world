from collections.abc import Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .world import cheaptrick, d4c, dio, stonemask
from .world.dio import num_frames

_DEFAULT_SEED = 0


def world_analyze_batch(
    audios: Sequence[np.ndarray | torch.Tensor] | torch.Tensor,
    fs: int = 22050,
    fft_size: int = 1024,
    device: torch.device | str | None = None,
    seed: int = _DEFAULT_SEED,
) -> list[dict[str, torch.Tensor]]:
    """Extract WORLD acoustic features from a batch of mono utterances.

    audios: a (B, L) tensor of equal-length clips, or a sequence of 1-D
    arrays/tensors of any lengths. Every stage runs once for the whole batch:
    the frames of all utterances are analyzed together, each one clamped to
    its own utterance's samples, so a batch costs a fixed handful of host
    syncs and kernel launches rather than a handful per utterance. That is
    the win on CUDA. On CPU, keep batches small (B <= 4 clips of 2 s): with
    glibc's default malloc, every large temporary is mmap'd fresh and
    page-faulted in, which costs more than the launches batching saves.

    Returns one dict per utterance, with tensors on `device` (default: the
    input's device, CPU for numpy):
        f0:  (T_frames,) fundamental frequency per frame (0 = unvoiced)
        sp:  (T_frames, fft_size//2+1) spectral envelope (power spectrum)
        ap:  (T_frames, fft_size//2+1) aperiodicity ratio in [0, 1]
        t:   (T_frames,) frame timestamps in seconds
    """
    if isinstance(audios, torch.Tensor) and audios.dim() == 2:
        dev = torch.device(device) if device is not None else audios.device
        x = audios.to(dev, torch.float64)
        lengths = [x.shape[1]] * x.shape[0]
    else:
        rows = [a if isinstance(a, torch.Tensor) else torch.from_numpy(np.asarray(a)) for a in audios]
        dev = torch.device(device) if device is not None else rows[0].device
        lengths = [r.shape[0] for r in rows]
        x = pad_sequence([r.to(torch.float64) for r in rows], batch_first=True).to(dev)
    B, L = x.shape

    # CheapTrick and D4C dither their inputs; a fixed seed keeps runs reproducible.
    generator = torch.Generator(device=dev).manual_seed(seed)
    f0, t = dio(x, fs, lengths=lengths)

    # Flatten every utterance's frames; indices are built on the host, so the
    # per-utterance frame counts never need a device round trip.
    frames = [num_frames(n, fs) for n in lengths]
    utt = torch.repeat_interleave(torch.arange(B), torch.tensor(frames))
    idx = torch.cat([torch.arange(n) for n in frames])
    bounds = ((utt * L).to(dev), torch.tensor(lengths)[utt].to(dev))
    utt, idx = utt.to(dev), idx.to(dev)
    f0, t_flat = f0[utt, idx], t[idx]

    x = x.reshape(-1)
    f0 = stonemask(x, f0, t_flat, fs, bounds=bounds)
    sp = cheaptrick(x, f0, t_flat, fs, fft_size=fft_size, generator=generator, bounds=bounds)
    ap = d4c(x, f0, t_flat, fs, fft_size=fft_size, generator=generator, bounds=bounds)
    return [
        {"f0": a, "sp": b, "ap": c, "t": t[:n]}
        for a, b, c, n in zip(f0.split(frames), sp.split(frames), ap.split(frames), frames)
    ]


def world_analyze(
    audio: np.ndarray, fs: int = 22050, fft_size: int = 1024
) -> dict[str, np.ndarray]:
    """Extract WORLD acoustic features from mono audio.

    Returns dict with:
        f0:  (T_frames,) fundamental frequency per frame (0 = unvoiced)
        sp:  (T_frames, fft_size//2+1) spectral envelope (power spectrum)
        ap:  (T_frames, fft_size//2+1) aperiodicity ratio in [0, 1]
        t:   (T_frames,) frame timestamps in seconds
    """
    out = world_analyze_batch([np.asarray(audio, dtype=np.float64)], fs, fft_size)[0]
    return {k: v.cpu().numpy() for k, v in out.items()}
