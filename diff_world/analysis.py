import numpy as np
import torch

from .world import cheaptrick, d4c, dio, stonemask

_DEFAULT_SEED = 0


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
    x = torch.as_tensor(np.asarray(audio, dtype=np.float64))
    # CheapTrick and D4C dither their inputs; a fixed seed keeps runs reproducible.
    generator = torch.Generator(device=x.device).manual_seed(_DEFAULT_SEED)
    f0, t = dio(x, fs)
    f0 = stonemask(x, f0, t, fs)
    sp = cheaptrick(x, f0, t, fs, fft_size=fft_size, generator=generator)
    ap = d4c(x, f0, t, fs, fft_size=fft_size, generator=generator)
    return {
        "f0": f0.numpy(),
        "sp": sp.numpy(),
        "ap": ap.numpy(),
        "t": t.numpy(),
    }
