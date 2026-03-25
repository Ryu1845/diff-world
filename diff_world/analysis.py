import numpy as np
import pyworld as pw


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
    audio = audio.astype(np.float64)
    f0, t = pw.dio(audio, fs)
    f0 = pw.stonemask(audio, f0, t, fs)
    sp = pw.cheaptrick(audio, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(audio, f0, t, fs, fft_size=fft_size)
    return {"f0": f0, "sp": sp, "ap": ap, "t": t}


def world_synthesize_ref(
    f0: np.ndarray,
    sp: np.ndarray,
    ap: np.ndarray,
    fs: int = 22050,
) -> np.ndarray:
    """Reference synthesis via pyworld for comparison."""
    return pw.synthesize(f0, sp, ap, fs)
