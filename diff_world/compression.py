import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mel_filterbank(
    n_mels: int = 80, n_fft: int = 1024, fs: int = 22050
) -> np.ndarray:
    """Build triangular Mel filterbank matrix.

    Returns:
        M: (n_mels, n_fft//2+1) Mel basis matrix
    """
    n_freqs = n_fft // 2 + 1
    mel_lo = 0.0
    mel_hi = 2595.0 * np.log10(1.0 + (fs / 2.0) / 700.0)
    mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    filterbank = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        for k in range(bin_points[m], bin_points[m + 1]):
            filterbank[m, k] = (k - bin_points[m]) / max(
                bin_points[m + 1] - bin_points[m], 1
            )
        for k in range(bin_points[m + 1], bin_points[m + 2]):
            filterbank[m, k] = (bin_points[m + 2] - k) / max(
                bin_points[m + 2] - bin_points[m + 1], 1
            )
    return filterbank


class ParameterCompressor(nn.Module):
    """Differentiable compression/decompression of WORLD acoustic features.

    Compresses spectral envelope (513 bins) to WORLD log Mel spectrogram (80 bands)
    and aperiodicity (513 bins) to 16 components via linear interpolation.
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 1024,
        fs: int = 22050,
        n_ap_compressed: int = 16,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_freqs = n_fft // 2 + 1
        self.n_ap_compressed = n_ap_compressed
        self.eps = eps

        # Build Mel basis and clamped pseudo-inverse
        M = mel_filterbank(n_mels, n_fft, fs)
        M_tensor = torch.from_numpy(M).float()
        M_pinv = torch.linalg.pinv(M_tensor)  # (n_freqs, n_mels)
        M_pinv_clamped = torch.clamp(M_pinv, min=0.0)

        self.register_buffer("mel_basis", M_tensor, persistent=False)
        self.register_buffer("mel_basis_pinv", M_pinv_clamped, persistent=False)

    def compress_sp(self, sp: torch.Tensor) -> torch.Tensor:
        """Compress spectral envelope to WORLD log Mel spectrogram.

        s = log10(M @ sqrt(sp) + eps)

        Args:
            sp: (B, T, n_freqs) spectral envelope (power spectrum)
        Returns:
            s: (B, T, n_mels) WORLD log Mel spectrogram
        """
        amp = torch.sqrt(sp + self.eps)
        mel_amp = torch.matmul(amp, self.mel_basis.T)
        return torch.log10(mel_amp + self.eps)

    def decompress_sp(self, s: torch.Tensor) -> torch.Tensor:
        """Decompress WORLD log Mel spectrogram to spectral envelope.

        sp† = [M†₀ @ clamp(10^s - eps, 0)]²

        Args:
            s: (B, T, n_mels) WORLD log Mel spectrogram
        Returns:
            sp_dag: (B, T, n_freqs) decompressed spectral envelope
        """
        amp_mel = torch.clamp(torch.pow(10.0, s) - self.eps, min=0.0)
        amp = torch.matmul(amp_mel, self.mel_basis_pinv.T)
        return amp ** 2

    def compress_ap(self, ap: torch.Tensor) -> torch.Tensor:
        """Compress aperiodicity via linear interpolation downsampling.

        Args:
            ap: (B, T, n_freqs) aperiodicity ratio [0, 1]
        Returns:
            a: (B, T, n_ap_compressed) compressed aperiodicity
        """
        B, T, n_f = ap.shape
        # F.interpolate needs (batch, channels, length)
        x = ap.reshape(B * T, 1, n_f)
        x = F.interpolate(x, size=self.n_ap_compressed, mode="linear", align_corners=True)
        return x.reshape(B, T, self.n_ap_compressed)

    def decompress_ap(self, a: torch.Tensor) -> torch.Tensor:
        """Decompress aperiodicity via linear interpolation upsampling.

        Args:
            a: (B, T, n_ap_compressed) compressed aperiodicity
        Returns:
            ap_dag: (B, T, n_freqs) decompressed aperiodicity [0, 1]
        """
        B, T, C = a.shape
        x = a.reshape(B * T, 1, C)
        x = F.interpolate(x, size=self.n_freqs, mode="linear", align_corners=True)
        return x.reshape(B, T, self.n_freqs)

    def compress(
        self, sp: torch.Tensor, ap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress both spectral envelope and aperiodicity."""
        return self.compress_sp(sp), self.compress_ap(ap)

    def decompress(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress both spectral envelope and aperiodicity."""
        return self.decompress_sp(s), self.decompress_ap(a)
