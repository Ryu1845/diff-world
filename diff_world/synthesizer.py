from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .compression import ParameterCompressor


class DiffWORLDSynthesizer(nn.Module):
    """Differentiable WORLD synthesizer (baseline, no learnable parameters).

    Synthesizes audio from WORLD acoustic features (f0, spectral envelope,
    aperiodicity) using alias-free harmonic + noise synthesis with
    STFT-domain filtering.
    """

    def __init__(
        self,
        fs: int = 22050,
        fft_size: int = 1024,
        hop_size: int = 256,
        f_min: float = 71.0,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.fs = fs
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.f_min = f_min
        self.eps = eps
        self.n_freqs = fft_size // 2 + 1
        self.K = int(fs / 2 / f_min)  # 155 for fs=22050, f_min=71

        self.register_buffer(
            "window", torch.hann_window(fft_size), persistent=False
        )

    def forward(
        self,
        f0: torch.Tensor,
        sp: torch.Tensor,
        ap: torch.Tensor,
        n_samples: int | None = None,
        g_h: float = 1.0,
        g_n: float = 1.0,
    ) -> torch.Tensor:
        """Synthesize audio from WORLD features.

        Args:
            f0: (B, T_frames) fundamental frequency in Hz, 0 = unvoiced
            sp: (B, T_frames, N//2+1) spectral envelope (power spectrum)
            ap: (B, T_frames, N//2+1) aperiodicity ratio [0, 1]
            n_samples: desired output length in samples (optional)
            g_h: harmonic gain (default 1.0)
            g_n: noise gain (default 1.0)

        Returns:
            y: (B, n_samples) synthesized waveform
        """
        B, T_frames, _ = sp.shape

        if n_samples is None:
            n_samples = T_frames * self.hop_size

        # Enforce unvoiced: where f0=0, set ap=1 (pure noise)
        unvoiced = (f0 == 0).unsqueeze(-1).float()  # (B, T_frames, 1)
        ap = ap * (1 - unvoiced) + unvoiced

        # Harmonic synthesis
        h = self._harmonic_synthesis(f0, sp, ap, n_samples)

        # Noise synthesis
        n = self._noise_synthesis(sp, ap, n_samples)

        return g_h * h + g_n * n

    def _harmonic_synthesis(
        self,
        f0: torch.Tensor,
        sp: torch.Tensor,
        ap: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Synthesize harmonic component via alias-free pulse train + STFT filtering."""
        B = f0.shape[0]

        # Interpolate f0 to sample rate
        f0_interp = F.interpolate(
            f0.unsqueeze(1), size=n_samples, mode="linear", align_corners=False
        ).squeeze(1)  # (B, n_samples)

        # Cumulative phase (in radians)
        phase_base = torch.cumsum(f0_interp / self.fs, dim=-1)  # cycles
        phase_base = 2 * math.pi * phase_base  # (B, n_samples)

        # Harmonic indices [1, 2, ..., K]
        k_vals = torch.arange(
            1, self.K + 1, device=f0.device, dtype=f0.dtype
        )  # (K,)

        # Compute all harmonic phases: (B, n_samples, K)
        phases = phase_base.unsqueeze(-1) * k_vals

        # Nyquist masking: zero out harmonics above fs/2
        inst_freq = f0_interp.unsqueeze(-1) * k_vals  # (B, n_samples, K)
        nyquist = self.fs / 2.0
        active = (inst_freq < nyquist).float()

        # Energy normalization: sqrt(2/N) gives unit RMS per pulse
        num_active = active.sum(dim=-1, keepdim=True).clamp(min=1)
        c_k = active * torch.sqrt(2.0 / num_active)

        # Alias-free pulse train
        e_h = (c_k * torch.sin(phases)).sum(dim=-1)  # (B, n_samples)

        # STFT-domain filtering: h = ISTFT[(1 - ap) * sqrt(sp) * STFT(e_h)]
        h = self._stft_filter(e_h, (1 - ap) * torch.sqrt(sp + self.eps), n_samples)
        return h

    def _noise_synthesis(
        self,
        sp: torch.Tensor,
        ap: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Synthesize noise component via filtered Gaussian noise."""
        B = sp.shape[0]

        e_n = torch.randn(B, n_samples, device=sp.device, dtype=sp.dtype)

        # STFT-domain filtering: n = ISTFT[ap * sqrt(sp) * STFT(e_n)]
        n = self._stft_filter(e_n, ap * torch.sqrt(sp + self.eps), n_samples)
        return n

    def _stft_filter(
        self,
        signal: torch.Tensor,
        spectral_filter: torch.Tensor,
        n_samples: int,
    ) -> torch.Tensor:
        """Apply frame-level spectral filter in STFT domain.

        Args:
            signal: (B, n_samples) time-domain input
            spectral_filter: (B, T_frames, N//2+1) filter to apply
            n_samples: desired output length

        Returns:
            filtered: (B, n_samples) filtered time-domain output
        """
        # STFT
        X = torch.stft(
            signal,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.window,
            return_complex=True,
        )  # (B, N//2+1, T_stft)

        T_stft = X.shape[-1]

        # Interpolate filter from WORLD frames to STFT frames
        # spectral_filter: (B, T_frames, F) -> (B, F, T_stft)
        filt = F.interpolate(
            spectral_filter.transpose(1, 2),
            size=T_stft,
            mode="linear",
            align_corners=False,
        )  # (B, F, T_stft)

        # Apply filter
        X_filtered = X * filt

        # ISTFT
        filtered = torch.istft(
            X_filtered,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self.window,
            length=n_samples,
        )  # (B, n_samples)

        return filtered

    def forward_compressed(
        self,
        f0: torch.Tensor,
        s: torch.Tensor,
        a: torch.Tensor,
        compressor: ParameterCompressor,
        n_samples: int | None = None,
        g_h: float = 1.0,
        g_n: float = 1.0,
    ) -> torch.Tensor:
        """Synthesize audio from compressed WORLD features.

        Args:
            f0: (B, T_frames) fundamental frequency in Hz
            s: (B, T_frames, n_mels) WORLD log Mel spectrogram
            a: (B, T_frames, n_ap_compressed) compressed aperiodicity
            compressor: ParameterCompressor for decompression
            n_samples: desired output length in samples
            g_h: harmonic gain (default 1.0)
            g_n: noise gain (default 1.0)

        Returns:
            y: (B, n_samples) synthesized waveform
        """
        sp_dag, ap_dag = compressor.decompress(s, a)
        return self.forward(f0, sp_dag, ap_dag, n_samples, g_h, g_n)
