# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Reproduce the paper "Differentiable WORLD Synthesizer-based Neural Vocoder with Application to End-to-End Audio Style Transfer" (Nercessian, iZotope — AES Convention paper, `./paper/main.tex`). The goal is a working differentiable WORLD synthesizer and its application to end-to-end singing voice conversion (SVC) and DDSP timbre transfer tasks.

## Setup

- Python 3.13, managed with uv (`pyproject.toml`)
- Install deps: `uv sync`
- Run: `uv run python main.py`

## Paper Architecture Summary

The system has three main blocks: **Encoder**, **Decoder**, and **Vocoder** (see `paper/architecture.pdf`).

### Differentiable WORLD Synthesizer (Vocoder)

Core component. Synthesizes audio from WORLD acoustic features (f0, spectral envelope `sp`, aperiodicity `ap`):

1. **Harmonic synthesis**: Alias-free pulse train via sum of K=155 sinusoids per the f0 contour, filtered by `(1-ap) * sqrt(sp)` in STFT domain.
2. **Noise synthesis**: Gaussian noise filtered by `ap * sqrt(sp)` in STFT domain.
3. **Output**: `y = g_h * harmonic + g_n * noise` (gains=1 during training).

Key parameters: fs=22050 Hz, STFT N=1024 (513 bins), hop=N/4, f_min=71 Hz.

### Parameter Compression/Decompression

- **Spectral envelope → WORLD log Mel spectrogram** (M=80 bands): `s = log10(M @ sqrt(sp) + eps)`. Decompression via pseudo-inverse: `sp† = (max(M†, 0) @ (10^s - eps))^2`.
- **Aperiodicity**: Interpolated down to 16 components, upsampled back via linear interpolation.

### Extensions

- **Residual postnets** (SEANET or HiFi-GAN temporal CNN): `y_d = g_0*y_0 + g_P*P(y_0)`, optionally conditioned on Mel spectrogram of synthesizer output.
- **Learned impulse response**: Causal filter w with w(0)=0, applied as convolution.
- **Alternate source excitation method**: Extract excitation spectrum from source audio, apply new spectral envelope for formant transformation (no f0/ap needed).

### Encoder (for style transfer tasks)

- Extracts from source audio: loudness L, pitch F (via WORLD), phonetic posteriorgram P (BiLSTM trained on TIMIT, SVC only), aperiodicity ap.
- Pitch shift to target octave for SVC; speaker embedding S from target (SVC only).

### Decoder

- SVC: Maps `[L, log(F_shifted), P, S]` → WORLD log Mel spectrogram `s`.
- DDSP-TT: Maps `[L, log(F_shifted)]` → `[s, a]` (compressed sp + compressed ap). Architecture: 1x1 conv → unidirectional LSTM (256 units) → linear projection. Sigmoid on aperiodicity output.

### Training

- **Losses**: MSE on acoustic features (α=1), multi-spectrogram loss (S=6 scales, window sizes 2^(5+k), 75% overlap, κ=1, β=1).
- **Adversarial** (optional): MelGAN multi-scale discriminator (K=3 scales, downsampled by 2^(k-1)), hinge loss (μ=1), deep feature matching loss (λ=10). GAN warmup: 50K steps generator-only before enabling discriminator.
- **Optimizer**: Adam, lr=1e-4, batch size=4, 2-second clips, 1M steps.
- **Constrained-manifold targets**: When no postnet, compare against WORLD-resynthesized audio instead of original.

### System Variants (SVC)

| Variant | Vocoder | Postnet | GAN | Target |
|---|---|---|---|---|
| E2E+DiffWORLDSynth | synthesis | no | no | original audio |
| E2E+DiffWORLDSynth+Postnet+GAN | synthesis | yes | yes | original audio |
| E2E+DiffWORLDSynth+Oracle | synthesis | no | no | WORLD-resynthesized |
| E2E+DiffWORLDExcite | excitation | no | no | original audio |
