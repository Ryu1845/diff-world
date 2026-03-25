# Roadmap

## Phase 1: Core Differentiable WORLD Synthesizer

- [x] WORLD analysis wrapper (extract f0, sp, ap from audio via pyworld)
- [x] Alias-free harmonic synthesis (K=155 sinusoids, f_min=71 Hz)
- [x] Noise synthesis (Gaussian noise filtered by ap * sqrt(sp))
- [x] STFT-domain filtering for both components (N=1024, hop=256)
- [x] Baseline synthesizer output: y = g_h * h + g_n * n
- [x] Validate: invert WORLD features and compare to pyworld's own synthesis

## Phase 2: Parameter Compression/Decompression

- [x] Mel basis matrix M (80 bands) and pseudo-inverse
- [x] Spectral envelope compression: s = log10(M @ sqrt(sp) + eps)
- [x] Spectral envelope decompression: sp† = (clamp(M†, 0) @ (10^s - eps))^2
- [x] Aperiodicity compression/decompression (16 components, linear interpolation)
- [x] Validate: round-trip compress→decompress introduces minimal degradation

## Phase 3: Vocoder Extensions

- [ ] Residual postnet (SEANET or HiFi-GAN style temporal CNN)
- [ ] Optional local conditioning (Mel spectrogram of synthesizer output)
- [ ] Learned causal impulse response (w(0)=0)
- [ ] Alternate source excitation method (extract excitation spectrum, apply new envelope)

## Phase 4: Loss Functions

- [ ] Multi-spectrogram loss (S=6 scales, window sizes 2^(5+k), 75% overlap, κ=1)
- [ ] MSE acoustic feature loss
- [ ] MelGAN multi-scale discriminator (K=3 scales)
- [ ] Generator hinge loss + deep feature matching loss (λ=10)
- [ ] Combined hybrid objective

## Phase 5: Encoder

- [ ] Loudness extraction
- [ ] f0 extraction + octave-based pitch shifting
- [ ] Phonetic posteriorgram (BiLSTM, SVC only)
- [ ] Speaker embedding (SVC only)
- [ ] Aperiodicity passthrough

## Phase 6: Decoder

- [ ] SVC decoder: Conv Prenet → LSTM stack → Dense → Conv Postnet
- [ ] DDSP-TT decoder: 1x1 conv → unidirectional LSTM (256) → linear projection
- [ ] Sigmoid activation on aperiodicity output

## Phase 7: End-to-End Training

- [ ] Data pipeline (2-second clips, fs=22050, batch size=4)
- [ ] Autoencoder training loop (Adam, lr=1e-4)
- [ ] GAN warmup schedule (50K generator-only steps before enabling discriminator)
- [ ] Train SVC variants (DiffWORLDSynth, +Postnet+GAN, +Oracle, DiffWORLDExcite)
- [ ] Train DDSP-TT variant
- [ ] Evaluate: Mel reconstruction error on test set
