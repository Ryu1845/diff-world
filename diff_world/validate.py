import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt

from .analysis import world_analyze
from .compression import ParameterCompressor, mel_filterbank
from .synthesizer import DiffWORLDSynthesizer


def compute_log_mel_spectrogram(
    audio: np.ndarray, fs: int = 22050, n_fft: int = 1024, hop: int = 256, n_mels: int = 80
) -> np.ndarray:
    """Compute log Mel spectrogram for visualization and metrics."""
    filterbank = mel_filterbank(n_mels, n_fft, fs)

    # STFT
    from numpy.lib.stride_tricks import sliding_window_view

    # Pad signal
    pad_len = n_fft // 2
    audio_padded = np.pad(audio, (pad_len, pad_len), mode="reflect")
    frames = sliding_window_view(audio_padded, n_fft)[::hop]
    window = np.hanning(n_fft)
    windowed = frames * window
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))

    mel_spec = filterbank @ spectrum.T
    log_mel = np.log10(mel_spec + 1e-10)
    return log_mel


def run_validation(audio_path: str, output_dir: str = "output"):
    """Compare differentiable WORLD synthesizer output to the original audio."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    audio, fs = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    if fs != 22050:
        raise ValueError(f"Expected 22050 Hz, got {fs}. Please resample first.")

    print(f"Loaded {audio_path}: {len(audio)} samples, {len(audio)/fs:.2f}s")

    # WORLD analysis
    features = world_analyze(audio, fs=fs)
    f0, sp, ap = features["f0"], features["sp"], features["ap"]
    print(f"WORLD analysis: {f0.shape[0]} frames, sp shape {sp.shape}")

    # Differentiable synthesis
    synth = DiffWORLDSynthesizer(fs=fs)
    with torch.no_grad():
        f0_t = torch.from_numpy(f0).float().unsqueeze(0)
        sp_t = torch.from_numpy(sp).float().unsqueeze(0)
        ap_t = torch.from_numpy(ap).float().unsqueeze(0)
        diff_out = synth(f0_t, sp_t, ap_t, n_samples=len(audio))
        diff_out = diff_out.squeeze(0).numpy()
    print(f"diff-world synthesis: {len(diff_out)} samples")

    # Trim to common length
    min_len = min(len(audio), len(diff_out))
    audio = audio[:min_len]
    diff_out = diff_out[:min_len]

    # Metrics (no amplitude normalization — compare raw outputs)
    mel_orig = compute_log_mel_spectrogram(audio, fs)
    mel_diff = compute_log_mel_spectrogram(diff_out, fs)

    # Trim mel spectrograms to common time length
    min_t = min(mel_orig.shape[1], mel_diff.shape[1])
    mel_orig = mel_orig[:, :min_t]
    mel_diff = mel_diff[:, :min_t]

    l1_diff = np.mean(np.abs(mel_orig - mel_diff))

    print(f"\nLog-Mel L1 errors:")
    print(f"  original vs diff-world: {l1_diff:.4f}")

    # RMS comparison
    rms_orig = np.sqrt(np.mean(audio**2))
    rms_diff = np.sqrt(np.mean(diff_out**2))
    print(f"\nRMS levels:")
    print(f"  original:   {rms_orig:.4f}")
    print(f"  diff-world: {rms_diff:.4f}")

    # Save audio
    sf.write(os.path.join(output_dir, "original.wav"), audio, fs)
    sf.write(os.path.join(output_dir, "diffworld.wav"), diff_out, fs)
    print(f"\nSaved WAVs to {output_dir}/")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    for ax, mel, title in zip(
        axes,
        [mel_orig, mel_diff],
        ["Original", "diff-world"],
    ):
        ax.imshow(mel, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("Mel bin")
    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {output_dir}/comparison.png")


def run_compression_validation(audio_path: str, output_dir: str = "output"):
    """Validate parameter compression/decompression round-trip fidelity."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    audio, fs = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if fs != 22050:
        raise ValueError(f"Expected 22050 Hz, got {fs}. Please resample first.")

    print(f"Loaded {audio_path}: {len(audio)} samples, {len(audio)/fs:.2f}s")

    # WORLD analysis
    features = world_analyze(audio, fs=fs)
    f0, sp, ap = features["f0"], features["sp"], features["ap"]
    print(f"WORLD analysis: {f0.shape[0]} frames, sp shape {sp.shape}")

    # Compress and decompress
    compressor = ParameterCompressor(fs=fs)
    with torch.no_grad():
        sp_t = torch.from_numpy(sp).float().unsqueeze(0)
        ap_t = torch.from_numpy(ap).float().unsqueeze(0)

        s, a = compressor.compress(sp_t, ap_t)
        sp_dag, ap_dag = compressor.decompress(s, a)

        print(f"Compressed: sp {sp_t.shape} -> s {s.shape}, ap {ap_t.shape} -> a {a.shape}")

    # Round-trip errors
    sp_log_orig = np.log10(sp + 1e-10)
    sp_log_dag = np.log10(sp_dag.squeeze(0).numpy() + 1e-10)
    sp_l1 = np.mean(np.abs(sp_log_orig - sp_log_dag))

    ap_orig = ap
    ap_dag_np = ap_dag.squeeze(0).numpy()
    ap_l1 = np.mean(np.abs(ap_orig - ap_dag_np))
    ap_max = np.max(np.abs(ap_orig - ap_dag_np))

    print(f"\nRound-trip errors:")
    print(f"  sp L1 (log domain): {sp_l1:.4f}")
    print(f"  ap L1:              {ap_l1:.6f}")
    print(f"  ap max:             {ap_max:.6f}")

    # Synthesize from original and decompressed features
    synth = DiffWORLDSynthesizer(fs=fs)
    with torch.no_grad():
        f0_t = torch.from_numpy(f0).float().unsqueeze(0)
        orig_out = synth(f0_t, sp_t, ap_t, n_samples=len(audio)).squeeze(0).numpy()
        comp_out = synth(f0_t, sp_dag, ap_dag, n_samples=len(audio)).squeeze(0).numpy()

    # Trim to common length
    min_len = min(len(audio), len(orig_out), len(comp_out))
    audio = audio[:min_len]
    orig_out = orig_out[:min_len]
    comp_out = comp_out[:min_len]

    # Synthesis comparison metrics
    mel_audio = compute_log_mel_spectrogram(audio, fs)
    mel_orig = compute_log_mel_spectrogram(orig_out, fs)
    mel_comp = compute_log_mel_spectrogram(comp_out, fs)

    min_t = min(mel_audio.shape[1], mel_orig.shape[1], mel_comp.shape[1])
    mel_audio = mel_audio[:, :min_t]
    mel_orig = mel_orig[:, :min_t]
    mel_comp = mel_comp[:, :min_t]

    l1_orig_vs_comp = np.mean(np.abs(mel_orig - mel_comp))
    l1_audio_vs_orig = np.mean(np.abs(mel_audio - mel_orig))
    l1_audio_vs_comp = np.mean(np.abs(mel_audio - mel_comp))

    spec_orig = np.abs(np.fft.rfft(orig_out))
    spec_comp = np.abs(np.fft.rfft(comp_out))
    spec_noise = spec_orig - spec_comp
    spectral_snr = 10 * np.log10(
        (np.sum(spec_orig**2) + 1e-10) / (np.sum(spec_noise**2) + 1e-10)
    )

    print(f"\nSynthesis comparison (original features vs decompressed):")
    print(f"  Log-Mel L1 (orig synth vs comp synth): {l1_orig_vs_comp:.4f}")
    print(f"  Log-Mel L1 (audio vs orig synth):      {l1_audio_vs_orig:.4f}")
    print(f"  Log-Mel L1 (audio vs comp synth):      {l1_audio_vs_comp:.4f}")
    print(f"  Spectral SNR:                          {spectral_snr:.1f} dB")

    # Save audio
    sf.write(os.path.join(output_dir, "original.wav"), audio, fs)
    sf.write(os.path.join(output_dir, "diffworld_orig.wav"), orig_out, fs)
    sf.write(os.path.join(output_dir, "diffworld_compressed.wav"), comp_out, fs)
    print(f"\nSaved WAVs to {output_dir}/")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Row 1: spectral envelope as log Mel spectrogram
    s_np = s.squeeze(0).numpy().T  # (n_mels, T)
    s_dag_np = compressor.compress_sp(sp_dag).squeeze(0).numpy().T
    world_mel_orig = compressor.compress_sp(sp_t).squeeze(0).numpy().T

    for ax, data, title in zip(
        axes[0],
        [world_mel_orig, s_np, s_dag_np],
        ["Original WORLD log Mel", "Compressed s", "Recompressed from sp†"],
    ):
        ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("Mel bin")

    # Row 2: aperiodicity
    a_np = a.squeeze(0).numpy().T  # (16, T)
    for ax, data, title in zip(
        axes[1],
        [ap.T, a_np, ap_dag_np.T],
        ["Original ap (513)", "Compressed a (16)", "Decompressed ap† (513)"],
    ):
        ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_ylabel("Freq bin")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 2].set_xlabel("Frame")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "compression_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved compression comparison plot to {output_dir}/compression_comparison.png")
