import sys

from diff_world.validate import run_validation, run_compression_validation


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_wav> [output_dir] [--compress]")
        print("  WAV must be mono, 22050 Hz.")
        sys.exit(1)

    compress = "--compress" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    audio_path = args[0]
    output_dir = args[1] if len(args) > 1 else "output"

    if compress:
        run_compression_validation(audio_path, output_dir)
    else:
        run_validation(audio_path, output_dir)


if __name__ == "__main__":
    main()
