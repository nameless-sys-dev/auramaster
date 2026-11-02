import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List

from .config import RemasterConfig
from .engine import RemasterEngine

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".wv", ".ape"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auramaster",
        description=(
            "AuraMaster: Intelligent Audio Restoration & Remastering CLI (Linux).\n"
            "Powered by hybrid SciPy DSP and an embedded stripped audio-only FFmpeg engine."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Input audio file or directory of audio files.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output audio file or directory (Required unless --analyze-only is used).",
    )
    parser.add_argument(
        "-p", "--preset",
        choices=["auto", "vinyl", "tape", "digital_clip", "tinny"],
        default="auto",
        help="Restoration preset baseline (default: auto).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run intelligent audio diagnostics and print the analysis report without remastering.",
    )
    parser.add_argument(
        "--format",
        choices=["flac", "wav", "mp3", "ogg", "opus", "aac"],
        default="flac",
        help="Output audio format (default: flac, or inferred from output filename).",
    )
    parser.add_argument(
        "--bitrate",
        default="320k",
        help="Bitrate for lossy output codecs (default: 320k).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Target sample rate in Hz (e.g. 48000, 96000). Preserves native rate by default.",
    )

    # Restoration overrides
    group_dsp = parser.add_argument_group("Manual DSP Intensity Overrides (0.0 to 1.0, or dB)")
    group_dsp.add_argument("--declick", type=float, default=None, help="Vinyl/impulse de-click sensitivity (0.0-1.0).")
    group_dsp.add_argument("--declip", type=float, default=None, help="De-clipping & dropout reconstruction (0.0-1.0).")
    group_dsp.add_argument("--denoise", type=float, default=None, help="White/static noise removal strength (0.0-1.0).")
    group_dsp.add_argument("--detinny", type=float, default=None, help="Tinny mid-range harshness removal (0.0-1.0).")
    group_dsp.add_argument("--bass-boost", type=float, default=None, help="Bass restoration EQ boost in dB (0.0-10.0).")
    group_dsp.add_argument("--subharmonics", type=float, default=None, help="Sub-harmonic synthesis mix (0.0-0.6).")
    group_dsp.add_argument("--high-boost", type=float, default=None, help="High-frequency EQ boost in dB (0.0-10.0).")
    group_dsp.add_argument("--exciter", type=float, default=None, help="Chebyshev harmonic exciter mix (0.0-0.6).")
    group_dsp.add_argument("--stereo-width", type=float, default=None, help="Stereo width multiplier (0.5-1.8).")

    # Stage disable flags
    group_toggle = parser.add_argument_group("Pipeline Stage Toggles")
    group_toggle.add_argument("--no-declick", action="store_true", help="Disable de-clicking.")
    group_toggle.add_argument("--no-declip", action="store_true", help="Disable de-clipping.")
    group_toggle.add_argument("--no-denoise", action="store_true", help="Disable spectral de-noising.")
    group_toggle.add_argument("--no-detinny", action="store_true", help="Disable anti-tinny filter.")
    group_toggle.add_argument("--no-bass", action="store_true", help="Disable bass/sub-harmonic restoration.")
    group_toggle.add_argument("--no-highs", action="store_true", help="Disable high-frequency exciter.")
    group_toggle.add_argument("--no-limiter", action="store_true", help="Disable EBU R128 loudness normalization & limiter.")

    # Mastering & Reporting
    parser.add_argument("--lufs", type=float, default=-14.0, help="Target integrated loudness in LUFS (default: -14.0).")
    parser.add_argument("--true-peak", type=float, default=-1.0, help="Target true peak ceiling in dBTP (default: -1.0).")
    parser.add_argument("--report-json", type=str, default=None, help="Write full diagnostic JSON report to file.")
    parser.add_argument("--allow-system-ffmpeg", action="store_true", help="Allow fallback to system FFmpeg if custom build is missing.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    return parser


def _print_comparison_summary(result: dict) -> None:
    pre = result["pre_analysis"]
    post = result["post_analysis"]
    params = result["applied_parameters"]

    print("\n" + "=" * 68)
    print(f" AURAMASTER REMASTERING REPORT: {Path(result['input_file']).name}")
    print("=" * 68)
    print(f" Output File        : {result['output_file']}")
    print(f" Sample Rate / Ch   : {post['sample_rate']} Hz / {post['channels']} ch ({post['duration_seconds']}s)")
    print("-" * 68)
    print(f" {'Metric':<28} | {'Before':<16} | {'After':<16}")
    print("-" * 68)
    print(f" {'Peak Level':<28} | {pre['peak_dbfs']:>11.2f} dBFS | {post['peak_dbfs']:>11.2f} dBFS")
    print(f" {'RMS Loudness':<28} | {pre['rms_dbfs']:>11.2f} dBFS | {post['rms_dbfs']:>11.2f} dBFS")
    print(f" {'Noise Floor':<28} | {pre['noise_floor_dbfs']:>11.2f} dBFS | {post['noise_floor_dbfs']:>11.2f} dBFS")
    print(f" {'Estimated SNR':<28} | {pre['estimated_snr_db']:>11.2f} dB   | {post['estimated_snr_db']:>11.2f} dB")
    print(f" {'Hard Clip Events':<28} | {pre['hard_clip_events']:>16d} | {post['hard_clip_events']:>16d}")
    print(f" {'Click/Pop Density':<28} | {pre['impulse_click_density']:>12.2f} /s  | {post['impulse_click_density']:>12.2f} /s")
    print(f" {'Tinny Harshness Imbalance':<28} | {pre['tinny_ratio_db']:>11.2f} dB   | {post['tinny_ratio_db']:>11.2f} dB")
    print(f" {'Bass Deficit':<28} | {pre['bass_deficit_db']:>11.2f} dB   | {post['bass_deficit_db']:>11.2f} dB")
    print(f" {'High-Freq Drop-off':<28} | {pre['high_freq_dropoff_db']:>11.2f} dB   | {post['high_freq_dropoff_db']:>11.2f} dB")
    print(f" {'95% Spectral Rolloff':<28} | {pre['spectral_rolloff_95_hz']:>11.1f} Hz   | {post['spectral_rolloff_95_hz']:>11.1f} Hz")
    print("-" * 68)
    print(" Applied Intelligent Parameters:")
    for k, v in params.items():
        print(f"   - {k:<24}: {v}")
    print("=" * 68 + "\n")


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s",
    )

    cfg = RemasterConfig.from_preset(args.preset)
    cfg.output_format = args.format
    cfg.output_bitrate = args.bitrate
    cfg.target_sample_rate = args.sample_rate
    cfg.target_lufs = args.lufs
    cfg.target_true_peak_db = args.true_peak

    # Apply stage toggles
    cfg.enable_declick = not args.no_declick
    cfg.enable_declip = not args.no_declip
    cfg.enable_denoise = not args.no_denoise
    cfg.enable_detinny = not args.no_detinny
    cfg.enable_bass_restore = not args.no_bass
    cfg.enable_high_restore = not args.no_highs
    cfg.enable_mastering_limiter = not args.no_limiter

    # Apply manual overrides
    if args.declick is not None:
        cfg.declick_sensitivity = args.declick
    if args.declip is not None:
        cfg.declip_strength = args.declip
    if args.denoise is not None:
        cfg.denoise_strength = args.denoise
    if args.detinny is not None:
        cfg.detinny_strength = args.detinny
    if args.bass_boost is not None:
        cfg.bass_boost_db = args.bass_boost
    if args.subharmonics is not None:
        cfg.subharmonic_mix = args.subharmonics
    if args.high_boost is not None:
        cfg.high_boost_db = args.high_boost
    if args.exciter is not None:
        cfg.harmonic_exciter_mix = args.exciter
    if args.stereo_width is not None:
        cfg.stereo_width = args.stereo_width

    try:
        engine = RemasterEngine(config=cfg, allow_system_ffmpeg_fallback=args.allow_system_ffmpeg)
        in_path = Path(args.input)

        if not in_path.exists():
            print(f"[ERROR] Input path '{in_path}' does not exist.", file=sys.stderr)
            return 1

        if args.analyze_only:
            report = engine.analyze_file(in_path)
            print(json.dumps(report.to_dict(), indent=2))
            if args.report_json:
                Path(args.report_json).write_text(json.dumps(report.to_dict(), indent=2))
            return 0

        if not args.output:
            print("[ERROR] -o/--output is required when remastering.", file=sys.stderr)
            return 1

        out_path = Path(args.output)
        all_reports = []

        if in_path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
            files = sorted(
                p for p in in_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not files:
                print(f"[ERROR] No supported audio files found in '{in_path}'.", file=sys.stderr)
                return 1

            for f in files:
                target_file = out_path / f"{f.stem}_remastered.{cfg.output_format}"
                res = engine.remaster(f, target_file)
                _print_comparison_summary(res)
                all_reports.append(res)
        else:
            res = engine.remaster(in_path, out_path)
            _print_comparison_summary(res)
            all_reports.append(res)

        if args.report_json:
            payload = all_reports[0] if len(all_reports) == 1 else all_reports
            Path(args.report_json).write_text(json.dumps(payload, indent=2))

        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())