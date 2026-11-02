import os
import json
import wave
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class FFmpegError(RuntimeError):
    """Raised when the internal FFmpeg binary encounters an execution error."""


class FFmpegBridge:
    """
    Interfaces exclusively with the package's built-in audio-only FFmpeg and FFprobe binaries.
    Provides fallback to system binaries only if explicitly permitted for development.
    """

    def __init__(self, allow_system_fallback: bool = False):
        package_dir = Path(__file__).resolve().parent
        bin_dir = package_dir / "bin"

        self.ffmpeg_path = bin_dir / "ffmpeg"
        self.ffprobe_path = bin_dir / "ffprobe"

        if not self.ffmpeg_path.is_file() or not os.access(self.ffmpeg_path, os.X_OK):
            if allow_system_fallback and shutil.which("ffmpeg"):
                self.ffmpeg_path = Path(shutil.which("ffmpeg"))
                self.ffprobe_path = Path(shutil.which("ffprobe") or "ffprobe")
            else:
                raise FFmpegError(
                    f"Built-in stripped FFmpeg binary not found at '{self.ffmpeg_path}'. "
                    "Please run 'bash scripts/build_custom_ffmpeg.sh' to compile the embedded engine."
                )

    def probe_audio(self, file_path: str | Path) -> Dict[str, Any]:
        """Extracts stream metadata (sample rate, channels, duration, codec) via built-in ffprobe."""
        file_path = str(file_path)
        cmd = [
            str(self.ffprobe_path),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-select_streams", "a:0",
            file_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise FFmpegError(f"ffprobe failed on '{file_path}': {res.stderr.strip()}")

        data = json.loads(res.stdout or "{}")
        streams = data.get("streams", [])
        if not streams:
            raise FFmpegError(f"No valid audio stream found in '{file_path}'.")

        stream = streams[0]
        fmt = data.get("format", {})
        return {
            "codec_name": stream.get("codec_name", "unknown"),
            "sample_rate": int(stream.get("sample_rate", 44100)),
            "channels": int(stream.get("channels", 2)),
            "duration": float(stream.get("duration") or fmt.get("duration") or 0.0),
            "bit_rate": int(stream.get("bit_rate") or fmt.get("bit_rate") or 0),
        }

    def load_audio_float32(
        self, file_path: str | Path, target_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Decodes any supported audio file into a normalized float64 NumPy array of shape (samples, channels)
        in range [-1.0, 1.0] using the built-in FFmpeg via raw 32-bit float PCM pipe.
        """
        info = self.probe_audio(file_path)
        sr = target_sr if target_sr is not None else info["sample_rate"]
        # Upsample low-bandwidth legacy recordings to at least 44.1kHz so high-frequency restoration has headroom
        if sr < 44100:
            sr = 44100
        channels = info["channels"]

        cmd = [
            str(self.ffmpeg_path),
            "-v", "error",
            "-i", str(file_path),
            "-vn",
            "-acodec", "pcm_f32le",
            "-ar", str(sr),
            "-ac", str(channels),
            "-f", "f32le",
            "pipe:1",
        ]

        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            raise FFmpegError(
                f"FFmpeg decoding failed for '{file_path}': {res.stderr.decode('utf-8', errors='replace')}"
            )

        audio_1d = np.frombuffer(res.stdout, dtype=np.float32).astype(np.float64)
        if audio_1d.size % channels != 0:
            trim = (audio_1d.size // channels) * channels
            audio_1d = audio_1d[:trim]

        audio = audio_1d.reshape(-1, channels)
        return audio, sr

    def run_filtergraph_and_encode(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str | Path,
        filter_chain: List[str],
        output_format: str = "flac",
        output_bitrate: str = "320k",
    ) -> None:
        """
        Pipes a NumPy float64 array (samples, channels) through the custom FFmpeg audio filtergraph
        and encodes directly to the requested output format.
        """
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        samples, channels = audio.shape
        raw_f32 = np.clip(audio, -1.0, 1.0).astype(np.float32).tobytes()

        af_string = ",".join(filter_chain) if filter_chain else "anull"
        codec_args = self._get_encoder_args(output_format, output_bitrate)

        cmd = [
            str(self.ffmpeg_path),
            "-y",
            "-v", "error",
            "-f", "f32le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-i", "pipe:0",
            "-af", af_string,
            *codec_args,
            str(output_path),
        ]

        res = subprocess.run(cmd, input=raw_f32, capture_output=True)
        if res.returncode != 0:
            err_msg = res.stderr.decode("utf-8", errors="replace").strip()
            raise FFmpegError(f"FFmpeg filtergraph/encoding failed: {err_msg}")

    @staticmethod
    def _get_encoder_args(output_format: str, bitrate: str) -> List[str]:
        fmt = output_format.lower().lstrip(".")
        if fmt == "flac":
            return ["-c:a", "flac", "-compression_level", "8"]
        elif fmt == "wav":
            return ["-c:a", "pcm_s24le"]
        elif fmt == "mp3":
            return ["-c:a", "libmp3lame", "-b:a", bitrate, "-q:a", "0"]
        elif fmt in ("ogg", "vorbis"):
            return ["-c:a", "vorbis", "-strict", "-2", "-b:a", bitrate]
        elif fmt == "opus":
            return ["-c:a", "opus", "-strict", "-2", "-b:a", bitrate]
        elif fmt in ("aac", "m4a"):
            return ["-c:a", "aac", "-b:a", bitrate]
        else:
            return ["-c:a", "pcm_s24le"]