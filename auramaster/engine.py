import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import RemasterConfig
from .ffmpeg_bridge import FFmpegBridge
from .analyzer import AudioAnalyzer, AnalysisReport
from .dsp import AudioDSP

logger = logging.getLogger("auramaster")


class RemasterEngine:
    """
    Main orchestrator for the AuraMaster hybrid restoration and remastering pipeline.
    Combines:
    1. Embedded FFmpeg lossless float32 decoding
    2. Pre-restoration intelligent signal analysis
    3. Python NumPy/SciPy scientific spectral & waveform reconstruction
    4. Embedded FFmpeg psychoacoustic filtergraph & EBU R128 mastering
    5. Post-restoration verification analysis
    """

    def __init__(
        self,
        config: Optional[RemasterConfig] = None,
        allow_system_ffmpeg_fallback: bool = False,
    ):
        self.config = config or RemasterConfig()
        self.bridge = FFmpegBridge(allow_system_fallback=allow_system_ffmpeg_fallback)

    def analyze_file(self, input_path: str | Path) -> AnalysisReport:
        """Decodes and analyzes an audio file without modifying it."""
        audio, sr = self.bridge.load_audio_float32(input_path, self.config.target_sample_rate)
        return AudioAnalyzer.analyze(audio, sr)

    def remaster(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Executes the full intelligent restoration and mastering pipeline on `input_path`
        and writes the remastered audio to `output_path`.
        Returns a dictionary containing pre-analysis, applied parameters, and post-analysis metrics.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading and decoding '%s' via embedded FFmpeg...", input_path.name)
        audio, sr = self.bridge.load_audio_float32(input_path, self.config.target_sample_rate)

        logger.info("Analyzing spectral and temporal characteristics...")
        pre_report = AudioAnalyzer.analyze(audio, sr)
        active_params = self._resolve_parameters(pre_report)
        logger.info("Resolved restoration parameters: %s", active_params)

        # --- Stage 1: Python Scientific DSP Reconstruction ---
        logger.info("Stage 1/6: Removing DC offset and subsonic rumble...")
        audio = AudioDSP.remove_dc_and_rumble(audio, sr)

        if self.config.enable_declip and active_params["declip_strength"] > 0.05:
            logger.info(
                "Stage 2/6: Reconstructing clipped peaks & dropouts (strength=%.2f)...",
                active_params["declip_strength"],
            )
            audio = AudioDSP.reconstruct_clipped_and_dropouts(
                audio, sr, strength=active_params["declip_strength"]
            )

        if self.config.enable_denoise and active_params["denoise_strength"] > 0.05:
            logger.info(
                "Stage 3/6: Applying MMSE Wiener spectral denoising (strength=%.2f)...",
                active_params["denoise_strength"],
            )
            audio = AudioDSP.spectral_denoise(
                audio, sr, strength=active_params["denoise_strength"]
            )

        if self.config.enable_detinny and active_params["detinny_strength"] > 0.05:
            logger.info(
                "Stage 4/6: Taming tinny mid-range resonances & restoring warmth (strength=%.2f)...",
                active_params["detinny_strength"],
            )
            audio = AudioDSP.suppress_tinny_resonances(
                audio, sr, strength=active_params["detinny_strength"]
            )

        if self.config.enable_bass_restore:
            logger.info(
                "Stage 5/6: Synthesizing sub-harmonics (mix=%.2f, boost=%.1fdB)...",
                active_params["subharmonic_mix"],
                active_params["bass_boost_db"],
            )
            audio = AudioDSP.synthesize_subharmonics(
                audio,
                sr,
                mix=active_params["subharmonic_mix"],
                bass_boost_db=active_params["bass_boost_db"] * 0.6,
            )

        if self.config.enable_high_restore:
            logger.info(
                "Stage 6/6: Reconstructing upper harmonics via Chebyshev exciter (mix=%.2f, boost=%.1fdB)...",
                active_params["harmonic_exciter_mix"],
                active_params["high_boost_db"],
            )
            audio = AudioDSP.excite_high_frequencies(
                audio,
                sr,
                mix=active_params["harmonic_exciter_mix"],
                high_boost_db=active_params["high_boost_db"] * 0.6,
            )

        # --- Stage 2: Custom FFmpeg Mastering Filtergraph ---
        filter_chain = self._build_ffmpeg_filterchain(active_params, pre_report.channels)
        logger.info("Executing embedded FFmpeg mastering filtergraph: %s", ",".join(filter_chain))

        out_fmt = self.config.output_format
        if output_path.suffix:
            out_fmt = output_path.suffix.lstrip(".")

        self.bridge.run_filtergraph_and_encode(
            audio=audio,
            sample_rate=sr,
            output_path=output_path,
            filter_chain=filter_chain,
            output_format=out_fmt,
            output_bitrate=self.config.output_bitrate,
        )

        # --- Stage 3: Post-Mastering Verification Analysis ---
        remastered_audio, remastered_sr = self.bridge.load_audio_float32(output_path, sr)
        post_report = AudioAnalyzer.analyze(remastered_audio, remastered_sr)

        return {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "applied_parameters": active_params,
            "ffmpeg_filterchain": filter_chain,
            "pre_analysis": pre_report.to_dict(),
            "post_analysis": post_report.to_dict(),
        }

    def _resolve_parameters(self, report: AnalysisReport) -> Dict[str, float]:
        """Merges analyzer recommendations with user config overrides."""
        recs = dict(report.recommended_params)
        overrides = {
            "declick_sensitivity": self.config.declick_sensitivity,
            "declip_strength": self.config.declip_strength,
            "denoise_strength": self.config.denoise_strength,
            "detinny_strength": self.config.detinny_strength,
            "bass_boost_db": self.config.bass_boost_db,
            "subharmonic_mix": self.config.subharmonic_mix,
            "high_boost_db": self.config.high_boost_db,
            "harmonic_exciter_mix": self.config.harmonic_exciter_mix,
            "stereo_width": self.config.stereo_width,
        }

        for key, user_val in overrides.items():
            if user_val is not None:
                if self.config.auto_mode and self.config.preset != "auto":
                    # Blend preset baseline with intelligent signal analysis
                    recs[key] = round(0.55 * float(user_val) + 0.45 * recs[key], 3)
                else:
                    recs[key] = float(user_val)

        return recs

    def _build_ffmpeg_filterchain(self, params: Dict[str, float], channels: int) -> List[str]:
        """Constructs the filtergraph string for the custom stripped FFmpeg binary."""
        filters: List[str] = []

        # 1. Autoregressive De-clicking (Vinyl pops, tape ticks, digital clicks)
        if self.config.enable_declick and params["declick_sensitivity"] > 0.05:
            # threshold `t` ranges from 1.2 (aggressive) to 6.0 (gentle)
            thresh = round(max(1.2, 6.0 - 4.5 * params["declick_sensitivity"]), 2)
            filters.append(f"adeclick=w=55:o=75:a=2:t={thresh}:b=2")

        # 2. Secondary Autoregressive De-clipping
        if self.config.enable_declip and params["declip_strength"] > 0.15:
            thresh_clip = round(max(4.0, 12.0 - 7.0 * params["declip_strength"]), 2)
            filters.append(f"adeclip=w=55:o=75:a=8:t={thresh_clip}")

        # 3. Residual FFT Noise Reduction
        if self.config.enable_denoise and params["denoise_strength"] > 0.10:
            nr_db = round(min(28.0, max(3.0, params["denoise_strength"] * 22.0)), 1)
            filters.append(f"afftdn=nr={nr_db}:nf=-50:tn=1")

        # 4. Precision Mastering Shelving EQ (Bass & Treble polish)
        if self.config.enable_bass_restore and params["bass_boost_db"] > 0.2:
            b_gain = round(params["bass_boost_db"] * 0.45, 2)
            filters.append(f"bass=g={b_gain}:f=95:w=0.6")

        if self.config.enable_high_restore and params["high_boost_db"] > 0.2:
            t_gain = round(params["high_boost_db"] * 0.45, 2)
            filters.append(f"treble=g={t_gain}:f=10500:w=0.65")
            # Linear transient sharpening via crystalizer
            cryst_i = round(min(2.2, params["harmonic_exciter_mix"] * 3.5), 2)
            filters.append(f"crystalizer=i={cryst_i}:c=1")

        # 5. Stereo Field Recovery
        if self.config.enable_stereo_enhance and channels >= 2 and abs(params["stereo_width"] - 1.0) > 0.02:
            m = round(params["stereo_width"], 2)
            filters.append(f"extrastereo=m={m}:c=1")

        # 6. Gentle Mastering Glue Compression
        filters.append("acompressor=threshold=-18dB:ratio=2.0:attack=25:release=220:knee=4:makeup=1")

        # 7. EBU R128 Loudness Normalization & Brickwall True-Peak Limiter
        if self.config.enable_mastering_limiter:
            filters.append(
                f"loudnorm=I={self.config.target_lufs}:TP={self.config.target_true_peak_db}:LRA={self.config.target_lra}"
            )
            limit_lin = round(10.0 ** (self.config.target_true_peak_db / 20.0), 4)
            filters.append(f"alimiter=limit={limit_lin}:attack=5:release=50:level=0")

        return filters


def remaster_file(
    input_path: str | Path,
    output_path: str | Path,
    preset: str = "auto",
    **overrides: Any,
) -> Dict[str, Any]:
    """Convenience function for one-line Python usage."""
    cfg = RemasterConfig.from_preset(preset)
    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    engine = RemasterEngine(config=cfg)
    return engine.remaster(input_path, output_path)