from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class RemasterConfig:
    """
    Configuration parameters for the AuraMaster restoration and mastering pipeline.
    When `auto_mode=True`, the engine dynamically computes and overrides restoration
    intensities based on the input audio's spectral and temporal analysis.
    """
    auto_mode: bool = True
    preset: str = "auto"

    # Sample rate & bit-depth settings
    target_sample_rate: Optional[int] = None  # None preserves input or upsamples if < 44100
    output_format: str = "flac"               # wav, flac, mp3, ogg, aac, opus
    output_bitrate: str = "320k"              # Used for lossy codecs

    # Stage toggles
    enable_declick: bool = True
    enable_declip: bool = True
    enable_denoise: bool = True
    enable_detinny: bool = True
    enable_bass_restore: bool = True
    enable_high_restore: bool = True
    enable_stereo_enhance: bool = True
    enable_mastering_limiter: bool = True

    # Manual / Override intensities (0.0 to 1.0)
    # In auto_mode, these act as scaling multipliers or get populated by the analyzer
    declick_sensitivity: Optional[float] = None
    declip_strength: Optional[float] = None
    denoise_strength: Optional[float] = None
    detinny_strength: Optional[float] = None
    bass_boost_db: Optional[float] = None
    subharmonic_mix: Optional[float] = None
    high_boost_db: Optional[float] = None
    harmonic_exciter_mix: Optional[float] = None
    stereo_width: Optional[float] = None

    # Mastering targets (EBU R128)
    target_lufs: float = -14.0
    target_true_peak_db: float = -1.0
    target_lra: float = 11.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_preset(cls, preset_name: str) -> "RemasterConfig":
        preset_name = preset_name.lower().strip()
        if preset_name == "auto":
            return cls(auto_mode=True, preset="auto")

        elif preset_name == "vinyl":
            return cls(
                auto_mode=True,
                preset="vinyl",
                declick_sensitivity=0.85,
                declip_strength=0.45,
                denoise_strength=0.55,
                bass_boost_db=2.5,
                subharmonic_mix=0.20,
                high_boost_db=3.5,
                harmonic_exciter_mix=0.30,
                target_lufs=-14.0,
            )

        elif preset_name == "tape":
            return cls(
                auto_mode=True,
                preset="tape",
                declick_sensitivity=0.30,
                declip_strength=0.50,
                denoise_strength=0.75,
                detinny_strength=0.40,
                bass_boost_db=2.0,
                subharmonic_mix=0.18,
                high_boost_db=5.0,
                harmonic_exciter_mix=0.45,
                stereo_width=1.15,
                target_lufs=-14.0,
            )

        elif preset_name == "digital_clip":
            return cls(
                auto_mode=True,
                preset="digital_clip",
                declick_sensitivity=0.50,
                declip_strength=0.95,
                denoise_strength=0.20,
                detinny_strength=0.50,
                target_lufs=-13.5,
            )

        elif preset_name == "tinny":
            return cls(
                auto_mode=True,
                preset="tinny",
                detinny_strength=0.85,
                bass_boost_db=4.5,
                subharmonic_mix=0.35,
                high_boost_db=2.0,
                harmonic_exciter_mix=0.20,
                target_lufs=-14.0,
            )

        raise ValueError(
            f"Unknown preset '{preset_name}'. Choose from: auto, vinyl, tape, digital_clip, tinny."
        )