from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
from scipy import signal


@dataclass
class AnalysisReport:
    """Comprehensive diagnostic metrics of an audio signal."""
    sample_rate: int
    channels: int
    duration_seconds: float
    dc_offset: float
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    noise_floor_dbfs: float
    estimated_snr_db: float
    noise_spectral_flatness: float
    clipped_sample_ratio: float
    hard_clip_events: int
    impulse_click_density: float      # Clicks/pops per second
    sub_bass_energy_db: float         # 20 - 60 Hz
    bass_energy_db: float             # 60 - 180 Hz
    low_mid_warmth_db: float          # 180 - 450 Hz
    high_mid_presence_db: float       # 1000 - 4500 Hz
    brilliance_energy_db: float       # 6000 - 16000 Hz
    tinny_ratio_db: float             # High-mid vs Low-mid/Bass imbalance
    bass_deficit_db: float            # Deficit compared to ideal pink slope
    high_freq_dropoff_db: float       # High-end loss compared to ideal slope
    spectral_rolloff_95_hz: float
    stereo_correlation: float
    stereo_width_ratio: float

    # Computed intelligent recommendations
    recommended_params: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioAnalyzer:
    """
    Analyzes raw waveform and spectral distributions to quantify:
    - White/static noise floor & hiss
    - Digital hard clipping & tape saturation
    - Vinyl clicks, pops, and impulse artifacts
    - "Tinny" mid-range harshness & lack of body
    - Bass roll-off & sub-harmonic deficiency
    - High-frequency tape/vinyl/codec attenuation
    """

    EPS = 1e-12

    @classmethod
    def analyze(cls, audio: np.ndarray, sample_rate: int) -> AnalysisReport:
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]

        num_samples, channels = audio.shape
        duration = num_samples / float(sample_rate)
        mono = np.mean(audio, axis=1)

        # 1. DC Offset, Peak, RMS, Crest Factor
        dc_offset = float(np.mean(mono))
        mono_centered = mono - dc_offset
        peak_lin = float(np.max(np.abs(audio)))
        rms_lin = float(np.sqrt(np.mean(mono_centered ** 2)))

        peak_dbfs = 20.0 * np.log10(max(peak_lin, cls.EPS))
        rms_dbfs = 20.0 * np.log10(max(rms_lin, cls.EPS))
        crest_factor_db = peak_dbfs - rms_dbfs

        # 2. Digital Clipping & Flat-Top Detection
        clip_thresh = max(0.985, peak_lin * 0.992) if peak_lin > 0.85 else 0.995
        abs_audio = np.abs(audio)
        clipped_mask = abs_audio >= clip_thresh
        clipped_ratio = float(np.mean(clipped_mask))

        # Count consecutive clipped runs >= 2 samples
        diff_flat = (np.abs(np.diff(audio, axis=0)) < 1e-4) & (abs_audio[:-1] >= 0.90)
        hard_clip_events = int(np.sum(diff_flat))

        # 3. Vinyl Click / Impulse Artifact Detection (2nd derivative outlier analysis)
        d2 = np.diff(mono_centered, n=2)
        # High-pass local median variance
        med_abs_d2 = float(np.median(np.abs(d2))) + cls.EPS
        click_spikes = np.abs(d2) > (18.0 * med_abs_d2)
        impulse_click_density = float(np.sum(click_spikes) / max(duration, 0.1))

        # 4. Noise Floor & White/Static Hiss Estimation (Frame-based 5th percentile)
        frame_len = max(int(0.04 * sample_rate), 256)
        num_frames = num_samples // frame_len
        if num_frames > 4:
            frames = mono_centered[: num_frames * frame_len].reshape(num_frames, frame_len)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + cls.EPS)
            # Ignore complete digital silence frames (< -95 dBFS) when estimating analog noise floor
            active_rms = frame_rms[frame_rms > 1.8e-5]
            if active_rms.size > 0:
                noise_rms = float(np.percentile(active_rms, 8))
            else:
                noise_rms = float(np.percentile(frame_rms, 10))

            noise_floor_dbfs = 20.0 * np.log10(max(noise_rms, cls.EPS))

            # Spectral flatness of quietest frames (white noise has flatness -> 1.0)
            quiet_idx = np.argsort(frame_rms)[: max(2, num_frames // 10)]
            quiet_frames = frames[quiet_idx]
            fft_mag = np.abs(np.fft.rfft(quiet_frames, axis=1)) ** 2 + cls.EPS
            geom_mean = np.exp(np.mean(np.log(fft_mag), axis=1))
            arith_mean = np.mean(fft_mag, axis=1)
            noise_flatness = float(np.mean(geom_mean / arith_mean))
        else:
            noise_floor_dbfs = -75.0
            noise_flatness = 0.2

        estimated_snr_db = max(0.0, rms_dbfs - noise_floor_dbfs)

        # 5. Multi-Band Spectral Balance via Welch's Power Spectral Density
        nperseg = min(4096, max(512, num_samples // 4))
        freqs, psd = signal.welch(mono_centered, fs=sample_rate, nperseg=nperseg)
        psd = psd + cls.EPS

        def band_db(f_low: float, f_high: float) -> float:
            f_high_clamped = min(f_high, 0.49 * sample_rate)
            if f_low >= f_high_clamped:
                return -90.0
            mask = (freqs >= f_low) & (freqs <= f_high_clamped)
            if not np.any(mask):
                return -90.0
            power = float(np.mean(psd[mask]))
            return 10.0 * np.log10(max(power, cls.EPS))

        sub_bass_db = band_db(20.0, 60.0)
        bass_db = band_db(60.0, 180.0)
        low_mid_db = band_db(180.0, 450.0)
        mid_ref_db = band_db(450.0, 1000.0)
        high_mid_db = band_db(1000.0, 4500.0)
        brilliance_db = band_db(6000.0, 16000.0)

        # Tinny Ratio: Excessive 1kHz-4.5kHz energy relative to warmth/bass foundation (60Hz-450Hz)
        foundation_db = 0.5 * bass_db + 0.5 * low_mid_db
        # In balanced masters, 1-4.5kHz PSD mean is ~10-14dB lower per Hz than 60-450Hz due to 1/f pink spectrum
        expected_high_mid_db = foundation_db - 11.5
        tinny_ratio_db = float(max(0.0, high_mid_db - expected_high_mid_db))

        # Bass Deficit: Compare 40-160Hz against 450-1000Hz reference (pink spectrum expects bass +6dB/Hz higher)
        expected_bass_db = mid_ref_db + 5.5
        bass_deficit_db = float(max(0.0, expected_bass_db - bass_db))

        # High-Frequency Drop-off: Compare 6kHz-16kHz against 450-1000Hz (pink spectrum expects ~ -13dB/Hz)
        expected_brilliance_db = mid_ref_db - 13.5
        high_freq_dropoff_db = float(max(0.0, expected_brilliance_db - brilliance_db))

        # 95% Spectral Rolloff Frequency
        cum_power = np.cumsum(psd)
        rolloff_idx = np.searchsorted(cum_power, 0.95 * cum_power[-1])
        spectral_rolloff_95_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # 6. Stereo Field Analysis
        if channels >= 2:
            left, right = audio[:, 0], audio[:, 1]
            mid_sig = 0.5 * (left + right)
            side_sig = 0.5 * (left - right)
            mid_rms = float(np.sqrt(np.mean(mid_sig ** 2)) + cls.EPS)
            side_rms = float(np.sqrt(np.mean(side_sig ** 2)))
            stereo_width_ratio = float(side_rms / mid_rms)
            denom = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)) + cls.EPS)
            stereo_correlation = float(np.sum(left * right) / denom)
        else:
            stereo_width_ratio = 0.0
            stereo_correlation = 1.0

        # 7. Synthesize Intelligent Restoration Parameters
        recs = cls._compute_recommendations(
            estimated_snr_db=estimated_snr_db,
            noise_flatness=noise_flatness,
            clipped_ratio=clipped_ratio,
            hard_clip_events=hard_clip_events,
            impulse_click_density=impulse_click_density,
            tinny_ratio_db=tinny_ratio_db,
            bass_deficit_db=bass_deficit_db,
            sub_bass_db=sub_bass_db,
            bass_db=bass_db,
            high_freq_dropoff_db=high_freq_dropoff_db,
            spectral_rolloff_95_hz=spectral_rolloff_95_hz,
            stereo_width_ratio=stereo_width_ratio,
            channels=channels,
        )

        return AnalysisReport(
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=round(duration, 3),
            dc_offset=round(dc_offset, 6),
            peak_dbfs=round(peak_dbfs, 2),
            rms_dbfs=round(rms_dbfs, 2),
            crest_factor_db=round(crest_factor_db, 2),
            noise_floor_dbfs=round(noise_floor_dbfs, 2),
            estimated_snr_db=round(estimated_snr_db, 2),
            noise_spectral_flatness=round(noise_flatness, 3),
            clipped_sample_ratio=round(clipped_ratio, 6),
            hard_clip_events=hard_clip_events,
            impulse_click_density=round(impulse_click_density, 2),
            sub_bass_energy_db=round(sub_bass_db, 2),
            bass_energy_db=round(bass_db, 2),
            low_mid_warmth_db=round(low_mid_db, 2),
            high_mid_presence_db=round(high_mid_db, 2),
            brilliance_energy_db=round(brilliance_db, 2),
            tinny_ratio_db=round(tinny_ratio_db, 2),
            bass_deficit_db=round(bass_deficit_db, 2),
            high_freq_dropoff_db=round(high_freq_dropoff_db, 2),
            spectral_rolloff_95_hz=round(spectral_rolloff_95_hz, 1),
            stereo_correlation=round(stereo_correlation, 3),
            stereo_width_ratio=round(stereo_width_ratio, 3),
            recommended_params=recs,
        )

    @staticmethod
    def _compute_recommendations(
        estimated_snr_db: float,
        noise_flatness: float,
        clipped_ratio: float,
        hard_clip_events: int,
        impulse_click_density: float,
        tinny_ratio_db: float,
        bass_deficit_db: float,
        sub_bass_db: float,
        bass_db: float,
        high_freq_dropoff_db: float,
        spectral_rolloff_95_hz: float,
        stereo_width_ratio: float,
        channels: int,
    ) -> Dict[str, float]:
        # De-click sensitivity (0.0 to 1.0)
        declick = float(np.clip(impulse_click_density / 35.0, 0.15, 0.95))

        # De-clip strength (0.0 to 1.0)
        if hard_clip_events > 25 or clipped_ratio > 0.0005:
            declip = float(np.clip(0.45 + (clipped_ratio * 250.0) + (hard_clip_events / 1000.0), 0.45, 1.0))
        else:
            declip = 0.15

        # De-noise strength (0.0 to 1.0) - higher when SNR is low or noise is flat (white/tape hiss)
        snr_factor = max(0.0, (48.0 - estimated_snr_db) / 35.0)
        flatness_boost = 1.25 if noise_flatness > 0.35 else 1.0
        denoise = float(np.clip(snr_factor * flatness_boost, 0.10, 0.90))

        # De-tinny strength (0.0 to 1.0)
        detinny = float(np.clip(tinny_ratio_db / 12.0, 0.0, 0.95))

        # Bass restoration (dB boost & sub-harmonic synthesis mix)
        bass_boost_db = float(np.clip(bass_deficit_db * 0.65, 0.0, 6.5))
        sub_gap = max(0.0, (bass_db - sub_bass_db) - 4.0)
        subharmonic_mix = float(np.clip((bass_deficit_db * 0.03) + (sub_gap * 0.02), 0.05, 0.40))

        # High-frequency restoration (dB boost & Chebyshev harmonic exciter mix)
        rolloff_penalty = max(0.0, (9000.0 - spectral_rolloff_95_hz) / 2000.0)
        high_boost_db = float(np.clip((high_freq_dropoff_db * 0.55) + rolloff_penalty, 0.0, 7.0))
        harmonic_exciter_mix = float(np.clip((high_freq_dropoff_db * 0.035) + (rolloff_penalty * 0.06), 0.08, 0.50))

        # Stereo enhancement multiplier
        if channels >= 2 and stereo_width_ratio < 0.25:
            stereo_width = float(np.clip(1.0 + (0.30 - stereo_width_ratio) * 1.2, 1.0, 1.35))
        else:
            stereo_width = 1.05

        return {
            "declick_sensitivity": round(declick, 3),
            "declip_strength": round(declip, 3),
            "denoise_strength": round(denoise, 3),
            "detinny_strength": round(detinny, 3),
            "bass_boost_db": round(bass_boost_db, 2),
            "subharmonic_mix": round(subharmonic_mix, 3),
            "high_boost_db": round(high_boost_db, 2),
            "harmonic_exciter_mix": round(harmonic_exciter_mix, 3),
            "stereo_width": round(stereo_width, 3),
        }