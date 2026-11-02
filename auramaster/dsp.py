import numpy as np
from scipy import signal, interpolate, ndimage


class AudioDSP:
    """
    Scientific signal processing routines for audio restoration and harmonic synthesis.
    All methods accept and return float64 NumPy arrays of shape (samples, channels).
    """

    EPS = 1e-12

    @classmethod
    def remove_dc_and_rumble(cls, audio: np.ndarray, sample_rate: int, cutoff_hz: float = 16.0) -> np.ndarray:
        """Applies a zero-phase 4th-order Butterworth high-pass filter to remove DC offset and vinyl warp rumble."""
        centered = audio - np.mean(audio, axis=0, keepdims=True)
        nyq = 0.5 * sample_rate
        norm_cutoff = min(max(cutoff_hz / nyq, 0.0005), 0.45)
        sos = signal.butter(4, norm_cutoff, btype="highpass", output="sos")
        return signal.sosfiltfilt(sos, centered, axis=0)

    @classmethod
    def reconstruct_clipped_and_dropouts(
        cls, audio: np.ndarray, sample_rate: int, strength: float = 0.5
    ) -> np.ndarray:
        """
        Detects clipped plateaus and sudden tape/vinyl impulse dropouts, scales down to create headroom,
        and reconstructs missing waveform peaks using local Cubic Spline interpolation anchored to
        surrounding unclipped derivatives.
        """
        if strength <= 0.05:
            return audio

        out = audio.copy()
        num_samples, channels = out.shape
        peak = float(np.max(np.abs(out)))
        if peak < 0.5:
            return out

        # Threshold adapts to strength
        clip_thresh = peak * (0.995 - 0.045 * strength)

        # Create headroom so reconstructed peaks can exceed the old flat-top ceiling
        headroom_scale = 0.78
        out *= headroom_scale
        scaled_thresh = clip_thresh * headroom_scale

        max_span = max(3, int(0.008 * sample_rate))  # Reconstruct spans up to 8ms

        for ch in range(channels):
            col = out[:, ch]
            clipped = np.abs(col) >= scaled_thresh

            # Also flag flat-topped saturation runs
            d1 = np.abs(np.diff(col, prepend=col[0]))
            flat_top = (d1 < 1e-4) & (np.abs(col) >= scaled_thresh * 0.92)
            mask = clipped | flat_top

            if not np.any(mask):
                continue

            # Dilate mask by 1 sample on each side to avoid clipped boundary slopes
            mask = ndimage.binary_dilation(mask, iterations=1)

            # Find contiguous clipped intervals
            padded = np.concatenate(([False], mask, [False]))
            starts = np.flatnonzero(padded[1:] & ~padded[:-1])
            ends = np.flatnonzero(~padded[1:] & padded[:-1])

            anchor_radius = 6
            for s, e in zip(starts, ends):
                span = e - s
                if span < 1 or span > max_span:
                    continue

                left_start = max(0, s - anchor_radius)
                right_end = min(num_samples, e + anchor_radius)

                anchor_idx = np.concatenate((
                    np.arange(left_start, s),
                    np.arange(e, right_end),
                ))
                if anchor_idx.size < 4:
                    continue

                anchor_vals = col[anchor_idx]
                try:
                    cs = interpolate.CubicSpline(anchor_idx, anchor_vals, bc_type="natural")
                    target_idx = np.arange(s, e)
                    reconstructed = cs(target_idx)
                    # Blend according to strength and clamp unreasonable spline overshoots
                    max_allowed = headroom_scale * (1.0 + 0.35 * strength)
                    reconstructed = np.clip(reconstructed, -max_allowed, max_allowed)
                    col[target_idx] = (1.0 - strength) * col[target_idx] + strength * reconstructed
                except ValueError:
                    continue

            out[:, ch] = col

        return out

    @classmethod
    def spectral_denoise(
        cls, audio: np.ndarray, sample_rate: int, strength: float = 0.5
    ) -> np.ndarray:
        """
        Removes stationary white/static noise and tape hiss via Short-Time Fourier Transform (STFT)
        using noise-floor profile estimation and time-frequency smoothed Wiener soft masking.
        """
        if strength <= 0.05:
            return audio

        num_samples, channels = audio.shape
        nperseg = min(2048, max(512, num_samples // 8))
        noverlap = (nperseg * 3) // 4
        out = np.zeros_like(audio)

        over_subtraction = 1.0 + 2.2 * strength
        spectral_floor = max(0.02, 0.25 * (1.0 - strength))

        for ch in range(channels):
            f, t, Zxx = signal.stft(
                audio[:, ch],
                fs=sample_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
            )
            mag = np.abs(Zxx)
            power = mag ** 2 + cls.EPS

            # Estimate noise power spectrum from the quietest 10% of frames
            frame_energy = np.mean(power, axis=0)
            num_quiet = max(2, int(0.10 * power.shape[1]))
            quiet_indices = np.argsort(frame_energy)[:num_quiet]
            noise_psd = np.median(power[:, quiet_indices], axis=1, keepdims=True)

            # A-posteriori SNR and Wiener gain
            post_snr = np.maximum(power / (noise_psd * over_subtraction + cls.EPS) - 1.0, 0.0)
            gain = post_snr / (post_snr + 1.0)
            gain = np.maximum(gain, spectral_floor)

            # Smooth gain in time and frequency to eliminate musical noise artifacts
            gain_smoothed = ndimage.gaussian_filter(gain, sigma=(1.2, 1.5))

            Zxx_clean = Zxx * gain_smoothed
            _, x_rec = signal.istft(
                Zxx_clean,
                fs=sample_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
            )
            out[:, ch] = x_rec[:num_samples]

        return out

    @classmethod
    def suppress_tinny_resonances(
        cls, audio: np.ndarray, sample_rate: int, strength: float = 0.5
    ) -> np.ndarray:
        """
        Eliminates 'tinny', harsh, telephone-like sound by:
        1. Dynamically attenuating narrow resonant spikes in the 1.1 kHz - 4.2 kHz harshness band
           via STFT spectral envelope smoothing.
        2. Restoring low-mid body warmth (180 Hz - 380 Hz).
        """
        if strength <= 0.05:
            return audio

        num_samples, channels = audio.shape
        nperseg = min(2048, max(512, num_samples // 8))
        noverlap = (nperseg * 3) // 4
        out = np.zeros_like(audio)

        for ch in range(channels):
            f, t, Zxx = signal.stft(audio[:, ch], fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            mag = np.abs(Zxx) + cls.EPS

            # Locate harshness band (1100 Hz to 4200 Hz)
            harsh_mask = (f >= 1100.0) & (f <= min(4200.0, 0.45 * sample_rate))
            if np.any(harsh_mask):
                # Compute broad spectral envelope across frequency axis
                broad_env = ndimage.gaussian_filter1d(mag, sigma=8.0, axis=0) + cls.EPS
                spike_ratio = mag[harsh_mask, :] / broad_env[harsh_mask, :]

                # Compress bins that protrude above the local spectral envelope
                threshold = 1.25
                excess = np.maximum(spike_ratio - threshold, 0.0)
                attenuation = 1.0 / (1.0 + strength * 1.4 * excess)
                # Also apply a gentle broad dip across the 1.8kHz - 3.4kHz tinny core
                broad_dip = 1.0 - (0.32 * strength)
                Zxx[harsh_mask, :] *= (attenuation * broad_dip)

            _, x_rec = signal.istft(Zxx, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            out[:, ch] = x_rec[:num_samples]

        # Add low-mid warmth bell filter centered at 240 Hz
        warmth_gain_db = 3.5 * strength
        out = cls._peaking_eq(out, sample_rate, center_hz=240.0, q=0.85, gain_db=warmth_gain_db)
        return out

    @classmethod
    def synthesize_subharmonics(
        cls, audio: np.ndarray, sample_rate: int, mix: float = 0.2, bass_boost_db: float = 2.0
    ) -> np.ndarray:
        """
        Restores lost bass frequencies using a dual approach:
        1. Sub-harmonic synthesis: Isolates 55Hz - 130Hz fundamentals, uses a zero-crossing flip-flop
           frequency divider modulated by the original bass envelope to synthesize true 27Hz - 65Hz
           sub-octave harmonics, low-pass filters at 85Hz, and blends phase-coherently.
        2. Warm low-shelf EQ boost.
        """
        if mix <= 0.01 and bass_boost_db <= 0.1:
            return audio

        out = audio.copy()
        nyq = 0.5 * sample_rate

        if mix > 0.01 and nyq > 200.0:
            # 1. Isolate source bass band (55 Hz - 130 Hz)
            sos_bp = signal.butter(
                2, [55.0 / nyq, min(130.0 / nyq, 0.45)], btype="bandpass", output="sos"
            )
            bass_band = signal.sosfiltfilt(sos_bp, out, axis=0)

            # 2. Envelope detection via Hilbert transform magnitude or rectified lowpass
            env = np.abs(bass_band)
            sos_env = signal.butter(2, min(35.0 / nyq, 0.4), btype="lowpass", output="sos")
            env_smooth = signal.sosfiltfilt(sos_env, env, axis=0)

            # 3. Sub-octave flip-flop divider triggered on positive zero crossings
            sub_wave = np.zeros_like(bass_band)
            for ch in range(out.shape[1]):
                sig = bass_band[:, ch]
                pos_crossings = (sig[1:] >= 0.0) & (sig[:-1] < 0.0)
                toggle = np.cumsum(np.concatenate(([0], pos_crossings.astype(np.int32))))
                square_sub = np.where(toggle % 2 == 0, 1.0, -1.0)
                sub_wave[:, ch] = square_sub * env_smooth[:, ch]

            # 4. Smooth square sub-harmonics into pure sinusoidal sub-bass (24 Hz - 85 Hz)
            sos_sub = signal.butter(
                3, [24.0 / nyq, min(85.0 / nyq, 0.45)], btype="bandpass", output="sos"
            )
            sub_clean = signal.sosfiltfilt(sos_sub, sub_wave, axis=0)

            out = out + (mix * 1.4) * sub_clean

        if bass_boost_db > 0.1:
            out = cls._peaking_eq(out, sample_rate, center_hz=85.0, q=0.707, gain_db=bass_boost_db)

        return out

    @classmethod
    def excite_high_frequencies(
        cls, audio: np.ndarray, sample_rate: int, mix: float = 0.25, high_boost_db: float = 3.0
    ) -> np.ndarray:
        """
        Reconstructs high frequencies lost to tape wear, vinyl groove degradation, or old lossy encoding:
        1. Extracts surviving upper-mid/high band (3.2 kHz - 7.5 kHz).
        2. Applies 2nd, 3rd, and 4th order Chebyshev polynomial waveshaping (T2, T3, T4)
           to synthesize natural even and odd upper harmonics (6.4 kHz - 18 kHz).
        3. High-pass filters the generated harmonics and blends them into the dry signal.
        """
        if mix <= 0.01 and high_boost_db <= 0.1:
            return audio

        out = audio.copy()
        nyq = 0.5 * sample_rate

        if mix > 0.01 and nyq > 5000.0:
            low_f = min(3200.0 / nyq, 0.40)
            high_f = min(8000.0 / nyq, 0.48)
            if low_f < high_f:
                sos_seed = signal.butter(2, [low_f, high_f], btype="bandpass", output="sos")
                seed = signal.sosfiltfilt(sos_seed, out, axis=0)

                seed_peak = float(np.max(np.abs(seed))) + cls.EPS
                x = np.tanh((seed / seed_peak) * 1.6)

                # Chebyshev polynomials of the first kind (remove DC offset from even orders)
                t2 = (2.0 * (x ** 2) - 1.0) - np.mean(2.0 * (x ** 2) - 1.0, axis=0, keepdims=True)
                t3 = 4.0 * (x ** 3) - 3.0 * x
                t4 = (8.0 * (x ** 4) - 8.0 * (x ** 2) + 1.0) - np.mean(
                    8.0 * (x ** 4) - 8.0 * (x ** 2) + 1.0, axis=0, keepdims=True
                )

                harmonics = (0.50 * t2 + 0.35 * t3 + 0.15 * t4) * seed_peak

                # High-pass filter synthesized harmonics above 5.8 kHz
                hp_f = min(5800.0 / nyq, 0.46)
                sos_hp = signal.butter(3, hp_f, btype="highpass", output="sos")
                harmonics_hp = signal.sosfiltfilt(sos_hp, harmonics, axis=0)

                out = out + (mix * 0.85) * harmonics_hp

        if high_boost_db > 0.1 and nyq > 7500.0:
            center = min(9500.0, 0.42 * sample_rate)
            out = cls._peaking_eq(out, sample_rate, center_hz=center, q=0.65, gain_db=high_boost_db)

        return out

    @staticmethod
    def _peaking_eq(
        audio: np.ndarray, sample_rate: int, center_hz: float, q: float, gain_db: float
    ) -> np.ndarray:
        """RBJ Audio-EQ-Cookbook biquad peaking equalizer applied zero-phase via sosfiltfilt."""
        if abs(gain_db) < 1e-3 or center_hz >= 0.48 * sample_rate:
            return audio

        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * center_hz / sample_rate
        alpha = np.sin(w0) / (2.0 * q)
        cos_w0 = np.cos(w0)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)
        return signal.sosfiltfilt(sos, audio, axis=0)