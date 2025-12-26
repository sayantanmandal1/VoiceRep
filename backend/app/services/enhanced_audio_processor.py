"""
Enhanced Audio Processor for Voice Synthesis Quality Improvement.

This module provides advanced audio processing capabilities to fix common
synthesis issues like dual sounds, overlapping artifacts, and clarity problems.
"""

import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import scipy.ndimage
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.fft import fft, ifft
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class EnhancedAudioProcessor:
    """
    Enhanced audio processor for fixing synthesis artifacts and improving quality.
    
    Addresses common issues:
    - Dual sounds and overlapping artifacts
    - Echo and reverb artifacts
    - Poor clarity and muffled audio
    - Unnatural prosody and timing
    - Dynamic range issues
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        
        # Processing parameters
        self.artifact_detection_threshold = 0.25
        self.clarity_enhancement_strength = 0.3
        self.echo_removal_strength = 0.7
        
        logger.info(f"Enhanced Audio Processor initialized (SR: {sample_rate}Hz)")
    
    def process_synthesized_audio(
        self, 
        audio_path: str, 
        reference_audio_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Main processing pipeline for synthesized audio enhancement.
        
        Args:
            audio_path: Path to synthesized audio file
            reference_audio_path: Optional reference audio for matching
            output_path: Optional output path (defaults to enhanced version)
            
        Returns:
            Tuple of (enhanced_audio_path, processing_metrics)
        """
        try:
            logger.info(f"Starting enhanced audio processing for: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            original_audio = audio.copy()
            
            # Load reference audio if provided
            reference_audio = None
            if reference_audio_path and os.path.exists(reference_audio_path):
                reference_audio, _ = librosa.load(reference_audio_path, sr=self.sample_rate)
            
            processing_metrics = {
                "original_duration": len(audio) / sr,
                "original_rms": float(np.sqrt(np.mean(audio**2))),
                "processing_steps": []
            }
            
            # Step 1: Remove dual sound artifacts
            logger.info("Removing dual sound and overlapping artifacts...")
            audio, dual_metrics = self._remove_dual_sound_artifacts(audio)
            processing_metrics["processing_steps"].append({
                "step": "dual_sound_removal",
                "metrics": dual_metrics
            })
            
            # Step 2: Echo and reverb removal
            logger.info("Removing echo and reverb artifacts...")
            audio, echo_metrics = self._remove_echo_reverb_artifacts(audio)
            processing_metrics["processing_steps"].append({
                "step": "echo_removal",
                "metrics": echo_metrics
            })
            
            # Step 3: Clarity enhancement
            logger.info("Enhancing audio clarity...")
            audio, clarity_metrics = self._enhance_audio_clarity(audio)
            processing_metrics["processing_steps"].append({
                "step": "clarity_enhancement",
                "metrics": clarity_metrics
            })
            
            # Step 4: Spectral cleaning
            logger.info("Cleaning spectral artifacts...")
            audio, spectral_metrics = self._clean_spectral_artifacts(audio)
            processing_metrics["processing_steps"].append({
                "step": "spectral_cleaning",
                "metrics": spectral_metrics
            })
            
            # Step 5: Reference matching (if available)
            if reference_audio is not None:
                logger.info("Matching to reference audio characteristics...")
                audio, matching_metrics = self._match_reference_characteristics(
                    audio, reference_audio
                )
                processing_metrics["processing_steps"].append({
                    "step": "reference_matching",
                    "metrics": matching_metrics
                })
            
            # Step 6: Final quality enhancement
            logger.info("Applying final quality enhancements...")
            audio, final_metrics = self._apply_final_enhancements(audio)
            processing_metrics["processing_steps"].append({
                "step": "final_enhancement",
                "metrics": final_metrics
            })
            
            # Calculate final metrics
            processing_metrics.update({
                "final_duration": len(audio) / sr,
                "final_rms": float(np.sqrt(np.mean(audio**2))),
                "quality_improvement": self._calculate_quality_improvement(
                    original_audio, audio
                ),
                "processing_success": True
            })
            
            # Save enhanced audio
            if output_path is None:
                base_path = Path(audio_path)
                output_path = str(base_path.parent / f"{base_path.stem}_enhanced{base_path.suffix}")
            
            # Save with high quality
            sf.write(output_path, audio, sr, format='WAV', subtype='PCM_24')
            
            logger.info(f"Enhanced audio saved to: {output_path}")
            logger.info(f"Quality improvement: {processing_metrics['quality_improvement']:.2f}")
            
            return output_path, processing_metrics
            
        except Exception as e:
            logger.error(f"Enhanced audio processing failed: {e}")
            processing_metrics = {
                "processing_success": False,
                "error": str(e)
            }
            return audio_path, processing_metrics
    
    def _remove_dual_sound_artifacts(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove dual sound and overlapping artifacts."""
        try:
            original_audio = audio.copy()
            metrics = {"artifacts_detected": [], "artifacts_removed": []}
            
            # Ensure audio is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                metrics["artifacts_detected"].append("non_finite_values")
                metrics["artifacts_removed"].append("non_finite_values")
            
            # 1. Detect and remove echo-based dual sounds
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for secondary peaks (potential echoes)
            min_delay = int(0.02 * self.sample_rate)  # 20ms minimum
            max_delay = int(0.5 * self.sample_rate)   # 500ms maximum
            
            if max_delay < len(autocorr) and len(autocorr) > min_delay:
                search_region = autocorr[min_delay:max_delay]
                if len(search_region) > 0:
                    primary_peak = np.max(autocorr[:min_delay]) if min_delay > 0 else np.max(autocorr[:100])
                    
                    # Find significant secondary peaks
                    if primary_peak > 0:
                        peaks, _ = scipy.signal.find_peaks(
                            search_region, 
                            height=0.2 * primary_peak,
                            distance=int(0.01 * self.sample_rate)  # 10ms minimum separation
                        )
                        
                        if len(peaks) > 0:
                            # Remove the strongest echo
                            strongest_peak_idx = peaks[np.argmax(search_region[peaks])]
                            echo_delay = strongest_peak_idx + min_delay
                            echo_strength = search_region[strongest_peak_idx] / primary_peak
                            
                            if echo_strength > 0.15 and echo_delay < len(audio):  # Significant echo
                                # Create and subtract echo
                                echo_audio = np.zeros_like(audio)
                                echo_audio[echo_delay:] = audio[:-echo_delay] * echo_strength * 0.8
                                audio = audio - echo_audio
                                
                                metrics["artifacts_detected"].append("echo_artifact")
                                metrics["artifacts_removed"].append("echo_artifact")
                                metrics["echo_delay_ms"] = float(echo_delay / self.sample_rate * 1000)
                                metrics["echo_strength"] = float(echo_strength)
            
            # 2. Remove frequency-domain overlapping
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Ensure magnitude is finite
            if not np.all(np.isfinite(magnitude)):
                magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Detect problematic frequency bins
            freq_energy = np.mean(magnitude, axis=1)
            freq_variance = np.var(magnitude, axis=1)
            
            # Only process if we have valid statistics
            if len(freq_energy) > 0 and np.any(freq_energy > 0):
                # Identify bins with unusual energy patterns
                energy_threshold = np.percentile(freq_energy, 85)
                variance_threshold = np.percentile(freq_variance, 80)
                
                problematic_bins = np.where(
                    (freq_energy > energy_threshold) & 
                    (freq_variance > variance_threshold)
                )[0]
                
                if len(problematic_bins) > 0:
                    # Apply gentle suppression
                    suppression_factor = 0.6
                    magnitude[problematic_bins, :] *= suppression_factor
                    
                    # Reconstruct audio
                    enhanced_stft = magnitude * np.exp(1j * phase)
                    audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
                    
                    metrics["artifacts_detected"].append("frequency_overlap")
                    metrics["artifacts_removed"].append("frequency_overlap")
                    metrics["suppressed_bins"] = len(problematic_bins)
            
            # 3. Temporal smoothing for abrupt transitions
            if len(audio) > 1024:  # Only if audio is long enough
                # Detect sudden amplitude changes
                rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
                
                if len(rms) > 1:
                    rms_diff = np.abs(np.diff(rms))
                    if len(rms_diff) > 0:
                        sudden_changes = np.where(rms_diff > np.percentile(rms_diff, 95))[0]
                        
                        if len(sudden_changes) > 0:
                            # Apply smoothing around sudden changes
                            window_length = min(5, len(rms))
                            if window_length >= 3:  # Minimum for savgol_filter
                                rms_smooth = savgol_filter(rms, window_length=window_length, polyorder=2)
                                smoothing_ratio = np.divide(rms_smooth, rms, out=np.ones_like(rms), where=rms!=0)
                                
                                # Interpolate to audio length
                                if len(smoothing_ratio) > 1:
                                    # Create proper x-coordinates for interpolation
                                    x_coords = np.arange(0, len(audio), 256)[:len(smoothing_ratio)]
                                    if len(x_coords) == len(smoothing_ratio):
                                        smoothing_interp = np.interp(
                                            np.arange(len(audio)),
                                            x_coords,
                                            smoothing_ratio
                                        )
                                        
                                        audio = audio * smoothing_interp
                                    
                                    metrics["artifacts_detected"].append("sudden_transitions")
                                    metrics["artifacts_removed"].append("sudden_transitions")
                                    metrics["smoothed_transitions"] = len(sudden_changes)
            
            # Ensure output is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate improvement
            improvement = self._calculate_artifact_reduction(original_audio, audio)
            metrics["artifact_reduction_score"] = float(improvement)
            
            return audio.astype(np.float32), metrics
            
        except Exception as e:
            logger.warning(f"Dual sound artifact removal failed: {e}")
            return audio, {"error": str(e)}
    
    def _remove_echo_reverb_artifacts(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove echo and reverb artifacts."""
        try:
            original_audio = audio.copy()
            metrics = {}
            
            # Ensure audio is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                metrics["non_finite_fixed"] = True
            
            # Skip processing if audio is too short or all zeros
            if len(audio) < 1024 or np.max(np.abs(audio)) < 1e-10:
                metrics["skipped_short_or_silent"] = True
                return audio.astype(np.float32), metrics
            
            # 1. Cepstral-based echo detection and removal
            try:
                # Window the audio to avoid edge effects
                window = np.hanning(len(audio))
                windowed_audio = audio * window
                
                # Compute cepstrum with safety checks
                spectrum = np.fft.fft(windowed_audio)
                log_spectrum = np.log(np.abs(spectrum) + 1e-10)
                
                # Check for finite values
                if not np.all(np.isfinite(log_spectrum)):
                    log_spectrum = np.nan_to_num(log_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
                
                cepstrum = np.real(np.fft.ifft(log_spectrum))
                
                # Look for echo peaks in cepstrum
                min_quefrency = int(0.02 * self.sample_rate)  # 20ms
                max_quefrency = min(int(0.3 * self.sample_rate), len(cepstrum) // 2)   # 300ms or half length
                
                if max_quefrency > min_quefrency and max_quefrency < len(cepstrum):
                    cepstrum_region = cepstrum[min_quefrency:max_quefrency]
                    if len(cepstrum_region) > 0:
                        echo_peak_idx = np.argmax(np.abs(cepstrum_region))
                        echo_quefrency = echo_peak_idx + min_quefrency
                        echo_magnitude = np.abs(cepstrum_region[echo_peak_idx])
                        
                        if echo_magnitude > 0.05:  # Significant echo detected
                            # Remove echo by modifying cepstrum
                            cepstrum_modified = cepstrum.copy()
                            # Zero out the echo peak and surrounding area
                            zero_width = int(0.005 * self.sample_rate)  # 5ms width
                            start_idx = max(0, echo_quefrency - zero_width)
                            end_idx = min(len(cepstrum), echo_quefrency + zero_width)
                            cepstrum_modified[start_idx:end_idx] = 0
                            
                            # Reconstruct spectrum and audio
                            modified_log_spectrum = np.fft.fft(cepstrum_modified)
                            modified_spectrum = np.exp(modified_log_spectrum) * np.exp(1j * np.angle(spectrum))
                            reconstructed_audio = np.real(np.fft.ifft(modified_spectrum))
                            
                            # Apply window compensation safely
                            window_mean = np.mean(window)
                            if window_mean > 1e-10:
                                window_compensation = window / window_mean
                                # Avoid division by very small numbers
                                safe_compensation = np.where(window_compensation > 0.1, window_compensation, 1.0)
                                audio = reconstructed_audio / safe_compensation
                            else:
                                audio = reconstructed_audio
                            
                            metrics["echo_detected"] = True
                            metrics["echo_delay_ms"] = float(echo_quefrency / self.sample_rate * 1000)
                            metrics["echo_magnitude"] = float(echo_magnitude)
                            
            except Exception as e:
                logger.warning(f"Cepstral echo removal failed: {e}")
                metrics["cepstral_echo_removal_failed"] = str(e)
            
            # 2. Reverb reduction using spectral subtraction
            try:
                # Estimate reverb from the tail of the audio
                tail_length = min(int(0.5 * self.sample_rate), len(audio) // 4)
                if tail_length > 1024:
                    reverb_tail = audio[-tail_length:]
                    
                    # Compute reverb spectrum
                    reverb_stft = librosa.stft(reverb_tail, n_fft=self.n_fft, hop_length=self.hop_length)
                    reverb_magnitude = np.mean(np.abs(reverb_stft), axis=1, keepdims=True)
                    
                    # Apply reverb reduction to entire audio
                    audio_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                    audio_magnitude = np.abs(audio_stft)
                    audio_phase = np.angle(audio_stft)
                    
                    # Ensure magnitudes are finite
                    if not np.all(np.isfinite(audio_magnitude)):
                        audio_magnitude = np.nan_to_num(audio_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
                    if not np.all(np.isfinite(reverb_magnitude)):
                        reverb_magnitude = np.nan_to_num(reverb_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Subtract reverb spectrum
                    reverb_reduction = 0.4  # Moderate reduction to avoid artifacts
                    enhanced_magnitude = np.maximum(
                        audio_magnitude - reverb_reduction * reverb_magnitude,
                        0.1 * audio_magnitude  # Maintain 10% of original
                    )
                    
                    # Reconstruct audio
                    enhanced_stft = enhanced_magnitude * np.exp(1j * audio_phase)
                    audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
                    
                    metrics["reverb_reduced"] = True
                    metrics["reverb_reduction_factor"] = reverb_reduction
                    
            except Exception as e:
                logger.warning(f"Reverb reduction failed: {e}")
                metrics["reverb_reduction_failed"] = str(e)
            
            # Ensure output is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate improvement
            improvement = self._calculate_echo_reduction(original_audio, audio)
            metrics["echo_reduction_score"] = float(improvement)
            
            return audio.astype(np.float32), metrics
            
        except Exception as e:
            logger.warning(f"Echo/reverb removal failed: {e}")
            return audio, {"error": str(e)}
    
    def _enhance_audio_clarity(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhance audio clarity and remove muffled sound."""
        try:
            original_audio = audio.copy()
            metrics = {}
            
            # Ensure audio is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                metrics["non_finite_fixed"] = True
            
            # Skip processing if audio is too short or all zeros
            if len(audio) < 512 or np.max(np.abs(audio)) < 1e-10:
                metrics["skipped_short_or_silent"] = True
                return audio.astype(np.float32), metrics
            
            # 1. Spectral enhancement for clarity
            try:
                stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
                
                # Ensure magnitude is finite
                if not np.all(np.isfinite(magnitude)):
                    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Enhance speech frequencies (300-3400 Hz)
                freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
                speech_mask = (freqs >= 300) & (freqs <= 3400)
                
                # Apply gentle boost to speech frequencies
                enhancement_factor = 1.2
                magnitude[speech_mask, :] *= enhancement_factor
                
                # Enhance high frequencies for crispness (2-8 kHz)
                crisp_mask = (freqs >= 2000) & (freqs <= 8000)
                crisp_enhancement = 1.1
                magnitude[crisp_mask, :] *= crisp_enhancement
                
                # Reconstruct audio
                enhanced_stft = magnitude * np.exp(1j * phase)
                audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
                
                metrics["speech_enhancement"] = enhancement_factor
                metrics["crisp_enhancement"] = crisp_enhancement
                
            except Exception as e:
                logger.warning(f"Spectral enhancement failed: {e}")
                metrics["spectral_enhancement_failed"] = str(e)
            
            # 2. Dynamic range optimization
            try:
                # Apply gentle compression for consistency
                rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
                
                if len(rms) > 0 and np.max(rms) > 1e-10:
                    # Calculate compression parameters
                    threshold = np.percentile(rms, 70)
                    ratio = 2.5
                    
                    # Apply frame-based compression
                    compressed_rms = np.where(
                        rms > threshold,
                        threshold + (rms - threshold) / ratio,
                        rms
                    )
                    
                    # Apply compression to audio
                    compression_ratio = np.divide(compressed_rms, rms, out=np.ones_like(rms), where=rms!=0)
                    
                    if len(compression_ratio) > 1:
                        # Create proper x-coordinates for interpolation
                        x_coords = np.arange(0, len(audio), 512)[:len(compression_ratio)]
                        if len(x_coords) == len(compression_ratio):
                            compression_interp = np.interp(
                                np.arange(len(audio)),
                                x_coords,
                                compression_ratio
                            )
                            
                            audio = audio * compression_interp
                        
                        metrics["compression_applied"] = True
                        metrics["compression_ratio"] = ratio
                        
            except Exception as e:
                logger.warning(f"Dynamic range optimization failed: {e}")
                metrics["compression_failed"] = str(e)
            
            # 3. Noise reduction using spectral gating
            try:
                # Estimate noise floor
                rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
                
                if len(rms) > 0:
                    noise_threshold = np.percentile(rms, 15)  # Bottom 15% as noise
                    noise_frames = rms < noise_threshold
                    
                    if np.sum(noise_frames) > 0:
                        # Estimate noise spectrum
                        audio_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                        noise_spectrum = np.mean(np.abs(audio_stft[:, noise_frames]), axis=1, keepdims=True)
                        
                        # Apply spectral subtraction
                        audio_magnitude = np.abs(audio_stft)
                        audio_phase = np.angle(audio_stft)
                        
                        # Ensure magnitudes are finite
                        if not np.all(np.isfinite(audio_magnitude)):
                            audio_magnitude = np.nan_to_num(audio_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
                        if not np.all(np.isfinite(noise_spectrum)):
                            noise_spectrum = np.nan_to_num(noise_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Subtract noise with over-subtraction
                        alpha = 1.5  # Over-subtraction factor
                        beta = 0.1   # Spectral floor
                        
                        enhanced_magnitude = audio_magnitude - alpha * noise_spectrum
                        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * audio_magnitude)
                        
                        # Reconstruct
                        enhanced_stft = enhanced_magnitude * np.exp(1j * audio_phase)
                        audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
                        
                        metrics["noise_reduction_applied"] = True
                        metrics["noise_reduction_factor"] = alpha
                        
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
                metrics["noise_reduction_failed"] = str(e)
            
            # Ensure output is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate clarity improvement
            clarity_improvement = self._calculate_clarity_improvement(original_audio, audio)
            metrics["clarity_improvement_score"] = float(clarity_improvement)
            
            return audio.astype(np.float32), metrics
            
        except Exception as e:
            logger.warning(f"Clarity enhancement failed: {e}")
            return audio, {"error": str(e)}
    
    def _clean_spectral_artifacts(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Clean spectral artifacts and anomalies."""
        try:
            original_audio = audio.copy()
            metrics = {}
            
            # Compute spectrogram
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 1. Remove spectral outliers
            # Calculate statistics for each frequency bin
            freq_means = np.mean(magnitude, axis=1)
            freq_stds = np.std(magnitude, axis=1)
            
            # Identify and suppress outliers
            outlier_threshold = 3.0  # 3 standard deviations
            outliers_removed = 0
            
            for freq_bin in range(magnitude.shape[0]):
                outlier_mask = np.abs(magnitude[freq_bin, :] - freq_means[freq_bin]) > (
                    outlier_threshold * freq_stds[freq_bin]
                )
                if np.any(outlier_mask):
                    # Replace outliers with median values
                    median_val = np.median(magnitude[freq_bin, :])
                    magnitude[freq_bin, outlier_mask] = median_val
                    outliers_removed += np.sum(outlier_mask)
            
            metrics["spectral_outliers_removed"] = int(outliers_removed)
            
            # 2. Smooth spectral discontinuities
            # Apply gentle smoothing across frequency bins
            for time_frame in range(magnitude.shape[1]):
                magnitude[:, time_frame] = savgol_filter(
                    magnitude[:, time_frame], 
                    window_length=5, 
                    polyorder=2
                )
            
            # 3. Remove harmonic distortion artifacts
            # Detect and suppress non-harmonic peaks
            freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
            
            # Focus on speech fundamental frequency range
            f0_range = (80, 400)  # Hz
            f0_bins = np.where((freqs >= f0_range[0]) & (freqs <= f0_range[1]))[0]
            
            if len(f0_bins) > 0:
                # Find dominant frequency in each frame
                for frame_idx in range(magnitude.shape[1]):
                    frame_spectrum = magnitude[:, frame_idx]
                    
                    # Find peaks in F0 range
                    f0_spectrum = frame_spectrum[f0_bins]
                    peaks, _ = scipy.signal.find_peaks(f0_spectrum, height=np.max(f0_spectrum) * 0.3)
                    
                    if len(peaks) > 0:
                        # Get the strongest peak as fundamental
                        strongest_peak = peaks[np.argmax(f0_spectrum[peaks])]
                        f0_freq = freqs[f0_bins[strongest_peak]]
                        
                        # Check harmonics and suppress non-harmonic content
                        for harmonic in range(2, 8):  # Check first 7 harmonics
                            harmonic_freq = f0_freq * harmonic
                            if harmonic_freq < self.sample_rate / 2:
                                harmonic_bin = np.argmin(np.abs(freqs - harmonic_freq))
                                
                                # Suppress frequencies between harmonics
                                if harmonic > 2:  # Start from 3rd harmonic
                                    prev_harmonic_freq = f0_freq * (harmonic - 1)
                                    prev_harmonic_bin = np.argmin(np.abs(freqs - prev_harmonic_freq))
                                    
                                    # Suppress inter-harmonic content
                                    inter_harmonic_bins = range(
                                        prev_harmonic_bin + 5, 
                                        harmonic_bin - 5
                                    )
                                    for bin_idx in inter_harmonic_bins:
                                        if 0 <= bin_idx < len(frame_spectrum):
                                            magnitude[bin_idx, frame_idx] *= 0.7
            
            # Reconstruct audio
            cleaned_stft = magnitude * np.exp(1j * phase)
            audio = librosa.istft(cleaned_stft, hop_length=self.hop_length)
            
            # Calculate spectral cleaning improvement
            spectral_improvement = self._calculate_spectral_improvement(original_audio, audio)
            metrics["spectral_improvement_score"] = float(spectral_improvement)
            
            return audio.astype(np.float32), metrics
            
        except Exception as e:
            logger.warning(f"Spectral cleaning failed: {e}")
            return audio, {"error": str(e)}
    
    def _match_reference_characteristics(
        self, 
        audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Match audio characteristics to reference."""
        try:
            original_audio = audio.copy()
            metrics = {}
            
            # 1. Match spectral envelope
            audio_stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            ref_stft = librosa.stft(reference_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            
            audio_magnitude = np.abs(audio_stft)
            ref_magnitude = np.abs(ref_stft)
            
            # Calculate spectral envelopes
            audio_envelope = np.mean(audio_magnitude, axis=1, keepdims=True)
            ref_envelope = np.mean(ref_magnitude, axis=1, keepdims=True)
            
            # Apply partial spectral matching
            matching_strength = 0.3  # Gentle matching to avoid artifacts
            envelope_ratio = ref_envelope / (audio_envelope + 1e-10)
            
            # Smooth the ratio to prevent artifacts
            envelope_ratio_smooth = scipy.ndimage.gaussian_filter1d(
                envelope_ratio.flatten(), sigma=2
            ).reshape(envelope_ratio.shape)
            
            # Apply matching
            matched_magnitude = audio_magnitude * (
                (1 - matching_strength) + matching_strength * envelope_ratio_smooth
            )
            
            # Reconstruct
            matched_stft = matched_magnitude * np.exp(1j * np.angle(audio_stft))
            audio = librosa.istft(matched_stft, hop_length=self.hop_length)
            
            metrics["spectral_matching_applied"] = True
            metrics["matching_strength"] = matching_strength
            
            # 2. Match dynamic characteristics
            audio_rms = librosa.feature.rms(y=audio)[0]
            ref_rms = librosa.feature.rms(y=reference_audio)[0]
            
            # Match RMS envelope if reference is longer
            if len(ref_rms) > len(audio_rms):
                ref_rms_matched = ref_rms[:len(audio_rms)]
            else:
                ref_rms_matched = np.interp(
                    np.linspace(0, 1, len(audio_rms)),
                    np.linspace(0, 1, len(ref_rms)),
                    ref_rms
                )
            
            # Apply gentle RMS matching
            rms_ratio = ref_rms_matched / (audio_rms + 1e-10)
            rms_ratio_smooth = savgol_filter(rms_ratio, window_length=5, polyorder=2)
            
            # Interpolate to audio length
            x_coords = np.arange(0, len(audio), self.hop_length)[:len(rms_ratio_smooth)]
            if len(x_coords) == len(rms_ratio_smooth) and len(x_coords) > 0:
                rms_interp = np.interp(
                    np.arange(len(audio)),
                    x_coords,
                    rms_ratio_smooth
                )
                
                # Apply with reduced strength
                dynamic_strength = 0.2
                final_ratio = (1 - dynamic_strength) + dynamic_strength * rms_interp
                audio = audio * final_ratio
                
                metrics["dynamic_matching_applied"] = True
                metrics["dynamic_strength"] = dynamic_strength
            
            # Calculate matching quality
            matching_quality = self._calculate_matching_quality(original_audio, audio, reference_audio)
            metrics["matching_quality_score"] = float(matching_quality)
            
            return audio.astype(np.float32), metrics
            
        except Exception as e:
            logger.warning(f"Reference matching failed: {e}")
            return audio, {"error": str(e)}
    
    def _apply_final_enhancements(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply final quality enhancements."""
        try:
            original_audio = audio.copy()
            metrics = {}
            
            # Ensure audio is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                metrics["non_finite_fixed"] = True
            
            # Skip processing if audio is all zeros
            if np.max(np.abs(audio)) < 1e-10:
                metrics["skipped_silent"] = True
                return audio.astype(np.float32), metrics
            
            # 1. Remove DC offset
            audio = audio - np.mean(audio)
            
            # 2. Apply gentle limiting to prevent clipping
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0.95:
                audio = audio * (0.95 / max_amplitude)
                metrics["limiting_applied"] = True
                metrics["limiting_factor"] = 0.95 / max_amplitude
            
            # 3. Final normalization with headroom
            target_level = 0.8  # Leave headroom
            current_level = np.sqrt(np.mean(audio**2))
            if current_level > 1e-10:
                normalization_factor = target_level / current_level
                audio = audio * normalization_factor
                metrics["normalization_factor"] = float(normalization_factor)
            
            # 4. Apply gentle high-frequency roll-off to remove harshness
            try:
                nyquist = self.sample_rate / 2
                rolloff_freq = 0.9 * nyquist  # 90% of Nyquist
                
                if rolloff_freq > 100:  # Only if reasonable frequency
                    b, a = butter(2, rolloff_freq / nyquist, btype='low')
                    audio = filtfilt(b, a, audio)
                    
                    metrics["high_freq_rolloff_applied"] = True
                    metrics["rolloff_frequency"] = rolloff_freq
                    
            except Exception as e:
                logger.warning(f"High-frequency rolloff failed: {e}")
                metrics["rolloff_failed"] = str(e)
            
            # Ensure output is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate final enhancement score
            enhancement_score = self._calculate_final_enhancement(original_audio, audio)
            metrics["final_enhancement_score"] = float(enhancement_score)
            
            return audio.astype(np.float32), metrics
            
        except Exception as e:
            logger.warning(f"Final enhancement failed: {e}")
            return audio, {"error": str(e)}
    
    # Helper methods for quality assessment
    def _calculate_quality_improvement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate overall quality improvement score."""
        try:
            # Ensure both arrays are finite
            if not np.all(np.isfinite(original)):
                original = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(enhanced)):
                enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Skip if either is silent
            if np.max(np.abs(original)) < 1e-10 or np.max(np.abs(enhanced)) < 1e-10:
                return 0.0
            
            # Calculate various quality metrics
            original_snr = self._estimate_snr(original)
            enhanced_snr = self._estimate_snr(enhanced)
            
            original_clarity = self._estimate_clarity(original)
            enhanced_clarity = self._estimate_clarity(enhanced)
            
            # Combine metrics with safety checks
            snr_improvement = 0.0
            if abs(original_snr) > 1e-10:
                snr_improvement = (enhanced_snr - original_snr) / abs(original_snr)
            
            clarity_improvement = 0.0
            if abs(original_clarity) > 1e-10:
                clarity_improvement = (enhanced_clarity - original_clarity) / abs(original_clarity)
            
            overall_improvement = (snr_improvement + clarity_improvement) / 2
            return float(np.clip(overall_improvement, -1.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Quality improvement calculation failed: {e}")
            return 0.0
    
    def _calculate_artifact_reduction(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate artifact reduction score."""
        try:
            # Measure autocorrelation artifacts
            orig_autocorr = np.correlate(original, original, mode='full')
            proc_autocorr = np.correlate(processed, processed, mode='full')
            
            # Look for secondary peaks (artifacts)
            center = len(orig_autocorr) // 2
            search_range = slice(center + 100, center + 5000)  # 100-5000 samples delay
            
            orig_artifacts = np.sum(np.abs(orig_autocorr[search_range]))
            proc_artifacts = np.sum(np.abs(proc_autocorr[search_range]))
            
            reduction = (orig_artifacts - proc_artifacts) / (orig_artifacts + 1e-10)
            return np.clip(reduction, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_echo_reduction(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate echo reduction score."""
        try:
            # Use cepstral analysis to measure echo reduction
            orig_cepstrum = np.abs(np.fft.ifft(np.log(np.abs(np.fft.fft(original)) + 1e-10)))
            proc_cepstrum = np.abs(np.fft.ifft(np.log(np.abs(np.fft.fft(processed)) + 1e-10)))
            
            # Look for echo peaks in quefrency domain
            echo_range = slice(100, 2000)  # Typical echo range
            
            orig_echo_energy = np.sum(orig_cepstrum[echo_range])
            proc_echo_energy = np.sum(proc_cepstrum[echo_range])
            
            reduction = (orig_echo_energy - proc_echo_energy) / (orig_echo_energy + 1e-10)
            return np.clip(reduction, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_clarity_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate clarity improvement score."""
        try:
            orig_clarity = self._estimate_clarity(original)
            proc_clarity = self._estimate_clarity(processed)
            
            improvement = (proc_clarity - orig_clarity) / (orig_clarity + 1e-10)
            return np.clip(improvement, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_spectral_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate spectral improvement score."""
        try:
            # Measure spectral smoothness
            orig_spectrum = np.abs(np.fft.fft(original))
            proc_spectrum = np.abs(np.fft.fft(processed))
            
            # Calculate spectral smoothness (lower variance = smoother)
            orig_smoothness = 1.0 / (np.var(np.diff(orig_spectrum)) + 1e-10)
            proc_smoothness = 1.0 / (np.var(np.diff(proc_spectrum)) + 1e-10)
            
            improvement = (proc_smoothness - orig_smoothness) / (orig_smoothness + 1e-10)
            return np.clip(improvement, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_matching_quality(self, original: np.ndarray, processed: np.ndarray, 
                                  reference: np.ndarray) -> float:
        """Calculate reference matching quality."""
        try:
            # Compare spectral similarity to reference
            orig_similarity = self._spectral_similarity(original, reference)
            proc_similarity = self._spectral_similarity(processed, reference)
            
            improvement = proc_similarity - orig_similarity
            return np.clip(improvement, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_final_enhancement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate final enhancement score."""
        try:
            # Combine multiple quality measures
            snr_improvement = self._calculate_quality_improvement(original, processed)
            clarity_improvement = self._calculate_clarity_improvement(original, processed)
            spectral_improvement = self._calculate_spectral_improvement(original, processed)
            
            # Weighted combination
            final_score = (
                0.4 * snr_improvement +
                0.4 * clarity_improvement +
                0.2 * spectral_improvement
            )
            
            return np.clip(final_score, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        try:
            # Ensure audio is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Skip if audio is silent
            if np.max(np.abs(audio)) < 1e-10:
                return 10.0  # Default reasonable SNR
            
            # Simple SNR estimation using energy distribution
            rms = librosa.feature.rms(y=audio)[0]
            
            if len(rms) == 0 or np.max(rms) < 1e-10:
                return 10.0
            
            signal_energy = np.percentile(rms, 80)  # Top 20% as signal
            noise_energy = np.percentile(rms, 20)   # Bottom 20% as noise
            
            if noise_energy < 1e-10:
                noise_energy = 1e-10  # Prevent division by zero
            
            snr = 20 * np.log10(signal_energy / noise_energy)
            return float(np.clip(snr, -20, 60))  # Reasonable SNR range
            
        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}")
            return 10.0  # Default reasonable SNR
    
    def _estimate_clarity(self, audio: np.ndarray) -> float:
        """Estimate audio clarity."""
        try:
            # Ensure audio is finite
            if not np.all(np.isfinite(audio)):
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Skip if audio is silent
            if np.max(np.abs(audio)) < 1e-10:
                return 0.5  # Default moderate clarity
            
            # Clarity based on spectral centroid and high-frequency content
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            
            if not np.isfinite(spectral_centroid) or spectral_centroid < 1:
                spectral_centroid = 1000  # Default reasonable value
            
            # High-frequency energy ratio
            try:
                stft = librosa.stft(audio)
                freqs = librosa.fft_frequencies(sr=self.sample_rate)
                
                high_freq_mask = freqs > 2000
                low_freq_mask = freqs < 2000
                
                if np.any(high_freq_mask) and np.any(low_freq_mask):
                    high_energy = np.mean(np.abs(stft[high_freq_mask, :]))
                    low_energy = np.mean(np.abs(stft[low_freq_mask, :]))
                    
                    if low_energy > 1e-10:
                        hf_ratio = high_energy / low_energy
                    else:
                        hf_ratio = 0.5
                else:
                    hf_ratio = 0.5
                    
            except Exception:
                hf_ratio = 0.5
            
            # Combine metrics
            clarity = (spectral_centroid / 3000) * 0.7 + (hf_ratio / 2) * 0.3
            return float(np.clip(clarity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Clarity estimation failed: {e}")
            return 0.5  # Default moderate clarity
    
    def _spectral_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate spectral similarity between two audio signals."""
        try:
            # Compute spectral features
            mfcc1 = librosa.feature.mfcc(y=audio1, sr=self.sample_rate, n_mfcc=13)
            mfcc2 = librosa.feature.mfcc(y=audio2, sr=self.sample_rate, n_mfcc=13)
            
            # Calculate mean MFCCs
            mfcc1_mean = np.mean(mfcc1, axis=1)
            mfcc2_mean = np.mean(mfcc2, axis=1)
            
            # Cosine similarity
            similarity = np.dot(mfcc1_mean, mfcc2_mean) / (
                np.linalg.norm(mfcc1_mean) * np.linalg.norm(mfcc2_mean) + 1e-10
            )
            
            return float(np.clip((similarity + 1) / 2, 0.0, 1.0))  # Normalize to [0, 1]
            
        except Exception:
            return 0.5  # Default moderate similarity


# Global processor instance
enhanced_audio_processor = EnhancedAudioProcessor()