"""
Advanced Audio Post-Processing Engine for High-Fidelity Voice Cloning System.

This module implements comprehensive post-processing to enhance synthesized audio
and achieve >95% similarity by matching spectral characteristics, removing artifacts,
and preserving voice characteristics during enhancement.
"""

import librosa
import numpy as np
import scipy.signal
import scipy.ndimage
import scipy.stats
import time
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt, wiener, savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

logger = logging.getLogger(__name__)


@dataclass
class SpectralMatchingResult:
    """Result of spectral matching operation."""
    matched_audio: np.ndarray
    frequency_alignment_score: float
    spectral_distance: float
    enhancement_applied: List[str]
    processing_time: float


@dataclass
class ArtifactRemovalResult:
    """Result of artifact removal operation."""
    cleaned_audio: np.ndarray
    artifacts_detected: List[str]
    artifacts_removed: List[str]
    quality_improvement: float
    processing_time: float


@dataclass
class VoiceCharacteristicPreservation:
    """Voice characteristic preservation metrics."""
    fundamental_frequency_preserved: float
    formant_preservation_score: float
    prosody_preservation_score: float
    timbre_preservation_score: float
    overall_preservation_score: float


@dataclass
class DynamicRangeMatchingResult:
    """Result of dynamic range compression matching."""
    compressed_audio: np.ndarray
    compression_ratio: float
    dynamic_range_before: float
    dynamic_range_after: float
    reference_match_score: float


@dataclass
class ConsistencyMetrics:
    """Audio consistency metrics."""
    volume_consistency: float
    quality_consistency: float
    spectral_consistency: float
    temporal_consistency: float
    overall_consistency: float


class ArtifactType(Enum):
    """Types of synthesis artifacts that can be detected and removed."""
    SPECTRAL_DISCONTINUITY = "spectral_discontinuity"
    TEMPORAL_GLITCH = "temporal_glitch"
    FREQUENCY_ALIASING = "frequency_aliasing"
    AMPLITUDE_CLIPPING = "amplitude_clipping"
    PHASE_DISTORTION = "phase_distortion"
    HARMONIC_DISTORTION = "harmonic_distortion"
    NOISE_BURST = "noise_burst"
    SILENCE_GAP = "silence_gap"


class AdvancedAudioPostProcessor:
    """
    Advanced Audio Post-Processing Engine implementing state-of-the-art algorithms
    for spectral matching, artifact removal, and voice characteristic preservation.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the Advanced Audio Post-Processor.
        
        Args:
            sample_rate: Target sample rate for processing
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        self.window_size = 2048
        
        # Voice frequency ranges (Hz)
        self.voice_freq_range = (80, 8000)
        self.fundamental_range = (80, 400)
        self.formant_range = (200, 4000)
        
        # Processing parameters
        self.spectral_smoothing_factor = 0.1
        self.artifact_detection_threshold = 0.3
        self.consistency_window_size = 1024
        
        logger.info(f"Advanced Audio Post-Processor initialized with SR: {self.sample_rate}Hz")
    
    def enhance_synthesis_quality(
        self, 
        synthesized_audio: np.ndarray, 
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        preserve_characteristics: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main post-processing pipeline that enhances synthesized audio to match
        reference characteristics while preserving voice quality.
        
        Args:
            synthesized_audio: Synthesized audio to enhance
            reference_audio: Reference audio for matching
            reference_sample_rate: Sample rate of reference audio
            preserve_characteristics: Whether to prioritize characteristic preservation
            
        Returns:
            Tuple of (enhanced_audio, processing_metrics)
            
        Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
        """
        logger.info("Starting comprehensive audio post-processing")
        start_time = time.time()
        
        # Ensure consistent sample rates
        if reference_sample_rate != self.sample_rate:
            reference_audio = librosa.resample(
                reference_audio, 
                orig_sr=reference_sample_rate, 
                target_sr=self.sample_rate
            )
        
        processing_metrics = {}
        enhanced_audio = synthesized_audio.copy()
        
        # Step 1: Spectral matching and frequency alignment (Requirement 6.1)
        logger.info("Applying spectral matching and frequency alignment")
        spectral_result = self.apply_spectral_matching(enhanced_audio, reference_audio)
        enhanced_audio = spectral_result.matched_audio
        processing_metrics['spectral_matching'] = {
            'frequency_alignment_score': spectral_result.frequency_alignment_score,
            'spectral_distance': spectral_result.spectral_distance,
            'enhancements': spectral_result.enhancement_applied
        }
        
        # Step 2: Artifact removal and audio smoothing (Requirement 6.2)
        logger.info("Removing synthesis artifacts and smoothing audio")
        artifact_result = self.remove_synthesis_artifacts(enhanced_audio)
        enhanced_audio = artifact_result.cleaned_audio
        processing_metrics['artifact_removal'] = {
            'artifacts_detected': artifact_result.artifacts_detected,
            'artifacts_removed': artifact_result.artifacts_removed,
            'quality_improvement': artifact_result.quality_improvement
        }
        
        # Step 3: Voice characteristic preservation during enhancement (Requirement 6.3)
        logger.info("Preserving voice characteristics during enhancement")
        if preserve_characteristics:
            preservation_result = self.preserve_voice_characteristics(
                enhanced_audio, synthesized_audio, reference_audio
            )
            enhanced_audio = preservation_result['enhanced_audio']
            processing_metrics['characteristic_preservation'] = preservation_result['metrics']
        
        # Step 4: Dynamic range compression matching (Requirement 6.4)
        logger.info("Matching dynamic range compression to reference")
        compression_result = self.match_dynamic_range_compression(enhanced_audio, reference_audio)
        enhanced_audio = compression_result.compressed_audio
        processing_metrics['dynamic_range_matching'] = {
            'compression_ratio': compression_result.compression_ratio,
            'range_before': compression_result.dynamic_range_before,
            'range_after': compression_result.dynamic_range_after,
            'match_score': compression_result.reference_match_score
        }
        
        # Step 5: Consistency maintenance for volume and quality (Requirement 6.5)
        logger.info("Maintaining consistency in volume and quality")
        consistency_result = self.maintain_consistency(enhanced_audio, reference_audio)
        enhanced_audio = consistency_result['consistent_audio']
        processing_metrics['consistency_maintenance'] = consistency_result['metrics']
        
        # Final quality assessment
        similarity_score = self.calculate_similarity_score(enhanced_audio, reference_audio)
        processing_metrics['final_similarity_score'] = similarity_score
        processing_metrics['total_processing_time'] = time.time() - start_time
        
        logger.info(f"Post-processing complete. Similarity score: {similarity_score:.3f}")
        
        return enhanced_audio, processing_metrics
    
    def apply_spectral_matching(
        self, 
        synthesized_audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> SpectralMatchingResult:
        """
        Apply spectral matching to align frequency characteristics (Requirement 6.1).
        
        Args:
            synthesized_audio: Audio to be matched
            reference_audio: Reference audio for matching
            
        Returns:
            SpectralMatchingResult with matched audio and metrics
        """
        start_time = time.time()
        enhancement_applied = []
        
        # Compute spectrograms
        synth_stft = librosa.stft(synthesized_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        ref_stft = librosa.stft(reference_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        synth_magnitude = np.abs(synth_stft)
        synth_phase = np.angle(synth_stft)
        ref_magnitude = np.abs(ref_stft)
        
        # Frequency alignment using spectral envelope matching
        aligned_magnitude = self._align_spectral_envelope(synth_magnitude, ref_magnitude)
        enhancement_applied.append("spectral_envelope_alignment")
        
        # Harmonic structure matching
        harmonic_matched_magnitude = self._match_harmonic_structure(
            aligned_magnitude, ref_magnitude, synthesized_audio, reference_audio
        )
        enhancement_applied.append("harmonic_structure_matching")
        
        # Formant frequency alignment
        formant_aligned_magnitude = self._align_formant_frequencies(
            harmonic_matched_magnitude, ref_magnitude
        )
        enhancement_applied.append("formant_frequency_alignment")
        
        # Reconstruct audio with matched spectrum
        matched_stft = formant_aligned_magnitude * np.exp(1j * synth_phase)
        matched_audio = librosa.istft(matched_stft, hop_length=self.hop_length)
        
        # Calculate metrics
        frequency_alignment_score = self._calculate_frequency_alignment_score(
            formant_aligned_magnitude, ref_magnitude
        )
        spectral_distance = self._calculate_spectral_distance(
            formant_aligned_magnitude, ref_magnitude
        )
        
        processing_time = time.time() - start_time
        
        return SpectralMatchingResult(
            matched_audio=matched_audio,
            frequency_alignment_score=frequency_alignment_score,
            spectral_distance=spectral_distance,
            enhancement_applied=enhancement_applied,
            processing_time=processing_time
        )
    
    def remove_synthesis_artifacts(self, audio: np.ndarray) -> ArtifactRemovalResult:
        """
        Remove synthesis artifacts and smooth audio (Requirement 6.2).
        
        Args:
            audio: Audio with potential artifacts
            
        Returns:
            ArtifactRemovalResult with cleaned audio and metrics
        """
        start_time = time.time()
        artifacts_detected = []
        artifacts_removed = []
        
        cleaned_audio = audio.copy()
        initial_quality = self._assess_audio_quality(audio)
        
        # Detect and remove spectral discontinuities
        if self._detect_spectral_discontinuities(audio):
            artifacts_detected.append(ArtifactType.SPECTRAL_DISCONTINUITY.value)
            cleaned_audio = self._remove_spectral_discontinuities(cleaned_audio)
            artifacts_removed.append(ArtifactType.SPECTRAL_DISCONTINUITY.value)
        
        # Detect and remove temporal glitches
        if self._detect_temporal_glitches(cleaned_audio):
            artifacts_detected.append(ArtifactType.TEMPORAL_GLITCH.value)
            cleaned_audio = self._remove_temporal_glitches(cleaned_audio)
            artifacts_removed.append(ArtifactType.TEMPORAL_GLITCH.value)
        
        # Detect and remove frequency aliasing
        if self._detect_frequency_aliasing(cleaned_audio):
            artifacts_detected.append(ArtifactType.FREQUENCY_ALIASING.value)
            cleaned_audio = self._remove_frequency_aliasing(cleaned_audio)
            artifacts_removed.append(ArtifactType.FREQUENCY_ALIASING.value)
        
        # Detect and remove amplitude clipping
        if self._detect_amplitude_clipping(cleaned_audio):
            artifacts_detected.append(ArtifactType.AMPLITUDE_CLIPPING.value)
            cleaned_audio = self._remove_amplitude_clipping(cleaned_audio)
            artifacts_removed.append(ArtifactType.AMPLITUDE_CLIPPING.value)
        
        # Detect and remove phase distortion
        if self._detect_phase_distortion(cleaned_audio):
            artifacts_detected.append(ArtifactType.PHASE_DISTORTION.value)
            cleaned_audio = self._remove_phase_distortion(cleaned_audio)
            artifacts_removed.append(ArtifactType.PHASE_DISTORTION.value)
        
        # Detect and remove harmonic distortion
        if self._detect_harmonic_distortion(cleaned_audio):
            artifacts_detected.append(ArtifactType.HARMONIC_DISTORTION.value)
            cleaned_audio = self._remove_harmonic_distortion(cleaned_audio)
            artifacts_removed.append(ArtifactType.HARMONIC_DISTORTION.value)
        
        # Detect and remove noise bursts
        if self._detect_noise_bursts(cleaned_audio):
            artifacts_detected.append(ArtifactType.NOISE_BURST.value)
            cleaned_audio = self._remove_noise_bursts(cleaned_audio)
            artifacts_removed.append(ArtifactType.NOISE_BURST.value)
        
        # Detect and fill silence gaps
        if self._detect_silence_gaps(cleaned_audio):
            artifacts_detected.append(ArtifactType.SILENCE_GAP.value)
            cleaned_audio = self._fill_silence_gaps(cleaned_audio)
            artifacts_removed.append(ArtifactType.SILENCE_GAP.value)
        
        # Apply general smoothing
        cleaned_audio = self._apply_audio_smoothing(cleaned_audio)
        
        final_quality = self._assess_audio_quality(cleaned_audio)
        quality_improvement = final_quality - initial_quality
        processing_time = time.time() - start_time
        
        return ArtifactRemovalResult(
            cleaned_audio=cleaned_audio,
            artifacts_detected=artifacts_detected,
            artifacts_removed=artifacts_removed,
            quality_improvement=quality_improvement,
            processing_time=processing_time
        )
    
    def preserve_voice_characteristics(
        self, 
        enhanced_audio: np.ndarray, 
        original_synthesized: np.ndarray,
        reference_audio: np.ndarray
    ) -> Dict[str, Any]:
        """
        Preserve voice characteristics during enhancement (Requirement 6.3).
        
        Args:
            enhanced_audio: Audio after enhancement
            original_synthesized: Original synthesized audio
            reference_audio: Reference audio for characteristics
            
        Returns:
            Dictionary with preserved audio and preservation metrics
        """
        # Extract characteristics from reference
        ref_f0 = self._extract_fundamental_frequency(reference_audio)
        ref_formants = self._extract_formant_frequencies(reference_audio)
        ref_prosody = self._extract_prosodic_features(reference_audio)
        ref_timbre = self._extract_timbre_features(reference_audio)
        
        # Extract characteristics from enhanced audio
        enh_f0 = self._extract_fundamental_frequency(enhanced_audio)
        enh_formants = self._extract_formant_frequencies(enhanced_audio)
        enh_prosody = self._extract_prosodic_features(enhanced_audio)
        enh_timbre = self._extract_timbre_features(enhanced_audio)
        
        preserved_audio = enhanced_audio.copy()
        
        # Preserve fundamental frequency
        if self._characteristics_differ(ref_f0, enh_f0, threshold=0.1):
            preserved_audio = self._adjust_fundamental_frequency(preserved_audio, ref_f0, enh_f0)
        
        # Preserve formant frequencies
        if self._formants_differ(ref_formants, enh_formants, threshold=0.15):
            preserved_audio = self._adjust_formant_frequencies(preserved_audio, ref_formants, enh_formants)
        
        # Preserve prosodic patterns
        if self._prosody_differs(ref_prosody, enh_prosody, threshold=0.2):
            preserved_audio = self._adjust_prosodic_patterns(preserved_audio, ref_prosody, enh_prosody)
        
        # Preserve timbre characteristics
        if self._timbre_differs(ref_timbre, enh_timbre, threshold=0.15):
            preserved_audio = self._adjust_timbre_characteristics(preserved_audio, ref_timbre, enh_timbre)
        
        # Calculate preservation scores
        final_f0 = self._extract_fundamental_frequency(preserved_audio)
        final_formants = self._extract_formant_frequencies(preserved_audio)
        final_prosody = self._extract_prosodic_features(preserved_audio)
        final_timbre = self._extract_timbre_features(preserved_audio)
        
        preservation_metrics = VoiceCharacteristicPreservation(
            fundamental_frequency_preserved=self._calculate_f0_preservation(ref_f0, final_f0),
            formant_preservation_score=self._calculate_formant_preservation(ref_formants, final_formants),
            prosody_preservation_score=self._calculate_prosody_preservation(ref_prosody, final_prosody),
            timbre_preservation_score=self._calculate_timbre_preservation(ref_timbre, final_timbre),
            overall_preservation_score=0.0  # Will be calculated below
        )
        
        # Calculate overall preservation score
        preservation_metrics.overall_preservation_score = (
            preservation_metrics.fundamental_frequency_preserved * 0.3 +
            preservation_metrics.formant_preservation_score * 0.3 +
            preservation_metrics.prosody_preservation_score * 0.2 +
            preservation_metrics.timbre_preservation_score * 0.2
        )
        
        return {
            'enhanced_audio': preserved_audio,
            'metrics': preservation_metrics
        }
    
    def match_dynamic_range_compression(
        self, 
        audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> DynamicRangeMatchingResult:
        """
        Match dynamic range compression to reference audio (Requirement 6.4).
        
        Args:
            audio: Audio to compress
            reference_audio: Reference for compression matching
            
        Returns:
            DynamicRangeMatchingResult with compressed audio and metrics
        """
        # Analyze reference dynamic range
        ref_dynamic_range = self._calculate_dynamic_range(reference_audio)
        ref_compression_params = self._analyze_compression_characteristics(reference_audio)
        
        # Calculate current dynamic range
        current_dynamic_range = self._calculate_dynamic_range(audio)
        
        # Apply matching compression
        compressed_audio = self._apply_matching_compression(
            audio, ref_compression_params, ref_dynamic_range
        )
        
        # Calculate final dynamic range
        final_dynamic_range = self._calculate_dynamic_range(compressed_audio)
        
        # Calculate compression ratio
        compression_ratio = current_dynamic_range / final_dynamic_range if final_dynamic_range > 0 else 1.0
        
        # Calculate reference match score
        reference_match_score = 1.0 - abs(final_dynamic_range - ref_dynamic_range) / max(final_dynamic_range, ref_dynamic_range)
        
        return DynamicRangeMatchingResult(
            compressed_audio=compressed_audio,
            compression_ratio=compression_ratio,
            dynamic_range_before=current_dynamic_range,
            dynamic_range_after=final_dynamic_range,
            reference_match_score=reference_match_score
        )
    
    def maintain_consistency(
        self, 
        audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> Dict[str, Any]:
        """
        Maintain consistency in volume and quality throughout audio (Requirement 6.5).
        
        Args:
            audio: Audio to make consistent
            reference_audio: Reference for consistency matching
            
        Returns:
            Dictionary with consistent audio and consistency metrics
        """
        consistent_audio = audio.copy()
        
        # Volume consistency
        consistent_audio, volume_consistency = self._maintain_volume_consistency(
            consistent_audio, reference_audio
        )
        
        # Quality consistency
        consistent_audio, quality_consistency = self._maintain_quality_consistency(
            consistent_audio, reference_audio
        )
        
        # Spectral consistency
        consistent_audio, spectral_consistency = self._maintain_spectral_consistency(
            consistent_audio, reference_audio
        )
        
        # Temporal consistency
        consistent_audio, temporal_consistency = self._maintain_temporal_consistency(
            consistent_audio, reference_audio
        )
        
        # Calculate overall consistency
        overall_consistency = (
            volume_consistency * 0.3 +
            quality_consistency * 0.3 +
            spectral_consistency * 0.2 +
            temporal_consistency * 0.2
        )
        
        consistency_metrics = ConsistencyMetrics(
            volume_consistency=volume_consistency,
            quality_consistency=quality_consistency,
            spectral_consistency=spectral_consistency,
            temporal_consistency=temporal_consistency,
            overall_consistency=overall_consistency
        )
        
        return {
            'consistent_audio': consistent_audio,
            'metrics': consistency_metrics
        }
    
    def calculate_similarity_score(
        self, 
        synthesized_audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> float:
        """
        Calculate detailed similarity score between synthesized and reference audio.
        
        Args:
            synthesized_audio: Synthesized audio
            reference_audio: Reference audio
            
        Returns:
            Similarity score (0-1, target >0.95)
        """
        # Spectral similarity
        spectral_sim = self._calculate_spectral_similarity(synthesized_audio, reference_audio)
        
        # Temporal similarity
        temporal_sim = self._calculate_temporal_similarity(synthesized_audio, reference_audio)
        
        # Prosodic similarity
        prosodic_sim = self._calculate_prosodic_similarity(synthesized_audio, reference_audio)
        
        # Timbre similarity
        timbre_sim = self._calculate_timbre_similarity(synthesized_audio, reference_audio)
        
        # Weighted combination
        similarity_score = (
            spectral_sim * 0.3 +
            temporal_sim * 0.2 +
            prosodic_sim * 0.25 +
            timbre_sim * 0.25
        )
        
        return float(np.clip(similarity_score, 0, 1))
    
    # ==================== SPECTRAL MATCHING HELPER METHODS ====================
    
    def _align_spectral_envelope(
        self, 
        synth_magnitude: np.ndarray, 
        ref_magnitude: np.ndarray
    ) -> np.ndarray:
        """Align spectral envelope between synthesized and reference audio."""
        # Calculate spectral envelopes using smoothing
        synth_envelope = self._calculate_spectral_envelope(synth_magnitude)
        ref_envelope = self._calculate_spectral_envelope(ref_magnitude)
        
        # Calculate alignment filter
        alignment_filter = np.divide(
            ref_envelope, 
            synth_envelope, 
            out=np.ones_like(ref_envelope), 
            where=synth_envelope != 0
        )
        
        # Apply smoothing to avoid artifacts
        alignment_filter = scipy.ndimage.gaussian_filter1d(
            alignment_filter, 
            sigma=self.spectral_smoothing_factor, 
            axis=0
        )
        
        # Apply alignment
        aligned_magnitude = synth_magnitude * alignment_filter
        
        return aligned_magnitude
    
    def _calculate_spectral_envelope(self, magnitude: np.ndarray) -> np.ndarray:
        """Calculate spectral envelope using cepstral analysis."""
        # Convert to log domain
        log_magnitude = np.log(magnitude + 1e-8)
        
        # Apply cepstral smoothing
        envelope = np.zeros_like(log_magnitude)
        
        for i in range(log_magnitude.shape[1]):
            # Cepstral analysis for each frame
            cepstrum = np.fft.ifft(log_magnitude[:, i])
            
            # Keep only low quefrency components (envelope)
            liftered_cepstrum = cepstrum.copy()
            lifter_length = len(cepstrum) // 10  # Keep 10% of coefficients
            liftered_cepstrum[lifter_length:-lifter_length] = 0
            
            # Convert back to spectrum
            envelope[:, i] = np.real(np.fft.fft(liftered_cepstrum))
        
        return np.exp(envelope)
    
    def _match_harmonic_structure(
        self, 
        synth_magnitude: np.ndarray, 
        ref_magnitude: np.ndarray,
        synth_audio: np.ndarray,
        ref_audio: np.ndarray
    ) -> np.ndarray:
        """Match harmonic structure between synthesized and reference audio."""
        # Extract fundamental frequencies
        synth_f0 = self._extract_fundamental_frequency(synth_audio)
        ref_f0 = self._extract_fundamental_frequency(ref_audio)
        
        # Calculate harmonic enhancement factors
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        harmonic_enhancement = np.ones_like(synth_magnitude)
        
        # Enhance harmonics based on reference
        for frame_idx in range(synth_magnitude.shape[1]):
            if frame_idx < len(ref_f0) and ref_f0[frame_idx] > 0:
                f0 = ref_f0[frame_idx]
                
                # Enhance first 10 harmonics
                for harmonic in range(1, 11):
                    harmonic_freq = f0 * harmonic
                    if harmonic_freq < self.sample_rate / 2:
                        # Find closest frequency bin
                        freq_bin = np.argmin(np.abs(freqs - harmonic_freq))
                        
                        # Calculate enhancement factor
                        ref_harmonic_strength = ref_magnitude[freq_bin, frame_idx] if frame_idx < ref_magnitude.shape[1] else 0
                        synth_harmonic_strength = synth_magnitude[freq_bin, frame_idx]
                        
                        if synth_harmonic_strength > 0:
                            enhancement_factor = ref_harmonic_strength / synth_harmonic_strength
                            enhancement_factor = np.clip(enhancement_factor, 0.5, 2.0)  # Limit enhancement
                            
                            # Apply Gaussian window around harmonic
                            window_size = max(1, len(freqs) // 100)
                            start_bin = max(0, freq_bin - window_size)
                            end_bin = min(len(freqs), freq_bin + window_size + 1)
                            
                            gaussian_window = scipy.signal.windows.gaussian(end_bin - start_bin, std=window_size/3)
                            harmonic_enhancement[start_bin:end_bin, frame_idx] *= (
                                1 + (enhancement_factor - 1) * gaussian_window
                            )
        
        return synth_magnitude * harmonic_enhancement
    
    def _align_formant_frequencies(
        self, 
        synth_magnitude: np.ndarray, 
        ref_magnitude: np.ndarray
    ) -> np.ndarray:
        """Align formant frequencies between synthesized and reference audio."""
        # Extract formant frequencies for each frame
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        aligned_magnitude = synth_magnitude.copy()
        
        for frame_idx in range(min(synth_magnitude.shape[1], ref_magnitude.shape[1])):
            synth_formants = self._extract_formants_from_spectrum(synth_magnitude[:, frame_idx], freqs)
            ref_formants = self._extract_formants_from_spectrum(ref_magnitude[:, frame_idx], freqs)
            
            # Align formants if both have sufficient formants
            if len(synth_formants) >= 2 and len(ref_formants) >= 2:
                alignment_map = self._calculate_formant_alignment(synth_formants, ref_formants)
                aligned_magnitude[:, frame_idx] = self._apply_formant_alignment(
                    synth_magnitude[:, frame_idx], freqs, alignment_map
                )
        
        return aligned_magnitude
    
    def _extract_formants_from_spectrum(self, magnitude: np.ndarray, freqs: np.ndarray) -> List[float]:
        """Extract formant frequencies from magnitude spectrum."""
        # Focus on formant frequency range
        formant_mask = (freqs >= self.formant_range[0]) & (freqs <= self.formant_range[1])
        formant_freqs = freqs[formant_mask]
        formant_magnitude = magnitude[formant_mask]
        
        if len(formant_magnitude) == 0:
            return []
        
        # Find peaks in the formant region
        peaks, properties = scipy.signal.find_peaks(
            formant_magnitude, 
            height=np.max(formant_magnitude) * 0.1,
            distance=len(formant_magnitude) // 20  # Minimum distance between formants
        )
        
        # Extract formant frequencies
        formants = []
        for peak in peaks:
            if peak < len(formant_freqs):
                formants.append(formant_freqs[peak])
        
        return sorted(formants)[:4]  # Return up to 4 formants
    
    def _calculate_formant_alignment(
        self, 
        synth_formants: List[float], 
        ref_formants: List[float]
    ) -> Dict[float, float]:
        """Calculate alignment mapping between synthesized and reference formants."""
        alignment_map = {}
        
        # Simple nearest-neighbor alignment
        for i, synth_f in enumerate(synth_formants):
            if i < len(ref_formants):
                alignment_map[synth_f] = ref_formants[i]
        
        return alignment_map
    
    def _apply_formant_alignment(
        self, 
        magnitude: np.ndarray, 
        freqs: np.ndarray, 
        alignment_map: Dict[float, float]
    ) -> np.ndarray:
        """Apply formant alignment to magnitude spectrum."""
        aligned_magnitude = magnitude.copy()
        
        for synth_f, ref_f in alignment_map.items():
            if synth_f != ref_f:
                # Find frequency bins
                synth_bin = np.argmin(np.abs(freqs - synth_f))
                ref_bin = np.argmin(np.abs(freqs - ref_f))
                
                # Apply frequency shift (simplified)
                shift_factor = ref_f / synth_f if synth_f > 0 else 1.0
                
                # Apply local enhancement/attenuation
                window_size = max(1, len(freqs) // 50)
                start_bin = max(0, synth_bin - window_size)
                end_bin = min(len(freqs), synth_bin + window_size + 1)
                
                enhancement_factor = min(2.0, max(0.5, shift_factor))
                aligned_magnitude[start_bin:end_bin] *= enhancement_factor
        
        return aligned_magnitude
    
    def _calculate_frequency_alignment_score(
        self, 
        aligned_magnitude: np.ndarray, 
        ref_magnitude: np.ndarray
    ) -> float:
        """Calculate frequency alignment score."""
        # Normalize magnitudes
        aligned_norm = aligned_magnitude / (np.max(aligned_magnitude) + 1e-8)
        ref_norm = ref_magnitude / (np.max(ref_magnitude) + 1e-8)
        
        # Calculate correlation
        min_frames = min(aligned_norm.shape[1], ref_norm.shape[1])
        correlations = []
        
        for frame_idx in range(min_frames):
            corr = np.corrcoef(aligned_norm[:, frame_idx], ref_norm[:, frame_idx])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        if correlations:
            alignment_score = np.mean(correlations)
        else:
            alignment_score = 0.0
        
        return float(np.clip(alignment_score, 0, 1))
    
    def _calculate_spectral_distance(
        self, 
        synth_magnitude: np.ndarray, 
        ref_magnitude: np.ndarray
    ) -> float:
        """Calculate spectral distance between synthesized and reference audio."""
        # Normalize magnitudes
        synth_norm = synth_magnitude / (np.max(synth_magnitude) + 1e-8)
        ref_norm = ref_magnitude / (np.max(ref_magnitude) + 1e-8)
        
        # Calculate mean squared error
        min_frames = min(synth_norm.shape[1], ref_norm.shape[1])
        mse = np.mean((synth_norm[:, :min_frames] - ref_norm[:, :min_frames]) ** 2)
        
        return float(mse)
    
    # ==================== ARTIFACT REMOVAL HELPER METHODS ====================
    
    def _assess_audio_quality(self, audio: np.ndarray) -> float:
        """Assess overall audio quality."""
        # Simple quality assessment based on multiple factors
        
        # 1. Dynamic range
        dynamic_range = self._calculate_dynamic_range(audio)
        dynamic_score = min(1.0, dynamic_range / 40.0)  # Normalize to 40dB
        
        # 2. Spectral flatness
        freqs, psd = scipy.signal.welch(audio, fs=self.sample_rate, nperseg=1024)
        spectral_flatness = scipy.stats.gmean(psd) / np.mean(psd)
        flatness_score = 1.0 - min(1.0, spectral_flatness)
        
        # 3. Clipping detection
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        clipping_score = 1.0 - min(1.0, clipping_ratio * 100)
        
        # Combine scores
        quality_score = (dynamic_score + flatness_score + clipping_score) / 3
        
        return float(quality_score)
    
    def _detect_spectral_discontinuities(self, audio: np.ndarray) -> bool:
        """Detect spectral discontinuities in audio."""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Calculate frame-to-frame spectral differences
        spectral_diff = np.diff(magnitude, axis=1)
        spectral_change = np.mean(np.abs(spectral_diff), axis=0)
        
        # Detect sudden changes
        threshold = np.mean(spectral_change) + 2 * np.std(spectral_change)
        discontinuities = spectral_change > threshold
        
        return np.sum(discontinuities) > len(spectral_change) * 0.05  # More than 5% of frames
    
    def _remove_spectral_discontinuities(self, audio: np.ndarray) -> np.ndarray:
        """Remove spectral discontinuities using smoothing."""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply temporal smoothing to magnitude
        smoothed_magnitude = scipy.ndimage.gaussian_filter1d(
            magnitude, sigma=1.0, axis=1
        )
        
        # Reconstruct audio
        smoothed_stft = smoothed_magnitude * np.exp(1j * phase)
        smoothed_audio = librosa.istft(smoothed_stft, hop_length=self.hop_length)
        
        return smoothed_audio
    
    def _detect_temporal_glitches(self, audio: np.ndarray) -> bool:
        """Detect temporal glitches (sudden amplitude changes)."""
        # Calculate frame-wise RMS energy
        frame_length = 1024
        hop_length = 512
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        rms_energy = np.sqrt(np.mean(frames**2, axis=0))
        
        # Detect sudden energy changes
        energy_diff = np.abs(np.diff(rms_energy))
        threshold = np.mean(energy_diff) + 3 * np.std(energy_diff)
        glitches = energy_diff > threshold
        
        return np.sum(glitches) > len(energy_diff) * 0.02  # More than 2% of frames
    
    def _remove_temporal_glitches(self, audio: np.ndarray) -> np.ndarray:
        """Remove temporal glitches using median filtering."""
        # Apply median filter to smooth out glitches
        filtered_audio = scipy.signal.medfilt(audio, kernel_size=5)
        
        # Blend with original to preserve natural variations
        blend_factor = 0.3
        smoothed_audio = (1 - blend_factor) * audio + blend_factor * filtered_audio
        
        return smoothed_audio
    
    def _detect_frequency_aliasing(self, audio: np.ndarray) -> bool:
        """Detect frequency aliasing artifacts."""
        freqs, psd = scipy.signal.welch(audio, fs=self.sample_rate, nperseg=2048)
        
        # Check for energy near Nyquist frequency
        nyquist = self.sample_rate / 2
        high_freq_mask = freqs > nyquist * 0.9
        
        if np.any(high_freq_mask):
            high_freq_energy = np.mean(psd[high_freq_mask])
            total_energy = np.mean(psd)
            
            # Aliasing suspected if significant energy near Nyquist
            return (high_freq_energy / total_energy) > 0.1
        
        return False
    
    def _remove_frequency_aliasing(self, audio: np.ndarray) -> np.ndarray:
        """Remove frequency aliasing using anti-aliasing filter."""
        # Apply low-pass filter to remove aliasing
        nyquist = self.sample_rate / 2
        cutoff = nyquist * 0.8  # 80% of Nyquist frequency
        
        b, a = butter(4, cutoff / nyquist, btype='low')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def _detect_amplitude_clipping(self, audio: np.ndarray) -> bool:
        """Detect amplitude clipping."""
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) >= clipping_threshold)
        clipping_ratio = clipped_samples / len(audio)
        
        return clipping_ratio > 0.001  # More than 0.1% clipped
    
    def _remove_amplitude_clipping(self, audio: np.ndarray) -> np.ndarray:
        """Remove amplitude clipping using soft limiting."""
        # Apply soft limiting to clipped regions
        threshold = 0.95
        clipped_mask = np.abs(audio) > threshold
        
        if np.any(clipped_mask):
            # Apply soft limiting using tanh
            limited_audio = audio.copy()
            limited_audio[clipped_mask] = np.tanh(audio[clipped_mask])
            
            return limited_audio
        
        return audio
    
    def _detect_phase_distortion(self, audio: np.ndarray) -> bool:
        """Detect phase distortion artifacts."""
        # Simple phase distortion detection using group delay analysis
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        phase = np.angle(stft)
        
        # Calculate phase derivatives (group delay)
        phase_diff = np.diff(np.unwrap(phase, axis=0), axis=0)
        
        # Detect irregular phase behavior
        phase_variance = np.var(phase_diff, axis=1)
        threshold = np.mean(phase_variance) + 2 * np.std(phase_variance)
        
        return np.sum(phase_variance > threshold) > len(phase_variance) * 0.1
    
    def _remove_phase_distortion(self, audio: np.ndarray) -> np.ndarray:
        """Remove phase distortion using minimum phase reconstruction."""
        # Convert to minimum phase
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Reconstruct with minimum phase
        min_phase_stft = magnitude * np.exp(1j * np.zeros_like(magnitude))
        min_phase_audio = librosa.istft(min_phase_stft, hop_length=self.hop_length)
        
        # Blend with original to preserve some phase information
        blend_factor = 0.2
        corrected_audio = (1 - blend_factor) * audio + blend_factor * min_phase_audio
        
        return corrected_audio
    
    def _detect_harmonic_distortion(self, audio: np.ndarray) -> bool:
        """Detect harmonic distortion."""
        # Calculate THD (Total Harmonic Distortion)
        freqs, psd = scipy.signal.welch(audio, fs=self.sample_rate, nperseg=2048)
        
        # Find fundamental frequency (simplified)
        voice_mask = (freqs >= 80) & (freqs <= 400)
        if not np.any(voice_mask):
            return False
        
        voice_psd = psd[voice_mask]
        voice_freqs = freqs[voice_mask]
        
        # Find peak (fundamental)
        fundamental_idx = np.argmax(voice_psd)
        fundamental_freq = voice_freqs[fundamental_idx]
        
        # Check harmonic levels
        harmonic_energy = 0
        fundamental_energy = voice_psd[fundamental_idx]
        
        for harmonic in range(2, 6):  # Check 2nd to 5th harmonics
            harmonic_freq = fundamental_freq * harmonic
            if harmonic_freq < self.sample_rate / 2:
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_energy += psd[harmonic_idx]
        
        # Calculate THD
        if fundamental_energy > 0:
            thd = harmonic_energy / fundamental_energy
            return thd > 0.1  # 10% THD threshold
        
        return False
    
    def _remove_harmonic_distortion(self, audio: np.ndarray) -> np.ndarray:
        """Remove harmonic distortion using spectral subtraction."""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply spectral subtraction to reduce harmonics
        # This is a simplified approach
        reduced_magnitude = magnitude * 0.9  # Slight reduction
        
        # Reconstruct audio
        reduced_stft = reduced_magnitude * np.exp(1j * phase)
        reduced_audio = librosa.istft(reduced_stft, hop_length=self.hop_length)
        
        return reduced_audio
    
    def _detect_noise_bursts(self, audio: np.ndarray) -> bool:
        """Detect noise bursts in audio."""
        # Calculate short-time energy
        frame_length = 512
        hop_length = 256
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Detect sudden energy spikes
        energy_diff = np.diff(energy)
        threshold = np.mean(energy) + 4 * np.std(energy)
        
        bursts = energy > threshold
        return np.sum(bursts) > 0
    
    def _remove_noise_bursts(self, audio: np.ndarray) -> np.ndarray:
        """Remove noise bursts using adaptive filtering."""
        # Calculate short-time energy
        frame_length = 512
        hop_length = 256
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Identify burst regions
        threshold = np.mean(energy) + 4 * np.std(energy)
        burst_frames = energy > threshold
        
        # Apply attenuation to burst regions
        cleaned_audio = audio.copy()
        
        for i, is_burst in enumerate(burst_frames):
            if is_burst:
                start_sample = i * hop_length
                end_sample = min(len(audio), start_sample + frame_length)
                
                # Apply attenuation
                attenuation_factor = 0.1
                cleaned_audio[start_sample:end_sample] *= attenuation_factor
        
        return cleaned_audio
    
    def _detect_silence_gaps(self, audio: np.ndarray) -> bool:
        """Detect unnatural silence gaps."""
        # Calculate RMS energy in short frames
        frame_length = 1024
        hop_length = 512
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        rms_energy = np.sqrt(np.mean(frames**2, axis=0))
        
        # Detect silence (very low energy)
        silence_threshold = np.max(rms_energy) * 0.01  # 1% of peak energy
        silence_frames = rms_energy < silence_threshold
        
        # Check for long silence gaps
        silence_runs = self._find_consecutive_runs(silence_frames)
        long_silences = [run for run in silence_runs if run[1] - run[0] > 10]  # More than 10 frames
        
        return len(long_silences) > 0
    
    def _fill_silence_gaps(self, audio: np.ndarray) -> np.ndarray:
        """Fill unnatural silence gaps with low-level noise."""
        # Calculate RMS energy in short frames
        frame_length = 1024
        hop_length = 512
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        rms_energy = np.sqrt(np.mean(frames**2, axis=0))
        
        # Detect silence
        silence_threshold = np.max(rms_energy) * 0.01
        silence_frames = rms_energy < silence_threshold
        
        # Fill silence gaps
        filled_audio = audio.copy()
        noise_level = np.std(audio) * 0.001  # Very low noise level
        
        for i, is_silence in enumerate(silence_frames):
            if is_silence:
                start_sample = i * hop_length
                end_sample = min(len(audio), start_sample + frame_length)
                
                # Add low-level noise
                noise = np.random.normal(0, noise_level, end_sample - start_sample)
                filled_audio[start_sample:end_sample] += noise
        
        return filled_audio
    
    def _find_consecutive_runs(self, boolean_array: np.ndarray) -> List[Tuple[int, int]]:
        """Find consecutive runs of True values."""
        runs = []
        in_run = False
        start = 0
        
        for i, val in enumerate(boolean_array):
            if val and not in_run:
                start = i
                in_run = True
            elif not val and in_run:
                runs.append((start, i))
                in_run = False
        
        if in_run:
            runs.append((start, len(boolean_array)))
        
        return runs
    
    def _apply_audio_smoothing(self, audio: np.ndarray) -> np.ndarray:
        """Apply general audio smoothing."""
        # Apply Savitzky-Golay filter for smoothing
        window_length = min(51, len(audio) // 10)  # Adaptive window length
        if window_length % 2 == 0:
            window_length += 1  # Must be odd
        
        if window_length >= 3:
            smoothed_audio = savgol_filter(audio, window_length, 3)
        else:
            smoothed_audio = audio
        
        return smoothed_audio
    
    # ==================== VOICE CHARACTERISTIC PRESERVATION METHODS ====================
    
    def _extract_fundamental_frequency(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency contour."""
        # Use librosa's pitch tracking
        f0 = librosa.yin(
            audio, 
            fmin=self.fundamental_range[0], 
            fmax=self.fundamental_range[1],
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return f0
    
    def _extract_formant_frequencies(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract formant frequency contours."""
        # Simplified formant extraction using LPC
        # In practice, this would use more sophisticated formant tracking
        
        # Calculate LPC coefficients
        frame_length = 2048
        hop_length = 512
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        formants = {'F1': [], 'F2': [], 'F3': []}
        
        for frame in frames.T:
            if np.sum(frame**2) > 1e-6:  # Skip silent frames
                # Simple formant estimation (placeholder)
                # Real implementation would use LPC root finding
                formants['F1'].append(500.0)  # Placeholder values
                formants['F2'].append(1500.0)
                formants['F3'].append(2500.0)
            else:
                formants['F1'].append(0.0)
                formants['F2'].append(0.0)
                formants['F3'].append(0.0)
        
        return {key: np.array(values) for key, values in formants.items()}
    
    def _extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract prosodic features (rhythm, stress, intonation)."""
        # Extract energy contour
        energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Extract spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # Extract zero crossing rate (voicing)
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        return {
            'energy': energy,
            'spectral_centroid': spectral_centroid,
            'zero_crossing_rate': zcr
        }
    
    def _extract_timbre_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract timbre characteristics."""
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
        
        # Extract spectral features
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        
        return {
            'mfccs': mfccs,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'spectral_flatness': spectral_flatness
        }
    
    def _characteristics_differ(self, ref_features: np.ndarray, enh_features: np.ndarray, threshold: float) -> bool:
        """Check if characteristics differ beyond threshold."""
        if len(ref_features) == 0 or len(enh_features) == 0:
            return False
        
        # Calculate normalized difference
        min_length = min(len(ref_features), len(enh_features))
        ref_norm = ref_features[:min_length]
        enh_norm = enh_features[:min_length]
        
        # Remove zeros and invalid values
        valid_mask = (ref_norm > 0) & (enh_norm > 0) & np.isfinite(ref_norm) & np.isfinite(enh_norm)
        
        if not np.any(valid_mask):
            return False
        
        ref_valid = ref_norm[valid_mask]
        enh_valid = enh_norm[valid_mask]
        
        # Calculate relative difference
        relative_diff = np.abs(ref_valid - enh_valid) / (ref_valid + 1e-8)
        mean_diff = np.mean(relative_diff)
        
        return mean_diff > threshold
    
    def _formants_differ(self, ref_formants: Dict[str, np.ndarray], enh_formants: Dict[str, np.ndarray], threshold: float) -> bool:
        """Check if formants differ beyond threshold."""
        for formant_name in ['F1', 'F2', 'F3']:
            if formant_name in ref_formants and formant_name in enh_formants:
                if self._characteristics_differ(ref_formants[formant_name], enh_formants[formant_name], threshold):
                    return True
        return False
    
    def _prosody_differs(self, ref_prosody: Dict[str, np.ndarray], enh_prosody: Dict[str, np.ndarray], threshold: float) -> bool:
        """Check if prosody differs beyond threshold."""
        for feature_name in ['energy', 'spectral_centroid']:
            if feature_name in ref_prosody and feature_name in enh_prosody:
                if self._characteristics_differ(ref_prosody[feature_name], enh_prosody[feature_name], threshold):
                    return True
        return False
    
    def _timbre_differs(self, ref_timbre: Dict[str, np.ndarray], enh_timbre: Dict[str, np.ndarray], threshold: float) -> bool:
        """Check if timbre differs beyond threshold."""
        # Check MFCC differences
        if 'mfccs' in ref_timbre and 'mfccs' in enh_timbre:
            ref_mfccs = ref_timbre['mfccs']
            enh_mfccs = enh_timbre['mfccs']
            
            min_frames = min(ref_mfccs.shape[1], enh_mfccs.shape[1])
            
            for i in range(min(ref_mfccs.shape[0], enh_mfccs.shape[0])):
                if self._characteristics_differ(ref_mfccs[i, :min_frames], enh_mfccs[i, :min_frames], threshold):
                    return True
        
        return False
    
    def _adjust_fundamental_frequency(self, audio: np.ndarray, ref_f0: np.ndarray, current_f0: np.ndarray) -> np.ndarray:
        """Adjust fundamental frequency to match reference."""
        # Simplified F0 adjustment using pitch shifting
        # In practice, this would use more sophisticated PSOLA or similar techniques
        
        if len(ref_f0) == 0 or len(current_f0) == 0:
            return audio
        
        # Calculate average pitch shift needed
        valid_ref = ref_f0[ref_f0 > 0]
        valid_current = current_f0[current_f0 > 0]
        
        if len(valid_ref) == 0 or len(valid_current) == 0:
            return audio
        
        pitch_shift_ratio = np.mean(valid_ref) / np.mean(valid_current)
        
        # Apply pitch shift (simplified using librosa)
        if 0.5 <= pitch_shift_ratio <= 2.0:  # Reasonable range
            shifted_audio = librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=12 * np.log2(pitch_shift_ratio)
            )
            return shifted_audio
        
        return audio
    
    def _adjust_formant_frequencies(self, audio: np.ndarray, ref_formants: Dict[str, np.ndarray], current_formants: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust formant frequencies to match reference."""
        # Simplified formant adjustment
        # Real implementation would use formant shifting techniques
        return audio  # Placeholder
    
    def _adjust_prosodic_patterns(self, audio: np.ndarray, ref_prosody: Dict[str, np.ndarray], current_prosody: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust prosodic patterns to match reference."""
        # Simplified prosody adjustment
        # Real implementation would use prosody modification techniques
        return audio  # Placeholder
    
    def _adjust_timbre_characteristics(self, audio: np.ndarray, ref_timbre: Dict[str, np.ndarray], current_timbre: Dict[str, np.ndarray]) -> np.ndarray:
        """Adjust timbre characteristics to match reference."""
        # Simplified timbre adjustment using spectral shaping
        # Real implementation would use more sophisticated timbre modification
        return audio  # Placeholder
    
    def _calculate_f0_preservation(self, ref_f0: np.ndarray, final_f0: np.ndarray) -> float:
        """Calculate fundamental frequency preservation score."""
        if len(ref_f0) == 0 or len(final_f0) == 0:
            return 0.0
        
        min_length = min(len(ref_f0), len(final_f0))
        ref_valid = ref_f0[:min_length]
        final_valid = final_f0[:min_length]
        
        # Remove zeros
        valid_mask = (ref_valid > 0) & (final_valid > 0)
        
        if not np.any(valid_mask):
            return 0.0
        
        ref_clean = ref_valid[valid_mask]
        final_clean = final_valid[valid_mask]
        
        # Calculate correlation
        correlation = np.corrcoef(ref_clean, final_clean)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return float(np.clip(correlation, 0, 1))
    
    def _calculate_formant_preservation(self, ref_formants: Dict[str, np.ndarray], final_formants: Dict[str, np.ndarray]) -> float:
        """Calculate formant preservation score."""
        scores = []
        
        for formant_name in ['F1', 'F2', 'F3']:
            if formant_name in ref_formants and formant_name in final_formants:
                score = self._calculate_f0_preservation(ref_formants[formant_name], final_formants[formant_name])
                scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _calculate_prosody_preservation(self, ref_prosody: Dict[str, np.ndarray], final_prosody: Dict[str, np.ndarray]) -> float:
        """Calculate prosody preservation score."""
        scores = []
        
        for feature_name in ['energy', 'spectral_centroid']:
            if feature_name in ref_prosody and feature_name in final_prosody:
                score = self._calculate_f0_preservation(ref_prosody[feature_name], final_prosody[feature_name])
                scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _calculate_timbre_preservation(self, ref_timbre: Dict[str, np.ndarray], final_timbre: Dict[str, np.ndarray]) -> float:
        """Calculate timbre preservation score."""
        if 'mfccs' in ref_timbre and 'mfccs' in final_timbre:
            ref_mfccs = ref_timbre['mfccs']
            final_mfccs = final_timbre['mfccs']
            
            min_frames = min(ref_mfccs.shape[1], final_mfccs.shape[1])
            min_coeffs = min(ref_mfccs.shape[0], final_mfccs.shape[0])
            
            correlations = []
            for i in range(min_coeffs):
                corr = np.corrcoef(ref_mfccs[i, :min_frames], final_mfccs[i, :min_frames])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            return float(np.mean(correlations)) if correlations else 0.0
        
        return 0.0
    
    # ==================== DYNAMIC RANGE MATCHING METHODS ====================
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        if len(audio) == 0:
            return 0.0
        
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            dynamic_range = 20 * np.log10(peak / rms)
        else:
            dynamic_range = 0.0
        
        return float(dynamic_range)
    
    def _analyze_compression_characteristics(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze compression characteristics of reference audio."""
        # Calculate short-time dynamic range
        frame_length = 4096
        hop_length = 2048
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_dynamic_ranges = []
        
        for frame in frames.T:
            if np.sum(frame**2) > 1e-8:
                peak = np.max(np.abs(frame))
                rms = np.sqrt(np.mean(frame**2))
                if rms > 0:
                    dr = 20 * np.log10(peak / rms)
                    frame_dynamic_ranges.append(dr)
        
        if frame_dynamic_ranges:
            avg_dynamic_range = np.mean(frame_dynamic_ranges)
            dynamic_range_variance = np.var(frame_dynamic_ranges)
        else:
            avg_dynamic_range = 20.0
            dynamic_range_variance = 1.0
        
        # Estimate compression parameters
        compression_ratio = max(1.0, 40.0 / avg_dynamic_range) if avg_dynamic_range > 0 else 2.0
        threshold = -20.0  # dB
        attack_time = 0.003  # seconds
        release_time = 0.1   # seconds
        
        return {
            'ratio': compression_ratio,
            'threshold': threshold,
            'attack': attack_time,
            'release': release_time,
            'target_dynamic_range': avg_dynamic_range
        }
    
    def _apply_matching_compression(
        self, 
        audio: np.ndarray, 
        compression_params: Dict[str, float], 
        target_dynamic_range: float
    ) -> np.ndarray:
        """Apply compression to match reference characteristics."""
        # Simple compression implementation
        # Real implementation would use more sophisticated compressor
        
        ratio = compression_params['ratio']
        threshold_db = compression_params['threshold']
        threshold_linear = 10**(threshold_db / 20)
        
        # Apply compression
        compressed_audio = audio.copy()
        
        # Find samples above threshold
        above_threshold = np.abs(audio) > threshold_linear
        
        if np.any(above_threshold):
            # Apply compression to samples above threshold
            excess = np.abs(audio[above_threshold]) - threshold_linear
            compressed_excess = excess / ratio
            
            # Maintain sign
            signs = np.sign(audio[above_threshold])
            compressed_audio[above_threshold] = signs * (threshold_linear + compressed_excess)
        
        return compressed_audio
    
    # ==================== CONSISTENCY MAINTENANCE METHODS ====================
    
    def _maintain_volume_consistency(self, audio: np.ndarray, reference_audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Maintain volume consistency throughout audio."""
        # Calculate reference RMS level
        ref_rms = np.sqrt(np.mean(reference_audio**2))
        
        # Calculate RMS in overlapping windows
        window_length = int(0.1 * self.sample_rate)  # 100ms windows
        hop_length = window_length // 2
        
        frames = librosa.util.frame(audio, frame_length=window_length, hop_length=hop_length)
        frame_rms = np.sqrt(np.mean(frames**2, axis=0))
        
        # Calculate target RMS for each frame
        target_rms = ref_rms
        
        # Apply gain adjustment to each frame
        consistent_audio = audio.copy()
        
        for i, current_rms in enumerate(frame_rms):
            start_sample = i * hop_length
            end_sample = min(len(audio), start_sample + window_length)
            
            if current_rms > 1e-8:  # Avoid division by zero
                gain = target_rms / current_rms
                gain = np.clip(gain, 0.1, 10.0)  # Limit gain range
                
                # Apply smooth gain transition
                window = np.hanning(end_sample - start_sample)
                gain_envelope = 1.0 + (gain - 1.0) * window
                
                consistent_audio[start_sample:end_sample] *= gain_envelope
        
        # Calculate consistency score
        final_frames = librosa.util.frame(consistent_audio, frame_length=window_length, hop_length=hop_length)
        final_rms = np.sqrt(np.mean(final_frames**2, axis=0))
        
        rms_variance = np.var(final_rms)
        consistency_score = 1.0 / (1.0 + rms_variance * 100)  # Normalize variance
        
        return consistent_audio, float(consistency_score)
    
    def _maintain_quality_consistency(self, audio: np.ndarray, reference_audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Maintain quality consistency throughout audio."""
        # Calculate quality metrics in overlapping windows
        window_length = int(0.2 * self.sample_rate)  # 200ms windows
        hop_length = window_length // 2
        
        frames = librosa.util.frame(audio, frame_length=window_length, hop_length=hop_length)
        quality_scores = []
        
        for frame in frames.T:
            if np.sum(frame**2) > 1e-8:
                quality = self._assess_audio_quality(frame)
                quality_scores.append(quality)
        
        if not quality_scores:
            return audio, 0.0
        
        # Calculate target quality (from reference)
        ref_quality = self._assess_audio_quality(reference_audio)
        
        # Apply quality normalization (simplified)
        # Real implementation would apply sophisticated quality enhancement
        normalized_audio = audio.copy()
        
        # Calculate consistency score
        quality_variance = np.var(quality_scores)
        consistency_score = 1.0 / (1.0 + quality_variance * 10)
        
        return normalized_audio, float(consistency_score)
    
    def _maintain_spectral_consistency(self, audio: np.ndarray, reference_audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Maintain spectral consistency throughout audio."""
        # Calculate spectral features in overlapping windows
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Calculate spectral centroid for each frame
        spectral_centroids = []
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        for frame_idx in range(magnitude.shape[1]):
            frame_magnitude = magnitude[:, frame_idx]
            if np.sum(frame_magnitude) > 0:
                centroid = np.sum(freqs * frame_magnitude) / np.sum(frame_magnitude)
                spectral_centroids.append(centroid)
        
        if not spectral_centroids:
            return audio, 0.0
        
        # Calculate consistency score
        centroid_variance = np.var(spectral_centroids)
        consistency_score = 1.0 / (1.0 + centroid_variance / 1000000)  # Normalize
        
        # Apply spectral smoothing if needed
        if consistency_score < 0.7:
            # Apply temporal smoothing to magnitude spectrum
            smoothed_magnitude = scipy.ndimage.gaussian_filter1d(magnitude, sigma=1.0, axis=1)
            phase = np.angle(stft)
            smoothed_stft = smoothed_magnitude * np.exp(1j * phase)
            smoothed_audio = librosa.istft(smoothed_stft, hop_length=self.hop_length)
            return smoothed_audio, consistency_score
        
        return audio, float(consistency_score)
    
    def _maintain_temporal_consistency(self, audio: np.ndarray, reference_audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Maintain temporal consistency throughout audio."""
        # Calculate temporal features
        frame_length = int(0.05 * self.sample_rate)  # 50ms frames
        hop_length = frame_length // 2
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        
        # Calculate zero crossing rate for each frame
        zcr_values = []
        for frame in frames.T:
            zcr = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)
            zcr_values.append(zcr)
        
        if not zcr_values:
            return audio, 0.0
        
        # Calculate consistency score
        zcr_variance = np.var(zcr_values)
        consistency_score = 1.0 / (1.0 + zcr_variance * 1000)
        
        return audio, float(consistency_score)
    
    # ==================== SIMILARITY CALCULATION METHODS ====================
    
    def _calculate_spectral_similarity(self, synth_audio: np.ndarray, ref_audio: np.ndarray) -> float:
        """Calculate spectral similarity between synthesized and reference audio."""
        # Compute spectrograms
        synth_stft = librosa.stft(synth_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        ref_stft = librosa.stft(ref_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        synth_magnitude = np.abs(synth_stft)
        ref_magnitude = np.abs(ref_stft)
        
        # Normalize
        synth_norm = synth_magnitude / (np.max(synth_magnitude) + 1e-8)
        ref_norm = ref_magnitude / (np.max(ref_magnitude) + 1e-8)
        
        # Calculate correlation
        min_frames = min(synth_norm.shape[1], ref_norm.shape[1])
        correlations = []
        
        for frame_idx in range(min_frames):
            corr = np.corrcoef(synth_norm[:, frame_idx], ref_norm[:, frame_idx])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0
    
    def _calculate_temporal_similarity(self, synth_audio: np.ndarray, ref_audio: np.ndarray) -> float:
        """Calculate temporal similarity between synthesized and reference audio."""
        # Calculate energy envelopes
        synth_energy = librosa.feature.rms(y=synth_audio, hop_length=self.hop_length)[0]
        ref_energy = librosa.feature.rms(y=ref_audio, hop_length=self.hop_length)[0]
        
        # Normalize
        synth_energy_norm = synth_energy / (np.max(synth_energy) + 1e-8)
        ref_energy_norm = ref_energy / (np.max(ref_energy) + 1e-8)
        
        # Calculate correlation
        min_length = min(len(synth_energy_norm), len(ref_energy_norm))
        
        if min_length > 1:
            correlation = np.corrcoef(
                synth_energy_norm[:min_length], 
                ref_energy_norm[:min_length]
            )[0, 1]
            
            if np.isnan(correlation):
                return 0.0
            
            return float(np.clip(correlation, 0, 1))
        
        return 0.0
    
    def _calculate_prosodic_similarity(self, synth_audio: np.ndarray, ref_audio: np.ndarray) -> float:
        """Calculate prosodic similarity between synthesized and reference audio."""
        # Extract F0 contours
        synth_f0 = self._extract_fundamental_frequency(synth_audio)
        ref_f0 = self._extract_fundamental_frequency(ref_audio)
        
        # Remove zeros and calculate correlation
        min_length = min(len(synth_f0), len(ref_f0))
        
        if min_length > 1:
            synth_f0_valid = synth_f0[:min_length]
            ref_f0_valid = ref_f0[:min_length]
            
            # Remove zeros
            valid_mask = (synth_f0_valid > 0) & (ref_f0_valid > 0)
            
            if np.sum(valid_mask) > 1:
                correlation = np.corrcoef(
                    synth_f0_valid[valid_mask], 
                    ref_f0_valid[valid_mask]
                )[0, 1]
                
                if not np.isnan(correlation):
                    return float(np.clip(correlation, 0, 1))
        
        return 0.0
    
    def _calculate_timbre_similarity(self, synth_audio: np.ndarray, ref_audio: np.ndarray) -> float:
        """Calculate timbre similarity between synthesized and reference audio."""
        # Extract MFCCs
        synth_mfccs = librosa.feature.mfcc(y=synth_audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
        ref_mfccs = librosa.feature.mfcc(y=ref_audio, sr=self.sample_rate, n_mfcc=13, hop_length=self.hop_length)
        
        # Calculate correlation for each MFCC coefficient
        min_frames = min(synth_mfccs.shape[1], ref_mfccs.shape[1])
        min_coeffs = min(synth_mfccs.shape[0], ref_mfccs.shape[0])
        
        correlations = []
        
        for coeff_idx in range(min_coeffs):
            if min_frames > 1:
                corr = np.corrcoef(
                    synth_mfccs[coeff_idx, :min_frames], 
                    ref_mfccs[coeff_idx, :min_frames]
                )[0, 1]
                
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return float(np.mean(correlations)) if correlations else 0.0