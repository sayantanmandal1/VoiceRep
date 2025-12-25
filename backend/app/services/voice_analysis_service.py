"""
Voice analysis service for extracting voice characteristics and creating voice models.
"""

import librosa
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from scipy import signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

from app.schemas.voice import (
    VoiceProfileSchema, VoiceCharacteristics, FrequencyRange,
    ProsodyFeaturesSchema, EmotionalProfileSchema, QualityMetrics,
    VoiceAnalysisResult
)
from app.models.voice import VoiceProfile, VoiceModel, ProsodyFeatures, EmotionalProfile
from app.core.config import settings
from app.services.multi_dimensional_voice_analyzer import MultiDimensionalVoiceAnalyzer

# Simple performance monitor mock
class PerformanceMonitor:
    def start_operation(self, operation_type, operation_id, metadata=None):
        pass
    
    def end_operation(self, operation_id, success=True, error_message=None, additional_metadata=None):
        pass

performance_monitor = PerformanceMonitor()

logger = logging.getLogger(__name__)


class VoiceAnalyzer:
    """Main voice analysis engine for extracting voice characteristics."""
    
    def __init__(self):
        self.sample_rate = 22050  # Standard sample rate for voice analysis
        self.hop_length = 512
        self.n_fft = 2048
        self.n_mfcc = 13
        self.n_formants = 4
        
        # Initialize multi-dimensional analyzer
        self.multi_dimensional_analyzer = MultiDimensionalVoiceAnalyzer(sample_rate=self.sample_rate)
        
    def analyze_voice_comprehensive_multidimensional(self, audio_path: str) -> VoiceAnalysisResult:
        """
        Perform comprehensive multi-dimensional voice analysis with 1000+ features.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            VoiceAnalysisResult containing comprehensive voice characteristics
        """
        start_time = time.time()
        operation_id = f"multidim_voice_analysis_{int(time.time() * 1000)}"
        
        # Start performance monitoring
        performance_monitor.start_operation(
            'multidimensional_voice_analysis', 
            operation_id,
            {'audio_path': audio_path}
        )
        
        try:
            # Use multi-dimensional analyzer for comprehensive analysis
            comprehensive_analysis = self.multi_dimensional_analyzer.analyze_voice_comprehensive(audio_path)
            
            # Extract components from comprehensive analysis
            pitch_features = comprehensive_analysis['pitch_features']
            formant_features = comprehensive_analysis['formant_features']
            timbre_features = comprehensive_analysis['timbre_features']
            prosodic_features = comprehensive_analysis['prosodic_features']
            emotional_features = comprehensive_analysis['emotional_features']
            voice_fingerprint = comprehensive_analysis['voice_fingerprint']
            quality_metrics = comprehensive_analysis['quality_metrics']
            
            # Convert to schema-compatible formats
            fundamental_freq = FrequencyRange(
                min_hz=float(np.min(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0])) if len(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0]) > 0 else 0.0,
                max_hz=float(np.max(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0])) if len(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0]) > 0 else 0.0,
                mean_hz=float(np.mean(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0])) if len(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0]) > 0 else 0.0,
                std_hz=float(np.std(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0])) if len(pitch_features.fundamental_frequency[pitch_features.fundamental_frequency > 0]) > 0 else 0.0
            )
            
            # Create prosody features schema
            prosody_schema = ProsodyFeaturesSchema(
                speech_rate=prosodic_features.syllable_timing.get('syllable_rate', 0.0),
                pause_frequency=prosodic_features.pause_patterns.get('pause_frequency', 0.0),
                emphasis_variance=prosodic_features.rhythm_patterns.get('rhythm_regularity', 0.0),
                pitch_range_semitones=pitch_features.pitch_range_semitones,
                pitch_contour_complexity=pitch_features.pitch_trajectory_complexity,
                pitch_contour=pitch_features.pitch_contour.tolist() if len(pitch_features.pitch_contour) > 0 else [],
                energy_contour=[],  # Will be filled by legacy method if needed
                syllable_duration_mean=prosodic_features.syllable_timing.get('mean_syllable_duration', 0.0),
                syllable_duration_std=prosodic_features.syllable_timing.get('syllable_duration_variance', 0.0),
                pause_duration_mean=prosodic_features.pause_patterns.get('mean_pause_duration', 0.0),
                pause_duration_std=prosodic_features.pause_patterns.get('pause_duration_variance', 0.0),
                stress_pattern_entropy=0.0,  # Could be calculated from stress patterns
                primary_stress_ratio=np.mean(prosodic_features.stress_patterns) if len(prosodic_features.stress_patterns) > 0 else 0.0,
                declination_slope=np.mean(prosodic_features.declination_patterns) if len(prosodic_features.declination_patterns) > 0 else 0.0,
                excitement_score=emotional_features.emotional_dimensions.get('arousal', 0.0),
                calmness_score=1.0 - emotional_features.emotional_dimensions.get('arousal', 0.0),
                confidence_score=emotional_features.confidence_measures.get('overall_confidence', 0.0),
                timing_features=prosodic_features.syllable_timing
            )
            
            # Create emotional profile schema
            emotional_schema = EmotionalProfileSchema(
                valence=emotional_features.emotional_dimensions.get('valence', 0.0),
                arousal=emotional_features.emotional_dimensions.get('arousal', 0.0),
                dominance=emotional_features.emotional_dimensions.get('dominance', 0.0),
                happiness_score=emotional_features.emotional_dimensions.get('happiness', 0.0),
                sadness_score=emotional_features.emotional_dimensions.get('sadness', 0.0),
                anger_score=emotional_features.emotional_dimensions.get('anger', 0.0),
                fear_score=emotional_features.emotional_dimensions.get('fear', 0.0),
                surprise_score=emotional_features.emotional_dimensions.get('surprise', 0.0),
                disgust_score=emotional_features.emotional_dimensions.get('disgust', 0.0),
                breathiness=emotional_features.voice_quality_measures.get('breathiness', 0.0),
                roughness=emotional_features.voice_quality_measures.get('roughness', 0.0),
                strain=emotional_features.voice_quality_measures.get('strain', 0.0),
                analysis_reliability=0.95  # High reliability for multi-dimensional analysis
            )
            
            # Create voice profile with comprehensive features
            voice_profile = VoiceProfileSchema(
                id="temp_id",
                reference_audio_id="temp_ref_id",
                fundamental_frequency=fundamental_freq,
                formant_frequencies=np.mean(formant_features.formant_frequencies, axis=0).tolist() if len(formant_features.formant_frequencies) > 0 else [],
                spectral_centroid_mean=float(np.mean(timbre_features.spectral_centroid)),
                spectral_rolloff_mean=float(np.mean(timbre_features.spectral_rolloff)),
                spectral_bandwidth_mean=0.0,  # Not directly available, could be calculated
                zero_crossing_rate_mean=0.0,  # Not directly available, could be calculated
                mfcc_features=timbre_features.mfcc_coefficients.tolist(),
                speech_rate=prosody_schema.speech_rate,
                pause_frequency=prosody_schema.pause_frequency,
                emphasis_variance=prosody_schema.emphasis_variance,
                energy_mean=0.0,  # Could be calculated from timbre features
                energy_variance=0.0,  # Could be calculated from timbre features
                pitch_variance=fundamental_freq.std_hz,
                signal_to_noise_ratio=quality_metrics.signal_to_noise_ratio,
                voice_activity_ratio=quality_metrics.voice_activity_ratio,
                quality_score=quality_metrics.overall_quality,
                analysis_duration=comprehensive_analysis['audio_metadata']['duration'],
                sample_rate=comprehensive_analysis['audio_metadata']['sample_rate'],
                total_frames=comprehensive_analysis['audio_metadata']['total_samples'],
                created_at=datetime.now()
            )
            
            # Create voice characteristics
            voice_characteristics = VoiceCharacteristics(
                timbre_features={
                    'brightness': timbre_features.brightness_measure,
                    'warmth': timbre_features.warmth_measure,
                    'breathiness': timbre_features.breathiness_measure,
                    'roughness': timbre_features.roughness_measure,
                    'nasality': timbre_features.nasality_measure,
                    'spectral_centroid': float(np.mean(timbre_features.spectral_centroid)),
                    'spectral_rolloff': float(np.mean(timbre_features.spectral_rolloff)),
                    **timbre_features.resonance_characteristics
                },
                pitch_characteristics=fundamental_freq,
                prosody_features=prosody_schema,
                emotional_markers=emotional_schema,
                quality_metrics=quality_metrics
            )
            
            processing_time = time.time() - start_time
            
            # Enhanced analysis metadata including fingerprint info
            analysis_metadata = {
                "audio_duration": comprehensive_analysis['audio_metadata']['duration'],
                "sample_rate": comprehensive_analysis['audio_metadata']['sample_rate'],
                "total_frames": comprehensive_analysis['audio_metadata']['total_samples'],
                "hop_length": self.multi_dimensional_analyzer.hop_length,
                "n_fft": self.multi_dimensional_analyzer.n_fft,
                "analysis_version": "2.0_multidimensional",
                "voice_fingerprint_features": voice_fingerprint.get('_total_features', 0),
                "fingerprint_version": voice_fingerprint.get('_fingerprint_version', '1.0'),
                "advanced_features": {
                    "pitch_jitter": pitch_features.jitter,
                    "pitch_shimmer": pitch_features.shimmer,
                    "harmonics_to_noise_ratio": pitch_features.harmonics_to_noise_ratio,
                    "vowel_space_area": formant_features.vowel_space_area,
                    "formant_dispersion": formant_features.formant_dispersion,
                    "personality_extraversion": emotional_features.personality_indicators.get('extraversion', 0.0),
                    "personality_neuroticism": emotional_features.personality_indicators.get('neuroticism', 0.0),
                    "speaking_style_confidence": emotional_features.speaking_style_markers.get('confidence', 0.0),
                    "speaking_style_formality": emotional_features.speaking_style_markers.get('formality', 0.0)
                }
            }
            
            # End performance monitoring
            performance_monitor.end_operation(
                operation_id, 
                success=True,
                additional_metadata={
                    'audio_duration': comprehensive_analysis['audio_metadata']['duration'],
                    'processing_time': processing_time,
                    'quality_score': quality_metrics.overall_quality,
                    'fingerprint_features': voice_fingerprint.get('_total_features', 0)
                }
            )
            
            return VoiceAnalysisResult(
                voice_profile=voice_profile,
                voice_characteristics=voice_characteristics,
                prosody_features=prosody_schema,
                emotional_profile=emotional_schema,
                processing_time=processing_time,
                analysis_metadata=analysis_metadata
            )
            
        except Exception as e:
            # End performance monitoring with error
            performance_monitor.end_operation(
                operation_id, 
                success=False,
                error_message=str(e)
            )
            logger.error(f"Multi-dimensional voice analysis failed for {audio_path}: {str(e)}")
            raise
        
    def analyze_voice_characteristics(self, audio_path: str) -> VoiceAnalysisResult:
        """
        Extract comprehensive voice characteristics from audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            VoiceAnalysisResult containing all extracted features
        """
        start_time = time.time()
        operation_id = f"voice_analysis_{int(time.time() * 1000)}"
        
        # Start performance monitoring
        performance_monitor.start_operation(
            'voice_analysis', 
            operation_id,
            {'audio_path': audio_path}
        )
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(y) == 0:
                raise ValueError("Audio file is empty or corrupted")
            
            # Extract all voice characteristics
            fundamental_freq = self._extract_fundamental_frequency(y, sr)
            formants = self._extract_formant_frequencies(y, sr)
            spectral_features = self._extract_spectral_features(y, sr)
            mfcc_features = self._extract_mfcc_features(y, sr)
            prosody_features = self._extract_prosody_features(y, sr)
            emotional_profile = self._extract_emotional_markers(y, sr)
            quality_metrics = self._assess_voice_quality(y, sr)
            
            # Create voice profile
            voice_profile = self._create_voice_profile(
                audio_path, y, sr, fundamental_freq, formants, 
                spectral_features, mfcc_features, prosody_features,
                emotional_profile, quality_metrics
            )
            
            # Combine characteristics
            voice_characteristics = VoiceCharacteristics(
                timbre_features=self._extract_timbre_features(spectral_features, mfcc_features),
                pitch_characteristics=fundamental_freq,
                prosody_features=prosody_features,
                emotional_markers=emotional_profile,
                quality_metrics=quality_metrics
            )
            
            processing_time = time.time() - start_time
            
            analysis_metadata = {
                "audio_duration": len(y) / sr,
                "sample_rate": sr,
                "total_frames": len(y),
                "hop_length": self.hop_length,
                "n_fft": self.n_fft,
                "analysis_version": "1.0"
            }
            
            # End performance monitoring
            performance_monitor.end_operation(
                operation_id, 
                success=True,
                additional_metadata={
                    'audio_duration': len(y) / sr,
                    'processing_time': processing_time,
                    'quality_score': voice_characteristics.quality_metrics.overall_quality
                }
            )
            
            return VoiceAnalysisResult(
                voice_profile=voice_profile,
                voice_characteristics=voice_characteristics,
                prosody_features=prosody_features,
                emotional_profile=emotional_profile,
                processing_time=processing_time,
                analysis_metadata=analysis_metadata
            )
            
        except Exception as e:
            # End performance monitoring with error
            performance_monitor.end_operation(
                operation_id, 
                success=False,
                error_message=str(e)
            )
            logger.error(f"Voice analysis failed for {audio_path}: {str(e)}")
            raise
    
    def _extract_fundamental_frequency(self, y: np.ndarray, sr: int) -> FrequencyRange:
        """Extract fundamental frequency characteristics."""
        # Use librosa's pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, hop_length=self.hop_length
        )
        
        # Filter out unvoiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) == 0:
            return FrequencyRange(min_hz=0, max_hz=0, mean_hz=0, std_hz=0)
        
        return FrequencyRange(
            min_hz=float(np.min(f0_voiced)),
            max_hz=float(np.max(f0_voiced)),
            mean_hz=float(np.mean(f0_voiced)),
            std_hz=float(np.std(f0_voiced))
        )
    
    def _extract_formant_frequencies(self, y: np.ndarray, sr: int) -> List[float]:
        """Extract formant frequencies using LPC analysis."""
        try:
            # Pre-emphasis filter
            pre_emphasis = 0.97
            y_preemph = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            # Window the signal
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.01 * sr)     # 10ms hop
            
            formants = []
            
            for i in range(0, len(y_preemph) - frame_length, hop_length):
                frame = y_preemph[i:i + frame_length]
                
                # Apply window
                windowed = frame * np.hanning(len(frame))
                
                # LPC analysis
                lpc_order = 12
                lpc_coeffs = librosa.lpc(windowed, order=lpc_order)
                
                # Find roots and convert to formants
                roots = np.roots(lpc_coeffs)
                roots = roots[np.imag(roots) >= 0]
                
                # Convert to frequencies
                freqs = np.angle(roots) * sr / (2 * np.pi)
                freqs = freqs[freqs > 0]
                freqs = np.sort(freqs)
                
                if len(freqs) >= self.n_formants:
                    formants.append(freqs[:self.n_formants])
            
            if formants:
                # Average formants across frames
                formants_array = np.array(formants)
                avg_formants = np.mean(formants_array, axis=0)
                return avg_formants.tolist()
            else:
                return [500.0, 1500.0, 2500.0, 3500.0]  # Default formant values
                
        except Exception as e:
            logger.warning(f"Formant extraction failed: {e}")
            return [500.0, 1500.0, 2500.0, 3500.0]  # Default values
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral characteristics."""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        
        return {
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
            "zero_crossing_rate_mean": float(np.mean(zcr))
        }
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> List[List[float]]:
        """Extract MFCC features."""
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )
        return mfccs.tolist()
    
    def _extract_prosody_features(self, y: np.ndarray, sr: int) -> ProsodyFeaturesSchema:
        """Extract prosodic features including rhythm, stress, and intonation."""
        # Extract pitch contour
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
            sr=sr, hop_length=self.hop_length
        )
        
        # Extract energy contour
        rms_energy = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Estimate speech rate (simplified)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
        speech_rate = len(onset_frames) / (len(y) / sr) if len(y) > 0 else 0
        
        # Detect pauses (simplified)
        energy_threshold = np.percentile(rms_energy, 20)
        pause_frames = rms_energy < energy_threshold
        pause_segments = self._find_segments(pause_frames)
        pause_frequency = len(pause_segments) / (len(y) / sr / 60) if len(y) > 0 else 0
        
        # Calculate pitch range in semitones
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) > 0:
            pitch_range_semitones = 12 * np.log2(np.max(f0_voiced) / np.min(f0_voiced))
        else:
            pitch_range_semitones = 0
        
        # Pitch contour complexity (entropy of pitch changes)
        if len(f0_voiced) > 1:
            pitch_diff = np.diff(f0_voiced)
            pitch_diff_normalized = (pitch_diff - np.mean(pitch_diff)) / (np.std(pitch_diff) + 1e-8)
            pitch_contour_complexity = entropy(np.histogram(pitch_diff_normalized, bins=10)[0] + 1e-8)
        else:
            pitch_contour_complexity = 0
        
        return ProsodyFeaturesSchema(
            speech_rate=speech_rate,
            pause_frequency=pause_frequency,
            pitch_range_semitones=float(pitch_range_semitones),
            pitch_contour_complexity=float(pitch_contour_complexity),
            pitch_contour=f0.tolist() if len(f0) > 0 else [],
            energy_contour=rms_energy.tolist(),
            emphasis_variance=float(np.var(rms_energy))
        )
    
    def _extract_emotional_markers(self, y: np.ndarray, sr: int) -> EmotionalProfileSchema:
        """Extract emotional characteristics from voice."""
        # Extract features for emotional analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        # Simplified emotional mapping based on acoustic features
        # These are heuristic mappings - in production, you'd use trained models
        
        # Arousal (energy-based)
        arousal = float(np.tanh(np.mean(rms) * 10 - 1))  # Normalize to [-1, 1]
        
        # Valence (spectral-based)
        valence = float(np.tanh((np.mean(spectral_centroid) - 2000) / 1000))
        
        # Dominance (pitch and energy based)
        f0, _, _ = librosa.pyin(y, fmin=80, fmax=400, sr=sr)
        f0_mean = np.nanmean(f0) if not np.all(np.isnan(f0)) else 150
        dominance = float(np.tanh((f0_mean - 150) / 50 + np.mean(rms) * 5 - 1))
        
        # Basic emotion scores (simplified heuristics)
        happiness_score = max(0, min(1, (arousal + valence) / 2 + 0.5))
        sadness_score = max(0, min(1, (-arousal - valence) / 2 + 0.5))
        anger_score = max(0, min(1, (arousal - valence) / 2 + 0.5))
        
        # Voice quality indicators
        breathiness = float(np.mean(zcr))  # Higher ZCR can indicate breathiness
        roughness = float(np.std(rms))     # Energy variability
        
        return EmotionalProfileSchema(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            happiness_score=happiness_score,
            sadness_score=sadness_score,
            anger_score=anger_score,
            fear_score=max(0, min(1, arousal - dominance + 0.5)),
            surprise_score=max(0, min(1, arousal * 0.8)),
            disgust_score=max(0, min(1, -valence * 0.6 + 0.3)),
            breathiness=min(1, breathiness * 10),
            roughness=min(1, roughness * 5),
            strain=max(0, min(1, (np.mean(spectral_rolloff) - 3000) / 2000)),
            analysis_reliability=0.7  # Placeholder - would be model confidence in production
        )
    
    def _assess_voice_quality(self, y: np.ndarray, sr: int) -> QualityMetrics:
        """Assess the quality of the voice recording."""
        # Signal-to-noise ratio estimation
        # Simple approach: compare energy in voiced vs unvoiced segments
        rms_energy = librosa.feature.rms(y=y)[0]
        energy_threshold = np.percentile(rms_energy, 30)
        
        voiced_energy = rms_energy[rms_energy > energy_threshold]
        noise_energy = rms_energy[rms_energy <= energy_threshold]
        
        if len(noise_energy) > 0 and len(voiced_energy) > 0:
            snr = 20 * np.log10(np.mean(voiced_energy) / (np.mean(noise_energy) + 1e-8))
        else:
            snr = 20.0  # Default reasonable SNR
        
        # Voice activity ratio
        voice_activity_ratio = len(voiced_energy) / len(rms_energy) if len(rms_energy) > 0 else 0
        
        # Overall quality score (heuristic combination)
        quality_factors = [
            min(1, max(0, (snr - 10) / 20)),  # SNR contribution
            voice_activity_ratio,              # Voice activity contribution
            min(1, len(y) / sr / 3)           # Duration contribution (3s+ is good)
        ]
        
        overall_quality = np.mean(quality_factors)
        
        return QualityMetrics(
            signal_to_noise_ratio=float(snr),
            overall_quality=float(overall_quality),
            voice_activity_ratio=float(voice_activity_ratio),
            spectral_similarity=0.8,  # Placeholder
            prosody_accuracy=0.8,     # Placeholder
            frequency_response_quality=0.8,  # Placeholder
            temporal_consistency=0.8  # Placeholder
        )
    
    def _extract_timbre_features(self, spectral_features: Dict[str, float], 
                               mfcc_features: List[List[float]]) -> Dict[str, float]:
        """Extract timbre characteristics."""
        mfcc_array = np.array(mfcc_features)
        
        return {
            "spectral_centroid": spectral_features["spectral_centroid_mean"],
            "spectral_rolloff": spectral_features["spectral_rolloff_mean"],
            "spectral_bandwidth": spectral_features["spectral_bandwidth_mean"],
            "mfcc_1_mean": float(np.mean(mfcc_array[1])) if mfcc_array.shape[0] > 1 else 0,
            "mfcc_2_mean": float(np.mean(mfcc_array[2])) if mfcc_array.shape[0] > 2 else 0,
            "mfcc_3_mean": float(np.mean(mfcc_array[3])) if mfcc_array.shape[0] > 3 else 0,
            "brightness": spectral_features["spectral_centroid_mean"] / 1000,  # Normalized
            "warmth": 1.0 - (spectral_features["spectral_centroid_mean"] / 4000)  # Inverse of brightness
        }
    
    def _create_voice_profile(self, audio_path: str, y: np.ndarray, sr: int,
                            fundamental_freq: FrequencyRange, formants: List[float],
                            spectral_features: Dict[str, float], mfcc_features: List[List[float]],
                            prosody_features: ProsodyFeaturesSchema,
                            emotional_profile: EmotionalProfileSchema,
                            quality_metrics: QualityMetrics) -> VoiceProfileSchema:
        """Create a voice profile schema from extracted features."""
        
        return VoiceProfileSchema(
            id="temp_id",  # Will be set by the database
            reference_audio_id="temp_ref_id",  # Will be set by caller
            fundamental_frequency=fundamental_freq,
            formant_frequencies=formants,
            spectral_centroid_mean=spectral_features["spectral_centroid_mean"],
            spectral_rolloff_mean=spectral_features["spectral_rolloff_mean"],
            spectral_bandwidth_mean=spectral_features["spectral_bandwidth_mean"],
            zero_crossing_rate_mean=spectral_features["zero_crossing_rate_mean"],
            mfcc_features=mfcc_features,
            speech_rate=prosody_features.speech_rate,
            pause_frequency=prosody_features.pause_frequency,
            emphasis_variance=prosody_features.emphasis_variance,
            energy_mean=float(np.mean(librosa.feature.rms(y=y)[0])),
            energy_variance=float(np.var(librosa.feature.rms(y=y)[0])),
            pitch_variance=fundamental_freq.std_hz,
            signal_to_noise_ratio=quality_metrics.signal_to_noise_ratio,
            voice_activity_ratio=quality_metrics.voice_activity_ratio,
            quality_score=quality_metrics.overall_quality,
            analysis_duration=len(y) / sr,
            sample_rate=sr,
            total_frames=len(y),
            created_at=datetime.now()
        )
    
    def _find_segments(self, boolean_array: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments where boolean_array is True."""
        segments = []
        start = None
        
        for i, val in enumerate(boolean_array):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(boolean_array)))
        
        return segments
    
    def create_voice_model(self, voice_profile: VoiceProfileSchema) -> Dict[str, Any]:
        """
        Create a voice model from analyzed characteristics.
        This is a placeholder for TorToiSe TTS integration.
        """
        # In a real implementation, this would:
        # 1. Load the TorToiSe TTS model
        # 2. Fine-tune or adapt it using the voice characteristics
        # 3. Save the adapted model
        # 4. Return model metadata
        
        model_characteristics = {
            "fundamental_frequency_range": {
                "min": voice_profile.fundamental_frequency.min_hz if voice_profile.fundamental_frequency else 80,
                "max": voice_profile.fundamental_frequency.max_hz if voice_profile.fundamental_frequency else 300,
                "mean": voice_profile.fundamental_frequency.mean_hz if voice_profile.fundamental_frequency else 150
            },
            "formant_frequencies": voice_profile.formant_frequencies or [500, 1500, 2500, 3500],
            "spectral_characteristics": {
                "centroid": voice_profile.spectral_centroid_mean or 2000,
                "rolloff": voice_profile.spectral_rolloff_mean or 4000,
                "bandwidth": voice_profile.spectral_bandwidth_mean or 1000
            },
            "prosody_parameters": {
                "speech_rate": voice_profile.speech_rate or 4.0,
                "pause_frequency": voice_profile.pause_frequency or 10.0,
                "emphasis_variance": voice_profile.emphasis_variance or 0.1
            },
            "quality_score": voice_profile.quality_score or 0.8
        }
        
        return {
            "model_type": "tortoise_tts",
            "model_version": "1.0",
            "characteristics": model_characteristics,
            "quality_score": voice_profile.quality_score or 0.8,
            "training_duration": 0.0,  # Placeholder
            "model_size_mb": 50.0,     # Placeholder
            "inference_time_ms": 1000.0  # Placeholder
        }


# Import datetime at the top of the file
from datetime import datetime