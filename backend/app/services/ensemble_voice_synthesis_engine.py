"""
Ensemble Voice Synthesis Engine for high-fidelity voice cloning.

This module implements an advanced ensemble synthesis system that integrates
multiple TTS models (XTTS, StyleTTS2, OpenVoice, RVC) for maximum voice 
replication accuracy with parallel synthesis and intelligent fusion.

Requirements: 6.1, 6.2, 6.3
"""

import asyncio
import logging
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import tempfile
import shutil
import concurrent.futures
from threading import Lock

# Audio processing libraries
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt

# TTS model imports
import torch
import torchaudio
from TTS.api import TTS

# Internal imports
from app.core.config import settings
from app.schemas.voice import VoiceProfileSchema, VoiceModelSchema
from app.services.multi_dimensional_voice_analyzer import MultiDimensionalVoiceAnalyzer
from app.services.advanced_audio_preprocessing import AdvancedAudioPreprocessor

logger = logging.getLogger(__name__)


class TTSModelType(Enum):
    """Supported TTS model types for ensemble synthesis."""
    XTTS_V2 = "xtts_v2"
    STYLETTS2 = "styletts2"
    OPENVOICE = "openvoice"
    BARK = "bark"
    YOUR_TTS = "your_tts"


@dataclass
class ModelConfiguration:
    """Configuration for individual TTS models."""
    model_type: TTSModelType
    model_name: str
    quality_score: float
    supports_cloning: bool
    languages: List[str]
    max_text_length: int
    processing_time_factor: float  # Relative processing time
    memory_usage: str  # "low", "medium", "high"


@dataclass
class SynthesisResult:
    """Result from individual model synthesis."""
    model_type: TTSModelType
    audio_data: Optional[np.ndarray]
    sample_rate: int
    quality_score: float
    confidence_score: float
    processing_time: float
    similarity_score: Optional[float] = None
    error_message: Optional[str] = None
    segment_scores: Dict[str, float] = field(default_factory=dict)
    per_segment_quality: List['SegmentQualityScore'] = field(default_factory=list)


@dataclass
class SegmentQualityScore:
    """Quality score for a specific audio segment."""
    segment_index: int
    start_time: float
    end_time: float
    similarity_score: float
    naturalness_score: float
    clarity_score: float
    prosody_score: float
    overall_score: float
    model_type: TTSModelType


@dataclass
class EnsembleWeights:
    """Weights for ensemble model combination."""
    model_weights: Dict[TTSModelType, float]
    quality_threshold: float
    confidence_threshold: float
    similarity_threshold: float


@dataclass
class FusionConfig:
    """Configuration for intelligent fusion algorithm."""
    segment_duration_ms: float = 500.0  # Duration of each segment for scoring
    crossfade_duration_ms: float = 50.0  # Duration of crossfade between segments
    min_quality_threshold: float = 0.6  # Minimum quality to include in fusion
    use_weighted_combination: bool = True  # Use weighted vs best-selection
    fallback_to_best: bool = True  # Fallback to best single model if fusion fails


class ModelSelector:
    """Intelligent model selection based on voice characteristics."""
    
    def __init__(self):
        self.voice_analyzer = MultiDimensionalVoiceAnalyzer()
        
        # Model suitability profiles
        self.model_profiles = {
            TTSModelType.XTTS_V2: {
                "best_for": ["multilingual", "high_quality", "fast_inference"],
                "pitch_range": (80, 400),
                "formant_sensitivity": 0.8,
                "prosody_preservation": 0.9,
                "emotion_capture": 0.7
            },
            TTSModelType.STYLETTS2: {
                "best_for": ["expressive_speech", "prosody_transfer", "human_level"],
                "pitch_range": (70, 450),
                "formant_sensitivity": 0.9,
                "prosody_preservation": 0.95,
                "emotion_capture": 0.9
            },
            TTSModelType.OPENVOICE: {
                "best_for": ["instant_cloning", "style_control", "cross_lingual"],
                "pitch_range": (80, 400),
                "formant_sensitivity": 0.85,
                "prosody_preservation": 0.85,
                "emotion_capture": 0.8
            },
            TTSModelType.BARK: {
                "best_for": ["emotional_speech", "natural_prosody", "code_switching"],
                "pitch_range": (60, 500),
                "formant_sensitivity": 0.7,
                "prosody_preservation": 0.95,
                "emotion_capture": 0.9
            },
            TTSModelType.YOUR_TTS: {
                "best_for": ["few_shot_cloning", "speaker_adaptation", "cross_lingual"],
                "pitch_range": (90, 350),
                "formant_sensitivity": 0.85,
                "prosody_preservation": 0.75,
                "emotion_capture": 0.6
            }
        }
    
    def select_optimal_models(self, voice_profile: VoiceProfileSchema, 
                            text: str, language: str) -> List[TTSModelType]:
        """Select optimal models based on voice characteristics and requirements."""
        try:
            # Analyze voice characteristics from individual fields
            characteristics = {}
            
            # Extract fundamental frequency characteristics
            if voice_profile.fundamental_frequency:
                characteristics["fundamental_frequency_range"] = {
                    "mean": voice_profile.fundamental_frequency.mean_hz,
                    "min": voice_profile.fundamental_frequency.min_hz,
                    "max": voice_profile.fundamental_frequency.max_hz
                }
            
            # Extract formant characteristics
            if voice_profile.formant_frequencies:
                characteristics["formant_frequencies"] = voice_profile.formant_frequencies
            
            # Extract prosody characteristics
            prosody_complexity = 0.5  # Default
            if voice_profile.speech_rate and voice_profile.pause_frequency:
                prosody_complexity = min(1.0, (voice_profile.speech_rate / 10.0 + voice_profile.pause_frequency) / 2)
            characteristics["prosody_parameters"] = {"complexity": prosody_complexity}
            
            # Extract emotional characteristics
            emotional_intensity = 0.5  # Default
            if voice_profile.energy_variance and voice_profile.pitch_variance:
                emotional_intensity = min(1.0, (voice_profile.energy_variance + voice_profile.pitch_variance) / 2)
            characteristics["emotional_profile"] = {"intensity": emotional_intensity}
            
            # Extract key features for model selection
            pitch_mean = characteristics.get("fundamental_frequency_range", {}).get("mean", 150)
            formant_features = characteristics.get("formant_frequencies", [])
            prosody_complexity = characteristics.get("prosody_parameters", {}).get("complexity", 0.5)
            emotional_intensity = characteristics.get("emotional_profile", {}).get("intensity", 0.5)
            
            # Score each model
            model_scores = {}
            
            for model_type, profile in self.model_profiles.items():
                score = 0.0
                
                # Pitch range compatibility
                pitch_min, pitch_max = profile["pitch_range"]
                if pitch_min <= pitch_mean <= pitch_max:
                    score += 0.3
                else:
                    # Penalty for out-of-range pitch
                    score -= 0.2
                
                # Formant sensitivity match
                if formant_features:
                    formant_complexity = len(formant_features) / 5.0  # Normalize
                    formant_match = 1.0 - abs(profile["formant_sensitivity"] - formant_complexity)
                    score += 0.25 * formant_match
                
                # Prosody preservation match
                prosody_match = 1.0 - abs(profile["prosody_preservation"] - prosody_complexity)
                score += 0.25 * prosody_match
                
                # Emotion capture match
                emotion_match = 1.0 - abs(profile["emotion_capture"] - emotional_intensity)
                score += 0.2 * emotion_match
                
                model_scores[model_type] = score
            
            # Sort models by score and select top performers
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top 3 models for ensemble
            selected_models = [model for model, score in sorted_models[:3] if score > 0.3]
            
            # Ensure at least one model is selected
            if not selected_models:
                selected_models = [TTSModelType.XTTS_V2]  # Default fallback
            
            logger.info(f"Selected models for synthesis: {[m.value for m in selected_models]}")
            return selected_models
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            return [TTSModelType.XTTS_V2]  # Safe fallback


class QualityAssessment:
    """Real-time quality assessment and model switching."""
    
    def __init__(self):
        self.quality_thresholds = {
            "similarity": 0.85,
            "naturalness": 0.8,
            "clarity": 0.75,
            "prosody": 0.8
        }
    
    def assess_synthesis_quality(self, synthesized_audio: np.ndarray, 
                               reference_audio: np.ndarray,
                               sample_rate: int) -> Dict[str, float]:
        """Assess quality of synthesized audio against reference."""
        try:
            quality_metrics = {}
            
            # Similarity assessment using MFCC comparison
            similarity_score = self._calculate_mfcc_similarity(
                synthesized_audio, reference_audio, sample_rate
            )
            quality_metrics["similarity"] = similarity_score
            
            # Naturalness assessment
            naturalness_score = self._assess_naturalness(synthesized_audio, sample_rate)
            quality_metrics["naturalness"] = naturalness_score
            
            # Clarity assessment
            clarity_score = self._assess_clarity(synthesized_audio, sample_rate)
            quality_metrics["clarity"] = clarity_score
            
            # Prosody assessment
            prosody_score = self._assess_prosody_preservation(
                synthesized_audio, reference_audio, sample_rate
            )
            quality_metrics["prosody"] = prosody_score
            
            # Overall quality score
            quality_metrics["overall"] = np.mean(list(quality_metrics.values()))
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"similarity": 0.5, "naturalness": 0.5, "clarity": 0.5, 
                   "prosody": 0.5, "overall": 0.5}
    
    def assess_segment_quality(
        self,
        segment: np.ndarray,
        reference_segment: np.ndarray,
        sample_rate: int,
        segment_index: int,
        start_time: float,
        end_time: float,
        model_type: TTSModelType
    ) -> SegmentQualityScore:
        """Assess quality of a specific audio segment."""
        try:
            metrics = self.assess_synthesis_quality(segment, reference_segment, sample_rate)
            
            return SegmentQualityScore(
                segment_index=segment_index,
                start_time=start_time,
                end_time=end_time,
                similarity_score=metrics["similarity"],
                naturalness_score=metrics["naturalness"],
                clarity_score=metrics["clarity"],
                prosody_score=metrics["prosody"],
                overall_score=metrics["overall"],
                model_type=model_type
            )
        except Exception as e:
            logger.error(f"Segment quality assessment failed: {e}")
            return SegmentQualityScore(
                segment_index=segment_index,
                start_time=start_time,
                end_time=end_time,
                similarity_score=0.5,
                naturalness_score=0.5,
                clarity_score=0.5,
                prosody_score=0.5,
                overall_score=0.5,
                model_type=model_type
            )
    
    def _calculate_mfcc_similarity(self, audio1: np.ndarray, audio2: np.ndarray, 
                                 sample_rate: int) -> float:
        """Calculate MFCC-based similarity between two audio signals."""
        try:
            # Extract MFCCs
            mfcc1 = librosa.feature.mfcc(y=audio1, sr=sample_rate, n_mfcc=13)
            mfcc2 = librosa.feature.mfcc(y=audio2, sr=sample_rate, n_mfcc=13)
            
            # Calculate mean MFCCs
            mfcc1_mean = np.mean(mfcc1, axis=1)
            mfcc2_mean = np.mean(mfcc2, axis=1)
            
            # Cosine similarity
            similarity = np.dot(mfcc1_mean, mfcc2_mean) / (
                np.linalg.norm(mfcc1_mean) * np.linalg.norm(mfcc2_mean) + 1e-8
            )
            
            return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to [0, 1]
            
        except Exception as e:
            logger.error(f"MFCC similarity calculation failed: {e}")
            return 0.5
    
    def _assess_naturalness(self, audio: np.ndarray, sample_rate: int) -> float:
        """Assess naturalness of synthesized speech."""
        try:
            # Spectral flatness (lower is more natural for speech)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
            naturalness_spectral = 1.0 - min(1.0, spectral_flatness * 10)
            
            # Zero crossing rate (should be in natural range for speech)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            naturalness_zcr = 1.0 - min(1.0, abs(zcr - 0.1) * 10)
            
            # Combine metrics
            naturalness = (naturalness_spectral + naturalness_zcr) / 2
            return max(0.0, min(1.0, naturalness))
            
        except Exception as e:
            logger.error(f"Naturalness assessment failed: {e}")
            return 0.5
    
    def _assess_clarity(self, audio: np.ndarray, sample_rate: int) -> float:
        """Assess clarity of synthesized speech."""
        try:
            # Spectral centroid (higher indicates clearer speech)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
            clarity_spectral = min(1.0, spectral_centroid / 3000)
            
            # RMS energy (adequate energy indicates clarity)
            rms_energy = np.mean(librosa.feature.rms(y=audio))
            clarity_energy = min(1.0, rms_energy * 10)
            
            # Combine metrics
            clarity = (clarity_spectral + clarity_energy) / 2
            return max(0.0, min(1.0, clarity))
            
        except Exception as e:
            logger.error(f"Clarity assessment failed: {e}")
            return 0.5
    
    def _assess_prosody_preservation(self, synthesized_audio: np.ndarray, 
                                   reference_audio: np.ndarray, sample_rate: int) -> float:
        """Assess prosody preservation in synthesized speech."""
        try:
            # Extract pitch contours
            f0_synth, _, _ = librosa.pyin(synthesized_audio, fmin=80, fmax=400, sr=sample_rate)
            f0_ref, _, _ = librosa.pyin(reference_audio, fmin=80, fmax=400, sr=sample_rate)
            
            # Remove NaN values
            f0_synth_clean = f0_synth[~np.isnan(f0_synth)]
            f0_ref_clean = f0_ref[~np.isnan(f0_ref)]
            
            if len(f0_synth_clean) == 0 or len(f0_ref_clean) == 0:
                return 0.5
            
            # Compare pitch statistics
            pitch_mean_diff = abs(np.mean(f0_synth_clean) - np.mean(f0_ref_clean))
            pitch_std_diff = abs(np.std(f0_synth_clean) - np.std(f0_ref_clean))
            
            # Normalize differences
            prosody_score = 1.0 - min(1.0, (pitch_mean_diff / 100 + pitch_std_diff / 50) / 2)
            
            return max(0.0, min(1.0, prosody_score))
            
        except Exception as e:
            logger.error(f"Prosody assessment failed: {e}")
            return 0.5


class CrossLanguageAdapter:
    """Cross-language phonetic adaptation system."""
    
    def __init__(self):
        # Phoneme mapping tables for cross-language adaptation
        self.phoneme_mappings = {
            ("en", "es"): {
                "θ": "s",  # th -> s
                "ð": "d",  # th -> d
                "ʃ": "ʧ",  # sh -> ch
                "ʒ": "ʤ",  # zh -> j
            },
            ("en", "fr"): {
                "θ": "s",
                "ð": "z",
                "w": "v",
                "r": "ʁ",  # English r -> French r
            },
            ("en", "de"): {
                "θ": "s",
                "ð": "z",
                "w": "v",
                "ʤ": "ʧ",  # j -> ch
            }
        }
        
        # Language-specific voice adaptations
        self.language_adaptations = {
            "es": {"pitch_shift": 0.02, "formant_shift": [1.02, 0.98, 1.01]},
            "fr": {"pitch_shift": -0.01, "formant_shift": [0.98, 1.03, 0.99]},
            "de": {"pitch_shift": -0.03, "formant_shift": [1.01, 0.97, 1.02]},
            "it": {"pitch_shift": 0.03, "formant_shift": [1.03, 1.01, 0.98]},
            "pt": {"pitch_shift": 0.01, "formant_shift": [1.01, 0.99, 1.01]}
        }
    
    def adapt_voice_for_language(self, voice_profile: VoiceProfileSchema, 
                               target_language: str) -> VoiceProfileSchema:
        """Adapt voice characteristics for target language."""
        try:
            if target_language not in self.language_adaptations:
                return voice_profile  # No adaptation needed
            
            # Get adaptation parameters
            adaptation = self.language_adaptations[target_language]
            
            # Create adapted voice profile with modified characteristics
            adapted_profile = VoiceProfileSchema(
                id=voice_profile.id + f"_adapted_{target_language}",
                reference_audio_id=voice_profile.reference_audio_id,
                fundamental_frequency=voice_profile.fundamental_frequency,
                formant_frequencies=voice_profile.formant_frequencies,
                spectral_centroid_mean=voice_profile.spectral_centroid_mean,
                spectral_rolloff_mean=voice_profile.spectral_rolloff_mean,
                spectral_bandwidth_mean=voice_profile.spectral_bandwidth_mean,
                zero_crossing_rate_mean=voice_profile.zero_crossing_rate_mean,
                mfcc_features=voice_profile.mfcc_features,
                speech_rate=voice_profile.speech_rate,
                pause_frequency=voice_profile.pause_frequency,
                emphasis_variance=voice_profile.emphasis_variance,
                energy_mean=voice_profile.energy_mean,
                energy_variance=voice_profile.energy_variance,
                pitch_variance=voice_profile.pitch_variance,
                signal_to_noise_ratio=voice_profile.signal_to_noise_ratio,
                voice_activity_ratio=voice_profile.voice_activity_ratio,
                quality_score=voice_profile.quality_score,
                analysis_duration=voice_profile.analysis_duration,
                sample_rate=voice_profile.sample_rate,
                total_frames=voice_profile.total_frames,
                created_at=voice_profile.created_at
            )
            
            # Apply pitch adaptation if fundamental frequency exists
            if adapted_profile.fundamental_frequency and "pitch_shift" in adaptation:
                pitch_shift = adaptation["pitch_shift"]
                freq_range = adapted_profile.fundamental_frequency
                
                # Create new frequency range with adaptation
                from app.schemas.voice import FrequencyRange
                adapted_profile.fundamental_frequency = FrequencyRange(
                    min_hz=freq_range.min_hz * (1 + pitch_shift),
                    max_hz=freq_range.max_hz * (1 + pitch_shift),
                    mean_hz=freq_range.mean_hz * (1 + pitch_shift),
                    std_hz=freq_range.std_hz
                )
            
            # Apply formant adaptation if formant frequencies exist
            if adapted_profile.formant_frequencies and "formant_shift" in adaptation:
                formant_shifts = adaptation["formant_shift"]
                formants = adapted_profile.formant_frequencies.copy()
                
                for i, shift in enumerate(formant_shifts):
                    if i < len(formants):
                        formants[i] *= shift
                
                adapted_profile.formant_frequencies = formants
            
            return adapted_profile
            
        except Exception as e:
            logger.error(f"Voice adaptation failed: {e}")
            return voice_profile




class IntelligentFusionEngine:
    """
    Intelligent fusion algorithm for combining outputs from multiple TTS models.
    
    Implements per-segment quality scoring and weighted combination with
    cross-fade at segment boundaries for seamless audio output.
    
    The fusion algorithm uses multi-dimensional quality metrics (similarity,
    naturalness, clarity, prosody) with adaptive weighting based on model
    strengths per segment. Softmax normalization ensures proper weight
    distribution across models.
    
    Requirements: 6.1, 6.2, 6.3
    """
    
    # Quality dimension weights for computing composite scores
    QUALITY_DIMENSION_WEIGHTS = {
        "similarity": 0.35,    # Speaker similarity is most important
        "naturalness": 0.25,   # Natural sounding speech
        "clarity": 0.20,       # Clear articulation
        "prosody": 0.20        # Rhythm and intonation
    }
    
    # Temperature for softmax weighting (lower = more selective)
    SOFTMAX_TEMPERATURE = 0.5
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.quality_assessor = QualityAssessment()
        
        # Model-specific strength profiles for adaptive weighting
        self.model_strength_profiles = {
            TTSModelType.XTTS_V2: {
                "similarity": 0.90,
                "naturalness": 0.85,
                "clarity": 0.88,
                "prosody": 0.82
            },
            TTSModelType.STYLETTS2: {
                "similarity": 0.85,
                "naturalness": 0.92,
                "clarity": 0.85,
                "prosody": 0.95
            },
            TTSModelType.OPENVOICE: {
                "similarity": 0.88,
                "naturalness": 0.80,
                "clarity": 0.82,
                "prosody": 0.78
            },
            TTSModelType.BARK: {
                "similarity": 0.75,
                "naturalness": 0.88,
                "clarity": 0.80,
                "prosody": 0.90
            },
            TTSModelType.YOUR_TTS: {
                "similarity": 0.80,
                "naturalness": 0.78,
                "clarity": 0.82,
                "prosody": 0.75
            }
        }
    
    def fuse_synthesis_results(
        self,
        results: List[SynthesisResult],
        reference_audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fuse multiple synthesis results using intelligent weighted combination.
        
        This method implements the core intelligent fusion algorithm that:
        1. Computes per-segment quality scores for each model
        2. Calculates adaptive weights based on multi-dimensional quality metrics
        3. Applies softmax normalization for proper weight distribution
        4. Combines audio segments using weighted averaging
        5. Applies cross-fade at segment boundaries for seamless output
        
        Requirements: 6.2 - Intelligent weighted fusion based on per-model quality scores
        
        Args:
            results: List of synthesis results from different models
            reference_audio: Reference audio for quality comparison
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (fused_audio, fusion_metadata)
        """
        # Filter valid results
        valid_results = [r for r in results if r.audio_data is not None]
        
        if not valid_results:
            raise ValueError("No valid synthesis results to fuse")
        
        if len(valid_results) == 1:
            # Still compute segment scores for single model
            segment_scores = self._compute_segment_scores(
                valid_results, reference_audio, sample_rate
            )
            # Store per-segment quality in the result
            for result in valid_results:
                if result.model_type in segment_scores:
                    result.per_segment_quality = segment_scores[result.model_type]
            
            return valid_results[0].audio_data, {
                "fusion_method": "single_model",
                "models_used": [valid_results[0].model_type.value],
                "per_model_segment_scores": self._format_segment_scores_for_metadata(segment_scores)
            }
        
        # Calculate segment-level quality scores for each result
        segment_scores = self._compute_segment_scores(
            valid_results, reference_audio, sample_rate
        )
        
        # Store per-segment quality scores in each SynthesisResult
        for result in valid_results:
            if result.model_type in segment_scores:
                result.per_segment_quality = segment_scores[result.model_type]
        
        # Perform intelligent fusion based on segment scores
        if self.config.use_weighted_combination:
            fused_audio, fusion_stats = self._intelligent_weighted_fusion(
                valid_results, segment_scores, sample_rate
            )
            fusion_method = "intelligent_weighted_fusion"
        else:
            fused_audio = self._best_segment_selection(
                valid_results, segment_scores, sample_rate
            )
            fusion_method = "best_segment_selection"
            fusion_stats = {}
        
        # Apply cross-fade at segment boundaries
        fused_audio = self._apply_segment_crossfade(fused_audio, sample_rate)
        
        # Compile fusion metadata with detailed per-segment scores
        metadata = {
            "fusion_method": fusion_method,
            "models_used": [r.model_type.value for r in valid_results],
            "segment_count": len(segment_scores.get(valid_results[0].model_type, [])),
            "average_quality": np.mean([
                np.mean([s.overall_score for s in scores])
                for scores in segment_scores.values()
            ]),
            "per_model_segment_scores": self._format_segment_scores_for_metadata(segment_scores),
            "fusion_statistics": fusion_stats
        }
        
        return fused_audio, metadata
    
    def _format_segment_scores_for_metadata(
        self,
        segment_scores: Dict[TTSModelType, List[SegmentQualityScore]]
    ) -> Dict[str, Any]:
        """
        Format per-segment quality scores for inclusion in metadata.
        
        Returns a dictionary with model names as keys and lists of segment
        quality information as values.
        """
        formatted = {}
        
        for model_type, scores in segment_scores.items():
            model_segments = []
            for score in scores:
                model_segments.append({
                    "segment_index": score.segment_index,
                    "start_time": round(score.start_time, 3),
                    "end_time": round(score.end_time, 3),
                    "similarity_score": round(score.similarity_score, 4),
                    "naturalness_score": round(score.naturalness_score, 4),
                    "clarity_score": round(score.clarity_score, 4),
                    "prosody_score": round(score.prosody_score, 4),
                    "overall_score": round(score.overall_score, 4)
                })
            
            # Calculate model-level statistics
            if scores:
                avg_overall = np.mean([s.overall_score for s in scores])
                avg_similarity = np.mean([s.similarity_score for s in scores])
                avg_naturalness = np.mean([s.naturalness_score for s in scores])
                avg_clarity = np.mean([s.clarity_score for s in scores])
                avg_prosody = np.mean([s.prosody_score for s in scores])
                
                formatted[model_type.value] = {
                    "segments": model_segments,
                    "statistics": {
                        "segment_count": len(scores),
                        "average_overall": round(avg_overall, 4),
                        "average_similarity": round(avg_similarity, 4),
                        "average_naturalness": round(avg_naturalness, 4),
                        "average_clarity": round(avg_clarity, 4),
                        "average_prosody": round(avg_prosody, 4),
                        "min_overall": round(min(s.overall_score for s in scores), 4),
                        "max_overall": round(max(s.overall_score for s in scores), 4)
                    }
                }
            else:
                formatted[model_type.value] = {
                    "segments": [],
                    "statistics": {}
                }
        
        return formatted
    
    def _compute_segment_scores(
        self,
        results: List[SynthesisResult],
        reference_audio: np.ndarray,
        sample_rate: int
    ) -> Dict[TTSModelType, List[SegmentQualityScore]]:
        """Compute per-segment quality scores for each model's output."""
        segment_duration_samples = int(
            self.config.segment_duration_ms * sample_rate / 1000
        )
        
        # Find the maximum audio length
        max_length = max(len(r.audio_data) for r in results)
        
        # Compute number of segments
        num_segments = max(1, max_length // segment_duration_samples)
        
        segment_scores: Dict[TTSModelType, List[SegmentQualityScore]] = {}
        
        for result in results:
            scores = []
            audio = result.audio_data
            
            # Pad audio if needed
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            
            # Pad reference if needed
            ref_audio = reference_audio
            if len(ref_audio) < max_length:
                ref_audio = np.pad(ref_audio, (0, max_length - len(ref_audio)), mode='constant')
            
            for i in range(num_segments):
                start_sample = i * segment_duration_samples
                end_sample = min((i + 1) * segment_duration_samples, len(audio))
                
                segment = audio[start_sample:end_sample]
                ref_segment = ref_audio[start_sample:end_sample]
                
                if len(segment) < 100:  # Skip very short segments
                    continue
                
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                
                score = self.quality_assessor.assess_segment_quality(
                    segment, ref_segment, sample_rate,
                    i, start_time, end_time, result.model_type
                )
                scores.append(score)
            
            segment_scores[result.model_type] = scores
        
        return segment_scores
    
    def _compute_adaptive_weights(
        self,
        segment_scores: Dict[TTSModelType, SegmentQualityScore],
        segment_index: int
    ) -> Dict[TTSModelType, float]:
        """
        Compute adaptive weights for each model based on multi-dimensional quality metrics.
        
        This method implements the core of the intelligent weighted fusion algorithm:
        1. Extracts quality dimensions (similarity, naturalness, clarity, prosody)
        2. Applies model-specific strength profiles for adaptive weighting
        3. Computes composite scores using dimension weights
        4. Applies softmax normalization for proper weight distribution
        
        Requirements: 6.2 - Intelligent weighted fusion based on per-model quality scores
        
        Args:
            segment_scores: Quality scores for each model for a specific segment
            segment_index: Index of the current segment
            
        Returns:
            Dictionary mapping model types to their fusion weights
        """
        if not segment_scores:
            return {}
        
        # Compute composite scores for each model
        composite_scores = {}
        
        for model_type, score in segment_scores.items():
            # Extract quality dimensions
            dimensions = {
                "similarity": score.similarity_score,
                "naturalness": score.naturalness_score,
                "clarity": score.clarity_score,
                "prosody": score.prosody_score
            }
            
            # Get model strength profile (default to neutral if not found)
            model_strengths = self.model_strength_profiles.get(
                model_type,
                {"similarity": 0.80, "naturalness": 0.80, "clarity": 0.80, "prosody": 0.80}
            )
            
            # Compute weighted composite score with adaptive model strengths
            composite = 0.0
            for dim_name, dim_weight in self.QUALITY_DIMENSION_WEIGHTS.items():
                dim_score = dimensions.get(dim_name, 0.5)
                model_strength = model_strengths.get(dim_name, 0.80)
                
                # Adaptive weighting: boost score if model is strong in this dimension
                # and the actual score is close to or above the model's expected strength
                strength_factor = 1.0 + 0.2 * (dim_score / model_strength - 0.8)
                strength_factor = max(0.5, min(1.5, strength_factor))  # Clamp
                
                composite += dim_weight * dim_score * strength_factor
            
            composite_scores[model_type] = composite
        
        # Apply quality threshold filtering
        filtered_scores = {
            model: score for model, score in composite_scores.items()
            if score >= self.config.min_quality_threshold
        }
        
        # If all models are below threshold, use all models with reduced weights
        if not filtered_scores:
            filtered_scores = composite_scores
        
        # Apply softmax normalization for proper weight distribution
        weights = self._softmax_normalize(filtered_scores)
        
        return weights
    
    def _softmax_normalize(
        self,
        scores: Dict[TTSModelType, float]
    ) -> Dict[TTSModelType, float]:
        """
        Apply softmax normalization to convert scores to weights.
        
        Uses temperature scaling to control the selectivity of the weighting:
        - Lower temperature = more selective (winner-take-all)
        - Higher temperature = more uniform distribution
        
        Args:
            scores: Dictionary of model scores
            
        Returns:
            Dictionary of normalized weights that sum to 1.0
        """
        if not scores:
            return {}
        
        # Convert to numpy for efficient computation
        models = list(scores.keys())
        score_values = np.array([scores[m] for m in models])
        
        # Apply temperature scaling
        scaled_scores = score_values / self.SOFTMAX_TEMPERATURE
        
        # Subtract max for numerical stability
        scaled_scores = scaled_scores - np.max(scaled_scores)
        
        # Compute softmax
        exp_scores = np.exp(scaled_scores)
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Convert back to dictionary
        return {model: float(weight) for model, weight in zip(models, softmax_weights)}
    
    def _intelligent_weighted_fusion(
        self,
        results: List[SynthesisResult],
        segment_scores: Dict[TTSModelType, List[SegmentQualityScore]],
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform intelligent weighted fusion of audio from multiple models.
        
        This method implements the full intelligent fusion algorithm:
        1. For each segment, compute adaptive weights based on quality metrics
        2. Apply weighted combination of audio samples
        3. Track fusion statistics for analysis
        4. Handle edge cases (missing segments, low quality)
        
        Requirements: 6.2 - Intelligent weighted fusion based on per-model quality scores
        
        Args:
            results: List of synthesis results from different models
            segment_scores: Per-segment quality scores for each model
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (fused_audio, fusion_statistics)
        """
        segment_duration_samples = int(
            self.config.segment_duration_ms * sample_rate / 1000
        )
        
        # Find the maximum audio length
        max_length = max(len(r.audio_data) for r in results)
        
        # Determine number of segments
        num_segments = max(1, max_length // segment_duration_samples)
        
        # Initialize output array and tracking
        fused_audio = np.zeros(max_length)
        weight_sum = np.zeros(max_length)
        
        # Statistics tracking
        segment_weights_history = []
        model_contribution_totals = {r.model_type: 0.0 for r in results}
        segments_per_model_won = {r.model_type: 0 for r in results}
        
        # Create audio lookup by model type
        audio_by_model = {}
        for result in results:
            audio = result.audio_data
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            audio_by_model[result.model_type] = audio
        
        # Process each segment
        for seg_idx in range(num_segments):
            start_sample = seg_idx * segment_duration_samples
            end_sample = min((seg_idx + 1) * segment_duration_samples, max_length)
            
            # Gather segment scores for this segment index
            segment_model_scores = {}
            for model_type, scores in segment_scores.items():
                if seg_idx < len(scores):
                    segment_model_scores[model_type] = scores[seg_idx]
            
            if not segment_model_scores:
                # No scores available, use equal weights
                equal_weight = 1.0 / len(results)
                for result in results:
                    if result.model_type in audio_by_model:
                        audio = audio_by_model[result.model_type]
                        fused_audio[start_sample:end_sample] += audio[start_sample:end_sample] * equal_weight
                        weight_sum[start_sample:end_sample] += equal_weight
                continue
            
            # Compute adaptive weights for this segment
            weights = self._compute_adaptive_weights(segment_model_scores, seg_idx)
            
            # Track which model has highest weight (won this segment)
            if weights:
                winning_model = max(weights.keys(), key=lambda m: weights[m])
                segments_per_model_won[winning_model] += 1
            
            # Apply weighted combination
            segment_weights = {}
            for model_type, weight in weights.items():
                if model_type in audio_by_model:
                    audio = audio_by_model[model_type]
                    fused_audio[start_sample:end_sample] += audio[start_sample:end_sample] * weight
                    weight_sum[start_sample:end_sample] += weight
                    
                    # Track statistics
                    model_contribution_totals[model_type] += weight
                    segment_weights[model_type.value] = round(weight, 4)
            
            segment_weights_history.append({
                "segment_index": seg_idx,
                "start_time": round(start_sample / sample_rate, 3),
                "end_time": round(end_sample / sample_rate, 3),
                "weights": segment_weights
            })
        
        # Normalize by weight sum
        weight_sum = np.maximum(weight_sum, 1e-8)  # Avoid division by zero
        fused_audio = fused_audio / weight_sum
        
        # Normalize amplitude
        max_val = np.max(np.abs(fused_audio))
        if max_val > 0:
            fused_audio = fused_audio / max_val * 0.95
        
        # Compute fusion statistics
        total_contribution = sum(model_contribution_totals.values())
        if total_contribution > 0:
            model_contribution_percentages = {
                model.value: round(contrib / total_contribution * 100, 2)
                for model, contrib in model_contribution_totals.items()
            }
        else:
            model_contribution_percentages = {}
        
        fusion_stats = {
            "total_segments": num_segments,
            "model_contribution_percentages": model_contribution_percentages,
            "segments_won_per_model": {
                model.value: count for model, count in segments_per_model_won.items()
            },
            "segment_weights_history": segment_weights_history[:10],  # First 10 for brevity
            "softmax_temperature": self.SOFTMAX_TEMPERATURE,
            "quality_dimension_weights": self.QUALITY_DIMENSION_WEIGHTS
        }
        
        logger.info(f"Intelligent fusion complete: {num_segments} segments, "
                   f"contributions: {model_contribution_percentages}")
        
        return fused_audio, fusion_stats
    
    def _calculate_segment_scores(
        self,
        results: List[SynthesisResult],
        ref_audio: np.ndarray,
        sample_rate: int
    ) -> Dict[TTSModelType, List[SegmentQualityScore]]:
        """Calculate quality scores for each segment of each model's output."""
        segment_duration_samples = int(
            self.config.segment_duration_ms * sample_rate / 1000
        )
        
        segment_scores = {}
        
        for result in results:
            audio = result.audio_data
            scores = []
            
            for i in range(0, len(audio), segment_duration_samples):
                start_sample = i
                end_sample = min(i + segment_duration_samples, len(audio))
                segment = audio[start_sample:end_sample]
                ref_segment = ref_audio[start_sample:end_sample] if start_sample < len(ref_audio) else np.zeros_like(segment)
                
                if len(segment) < 100:  # Skip very short segments
                    continue
                
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                
                score = self.quality_assessor.assess_segment_quality(
                    segment, ref_segment, sample_rate,
                    i, start_time, end_time, result.model_type
                )
                scores.append(score)
            
            segment_scores[result.model_type] = scores
        
        return segment_scores
    
    def _weighted_segment_fusion(
        self,
        results: List[SynthesisResult],
        segment_scores: Dict[TTSModelType, List[SegmentQualityScore]],
        sample_rate: int
    ) -> np.ndarray:
        """Fuse audio using weighted combination based on segment quality."""
        segment_duration_samples = int(
            self.config.segment_duration_ms * sample_rate / 1000
        )
        
        # Find the maximum audio length
        max_length = max(len(r.audio_data) for r in results)
        
        # Initialize output array
        fused_audio = np.zeros(max_length)
        weight_sum = np.zeros(max_length)
        
        for result in results:
            audio = result.audio_data
            scores = segment_scores.get(result.model_type, [])
            
            # Pad audio if needed
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            
            for i, score in enumerate(scores):
                if score.overall_score < self.config.min_quality_threshold:
                    continue
                
                start_sample = i * segment_duration_samples
                end_sample = min((i + 1) * segment_duration_samples, len(audio))
                
                # Weight based on quality score
                weight = score.overall_score ** 2  # Square to emphasize quality differences
                
                fused_audio[start_sample:end_sample] += audio[start_sample:end_sample] * weight
                weight_sum[start_sample:end_sample] += weight
        
        # Normalize by weight sum
        weight_sum = np.maximum(weight_sum, 1e-8)  # Avoid division by zero
        fused_audio = fused_audio / weight_sum
        
        # Normalize amplitude
        max_val = np.max(np.abs(fused_audio))
        if max_val > 0:
            fused_audio = fused_audio / max_val * 0.95
        
        return fused_audio
    
    def _best_segment_selection(
        self,
        results: List[SynthesisResult],
        segment_scores: Dict[TTSModelType, List[SegmentQualityScore]],
        sample_rate: int
    ) -> np.ndarray:
        """Select best segment from each model based on quality scores."""
        segment_duration_samples = int(
            self.config.segment_duration_ms * sample_rate / 1000
        )
        
        # Find the maximum audio length
        max_length = max(len(r.audio_data) for r in results)
        
        # Determine number of segments
        num_segments = max(1, max_length // segment_duration_samples)
        
        # Initialize output array
        fused_audio = np.zeros(max_length)
        
        # Create audio lookup by model type
        audio_by_model = {r.model_type: r.audio_data for r in results}
        
        for i in range(num_segments):
            start_sample = i * segment_duration_samples
            end_sample = min((i + 1) * segment_duration_samples, max_length)
            
            # Find best model for this segment
            best_model = None
            best_score = -1
            
            for model_type, scores in segment_scores.items():
                if i < len(scores):
                    if scores[i].overall_score > best_score:
                        best_score = scores[i].overall_score
                        best_model = model_type
            
            if best_model is not None and best_model in audio_by_model:
                audio = audio_by_model[best_model]
                if len(audio) < max_length:
                    audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
                fused_audio[start_sample:end_sample] = audio[start_sample:end_sample]
        
        return fused_audio
    
    def _apply_segment_crossfade(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply cross-fade at segment boundaries for smooth transitions."""
        crossfade_samples = int(
            self.config.crossfade_duration_ms * sample_rate / 1000
        )
        segment_duration_samples = int(
            self.config.segment_duration_ms * sample_rate / 1000
        )
        
        if crossfade_samples < 2:
            return audio
        
        # Apply crossfade at each segment boundary
        num_segments = len(audio) // segment_duration_samples
        
        for i in range(1, num_segments):
            boundary = i * segment_duration_samples
            
            if boundary + crossfade_samples > len(audio):
                continue
            
            # Create crossfade window
            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)
            
            # Apply crossfade around boundary
            start = boundary - crossfade_samples // 2
            end = boundary + crossfade_samples // 2
            
            if start >= 0 and end <= len(audio):
                # Smooth the transition
                window = np.hanning(crossfade_samples)
                audio[start:end] = audio[start:end] * window
        
        return audio



class ParallelSynthesisExecutor:
    """
    Executor for parallel synthesis across multiple TTS models (XTTS, StyleTTS2, OpenVoice).
    
    Manages concurrent execution of synthesis tasks with proper
    resource management, fallback handling, and intelligent coordination.
    
    Requirements: 6.1, 6.2, 6.3
    """
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self._executor_lock = Lock()
        self._synthesis_semaphore = asyncio.Semaphore(max_workers)
    
    async def synthesize_parallel(
        self,
        models: Dict[TTSModelType, Any],
        text: str,
        reference_audio_path: str,
        language: str,
        sample_rate: int,
        model_configs: Dict[TTSModelType, ModelConfiguration],
        progress_callback: Optional[Callable] = None
    ) -> List[SynthesisResult]:
        """
        Execute synthesis in parallel across multiple models (XTTS, StyleTTS2, OpenVoice).
        
        This method runs all available TTS models concurrently using asyncio.gather()
        for true parallel execution, with proper timeout handling and progress reporting.
        
        Requirements: 6.1, 6.2, 6.3
        
        Args:
            models: Dictionary of loaded TTS models
            text: Text to synthesize
            reference_audio_path: Path to reference audio
            language: Target language
            sample_rate: Target sample rate
            model_configs: Model configurations
            progress_callback: Optional progress callback
            
        Returns:
            List of synthesis results from all models
        """
        results = []
        
        # Filter valid models
        valid_models = {k: v for k, v in models.items() if v is not None}
        
        if not valid_models:
            logger.warning("No models available for parallel synthesis")
            return results
        
        logger.info(f"Starting parallel synthesis with {len(valid_models)} models: {[m.value for m in valid_models.keys()]}")
        
        if progress_callback:
            progress_callback(30, f"Running parallel synthesis with {len(valid_models)} models")
        
        # Create synthesis coroutines for each model
        synthesis_coroutines = []
        model_types = []
        
        for model_type, model in valid_models.items():
            config = model_configs.get(model_type)
            if config is None:
                logger.warning(f"No config for {model_type.value}, skipping")
                continue
            
            # Create coroutine with semaphore for resource management
            coro = self._synthesize_with_semaphore(
                model_type, model, text, reference_audio_path,
                language, sample_rate, config
            )
            synthesis_coroutines.append(coro)
            model_types.append(model_type)
        
        if not synthesis_coroutines:
            logger.warning("No synthesis coroutines created")
            return results
        
        # Execute all synthesis tasks in parallel using asyncio.gather
        # This ensures true parallel execution across XTTS, StyleTTS2, and OpenVoice
        try:
            parallel_results = await asyncio.gather(
                *synthesis_coroutines,
                return_exceptions=True
            )
            
            # Process results
            for i, (model_type, result) in enumerate(zip(model_types, parallel_results)):
                if isinstance(result, Exception):
                    logger.error(f"Parallel synthesis failed for {model_type.value}: {result}")
                    results.append(SynthesisResult(
                        model_type=model_type,
                        audio_data=None,
                        sample_rate=sample_rate,
                        quality_score=0.0,
                        confidence_score=0.0,
                        processing_time=0.0,
                        error_message=str(result)
                    ))
                elif result is not None:
                    results.append(result)
                    logger.info(f"Parallel synthesis completed for {model_type.value} "
                               f"(quality: {result.quality_score:.2f})")
                
                # Update progress
                if progress_callback:
                    progress = 30 + int(40 * (i + 1) / len(model_types))
                    progress_callback(progress, f"Completed {model_type.value}")
                    
        except Exception as e:
            logger.error(f"Parallel synthesis gather failed: {e}")
        
        # Log summary
        successful = [r for r in results if r.audio_data is not None]
        logger.info(f"Parallel synthesis complete: {len(successful)}/{len(model_types)} models succeeded")
        
        return results
    
    async def _synthesize_with_semaphore(
        self,
        model_type: TTSModelType,
        model: Any,
        text: str,
        reference_audio_path: str,
        language: str,
        sample_rate: int,
        config: ModelConfiguration
    ) -> Optional[SynthesisResult]:
        """Synthesize with semaphore for resource management."""
        async with self._synthesis_semaphore:
            try:
                # Apply timeout per model
                result = await asyncio.wait_for(
                    self._synthesize_with_model_async(
                        model_type, model, text, reference_audio_path,
                        language, sample_rate, config
                    ),
                    timeout=120  # 2 minute timeout per model
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Synthesis timeout for {model_type.value}")
                return SynthesisResult(
                    model_type=model_type,
                    audio_data=None,
                    sample_rate=sample_rate,
                    quality_score=0.0,
                    confidence_score=0.0,
                    processing_time=120.0,
                    error_message="Synthesis timeout"
                )
            except Exception as e:
                logger.error(f"Synthesis error for {model_type.value}: {e}")
                return SynthesisResult(
                    model_type=model_type,
                    audio_data=None,
                    sample_rate=sample_rate,
                    quality_score=0.0,
                    confidence_score=0.0,
                    processing_time=0.0,
                    error_message=str(e)
                )
    
    async def _synthesize_with_model_async(
        self,
        model_type: TTSModelType,
        model: Any,
        text: str,
        reference_audio_path: str,
        language: str,
        sample_rate: int,
        config: ModelConfiguration
    ) -> Optional[SynthesisResult]:
        """
        Synthesize with a single model asynchronously.
        
        Handles XTTS, StyleTTS2, and OpenVoice models with appropriate
        synthesis methods for each.
        
        Requirements: 6.1, 6.2, 6.3
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting synthesis with {model_type.value}")
            
            # Route to appropriate synthesis method based on model type
            if model_type == TTSModelType.STYLETTS2:
                result = await self._synthesize_styletts2(
                    model, text, reference_audio_path, language, sample_rate
                )
            elif model_type == TTSModelType.OPENVOICE:
                result = await self._synthesize_openvoice(
                    model, text, reference_audio_path, language, sample_rate
                )
            elif model_type == TTSModelType.XTTS_V2:
                result = await self._synthesize_xtts(
                    model, text, reference_audio_path, language, sample_rate, config
                )
            else:
                # Default TTS library synthesis for other models (Bark, YourTTS)
                result = await self._synthesize_tts_library(
                    model_type, model, text, reference_audio_path, 
                    language, sample_rate, config
                )
            
            if result is not None:
                result.processing_time = time.time() - start_time
                logger.info(f"Synthesis with {model_type.value} completed in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Model synthesis failed for {model_type.value}: {e}")
            return SynthesisResult(
                model_type=model_type,
                audio_data=None,
                sample_rate=sample_rate,
                quality_score=0.0,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
    
    async def _synthesize_xtts(
        self,
        model: TTS,
        text: str,
        reference_audio_path: str,
        language: str,
        sample_rate: int,
        config: ModelConfiguration
    ) -> Optional[SynthesisResult]:
        """
        Synthesize using XTTS v2 model.
        
        XTTS is the primary model for high-quality multilingual voice cloning.
        
        Requirements: 6.1
        """
        try:
            # Convert language to XTTS format
            language_code = self._get_model_language_code(
                TTSModelType.XTTS_V2, language, config
            )
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_output_path = tmp_file.name
            
            # Prepare synthesis kwargs
            synthesis_kwargs = {
                "text": text,
                "language": language_code
            }
            
            # Add reference audio for voice cloning
            if reference_audio_path and os.path.exists(reference_audio_path):
                synthesis_kwargs["speaker_wav"] = reference_audio_path
            
            # Execute XTTS synthesis
            try:
                await asyncio.to_thread(
                    model.tts_to_file,
                    file_path=temp_output_path,
                    **synthesis_kwargs
                )
            except Exception as synthesis_error:
                error_msg = str(synthesis_error).lower()
                
                # Handle language errors by falling back to English
                if "language" in error_msg:
                    logger.warning(f"XTTS language error, trying English")
                    synthesis_kwargs["language"] = "en"
                    await asyncio.to_thread(
                        model.tts_to_file,
                        file_path=temp_output_path,
                        **synthesis_kwargs
                    )
                else:
                    raise synthesis_error
            
            # Load the generated audio
            audio_data, sr = librosa.load(temp_output_path, sr=sample_rate)
            
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            return SynthesisResult(
                model_type=TTSModelType.XTTS_V2,
                audio_data=audio_data,
                sample_rate=sr,
                quality_score=config.quality_score,
                confidence_score=0.90,
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"XTTS synthesis failed: {e}")
            return None
    
    async def _synthesize_styletts2(
        self,
        model: Any,
        text: str,
        reference_audio_path: str,
        language: str,
        sample_rate: int
    ) -> Optional[SynthesisResult]:
        """
        Synthesize using StyleTTS2 for human-level voice cloning.
        
        StyleTTS2 provides exceptional prosody and naturalness through
        style diffusion and adversarial training.
        
        Requirements: 6.1, 6.2
        """
        try:
            from app.services.styletts2_synthesizer import get_styletts2_synthesizer
            
            synthesizer = get_styletts2_synthesizer()
            
            # Ensure synthesizer is initialized
            if not synthesizer._initialized:
                synthesizer.initialize()
            
            # Load reference audio
            ref_audio, ref_sr = librosa.load(reference_audio_path, sr=sample_rate)
            
            # Synthesize with StyleTTS2
            result = await asyncio.to_thread(
                synthesizer.synthesize,
                text=text,
                reference_audio=ref_audio,
                reference_sr=ref_sr,
                style_strength=1.0,
                prosody_strength=1.0
            )
            
            if result is not None and result.audio is not None:
                # Resample if needed
                audio = result.audio
                if result.sample_rate != sample_rate:
                    audio = librosa.resample(
                        audio, orig_sr=result.sample_rate, target_sr=sample_rate
                    )
                
                return SynthesisResult(
                    model_type=TTSModelType.STYLETTS2,
                    audio_data=audio,
                    sample_rate=sample_rate,
                    quality_score=result.quality_score,
                    confidence_score=0.88,
                    processing_time=0.0
                )
            
            logger.warning("StyleTTS2 returned no result")
            return None
            
        except Exception as e:
            logger.error(f"StyleTTS2 synthesis failed: {e}")
            return None
    
    async def _synthesize_openvoice(
        self,
        model: Any,
        text: str,
        reference_audio_path: str,
        language: str,
        sample_rate: int
    ) -> Optional[SynthesisResult]:
        """
        Synthesize using OpenVoice for instant voice cloning.
        
        OpenVoice enables instant voice cloning with granular control
        over voice styles including emotion, accent, and rhythm.
        
        Requirements: 6.1, 6.3
        """
        try:
            from app.services.openvoice_synthesizer import get_openvoice_synthesizer
            
            synthesizer = get_openvoice_synthesizer()
            
            # Ensure synthesizer is initialized
            if not synthesizer._initialized:
                synthesizer.initialize()
            
            # Load reference audio
            ref_audio, ref_sr = librosa.load(reference_audio_path, sr=sample_rate)
            
            # Synthesize with OpenVoice
            result = await asyncio.to_thread(
                synthesizer.synthesize_with_reference,
                text=text,
                reference_audio=ref_audio,
                reference_sr=ref_sr,
                style_params={"language": language}
            )
            
            if result is not None and result.audio is not None:
                # Resample if needed
                audio = result.audio
                if result.sample_rate != sample_rate:
                    audio = librosa.resample(
                        audio, orig_sr=result.sample_rate, target_sr=sample_rate
                    )
                
                return SynthesisResult(
                    model_type=TTSModelType.OPENVOICE,
                    audio_data=audio,
                    sample_rate=sample_rate,
                    quality_score=result.quality_score,
                    confidence_score=0.85,
                    processing_time=0.0
                )
            
            logger.warning("OpenVoice returned no result")
            return None
            
        except Exception as e:
            logger.error(f"OpenVoice synthesis failed: {e}")
            return None
    
    async def _synthesize_tts_library(
        self,
        model_type: TTSModelType,
        model: TTS,
        text: str,
        reference_audio_path: str,
        language: str,
        sample_rate: int,
        config: ModelConfiguration
    ) -> Optional[SynthesisResult]:
        """Synthesize using TTS library models (XTTS, Bark, YourTTS)."""
        try:
            # Convert language to model-specific format
            model_language = self._get_model_language_code(model_type, language, config)
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_output_path = tmp_file.name
            
            # Prepare synthesis kwargs
            synthesis_kwargs = {
                "text": text,
                "language": model_language
            }
            
            # Add reference audio if available
            if reference_audio_path and os.path.exists(reference_audio_path):
                synthesis_kwargs["speaker_wav"] = reference_audio_path
            
            # Execute synthesis
            try:
                if hasattr(model, 'tts_to_file'):
                    await asyncio.to_thread(
                        model.tts_to_file,
                        file_path=temp_output_path,
                        **synthesis_kwargs
                    )
                else:
                    audio_data = await asyncio.to_thread(model.tts, **synthesis_kwargs)
                    sf.write(temp_output_path, audio_data, sample_rate)
                    
            except Exception as synthesis_error:
                error_msg = str(synthesis_error).lower()
                
                # Handle language errors
                if "language" in error_msg:
                    logger.warning(f"Language error for {model_type.value}, trying English")
                    synthesis_kwargs["language"] = "en"
                    
                    if hasattr(model, 'tts_to_file'):
                        await asyncio.to_thread(
                            model.tts_to_file,
                            file_path=temp_output_path,
                            **synthesis_kwargs
                        )
                    else:
                        audio_data = await asyncio.to_thread(model.tts, **synthesis_kwargs)
                        sf.write(temp_output_path, audio_data, sample_rate)
                else:
                    raise synthesis_error
            
            # Load the generated audio
            audio_data, sr = librosa.load(temp_output_path, sr=sample_rate)
            
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            return SynthesisResult(
                model_type=model_type,
                audio_data=audio_data,
                sample_rate=sr,
                quality_score=config.quality_score,
                confidence_score=config.quality_score * 0.9,
                processing_time=0.0
            )
            
        except Exception as e:
            logger.error(f"TTS library synthesis failed for {model_type.value}: {e}")
            return None
    
    def _get_model_language_code(
        self,
        model_type: TTSModelType,
        language: str,
        config: ModelConfiguration
    ) -> str:
        """Convert language to model-specific format."""
        language_mappings = {
            TTSModelType.XTTS_V2: {
                'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
                'italian': 'it', 'portuguese': 'pt', 'polish': 'pl', 'turkish': 'tr',
                'russian': 'ru', 'dutch': 'nl', 'czech': 'cs', 'arabic': 'ar',
                'chinese': 'zh-cn', 'japanese': 'ja', 'hungarian': 'hu', 'korean': 'ko',
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'pt': 'pt'
            },
            TTSModelType.BARK: {
                'english': 'en', 'german': 'de', 'spanish': 'es', 'french': 'fr',
                'hindi': 'hi', 'italian': 'it', 'japanese': 'ja', 'korean': 'ko',
                'polish': 'pl', 'portuguese': 'pt', 'russian': 'ru', 'turkish': 'tr',
                'chinese': 'zh', 'en': 'en', 'de': 'de', 'es': 'es', 'fr': 'fr'
            },
            TTSModelType.YOUR_TTS: {
                'english': 'en', 'french': 'fr-fr', 'portuguese': 'pt-br', 'turkish': 'tr',
                'en': 'en', 'fr': 'fr-fr', 'pt': 'pt-br', 'tr': 'tr'
            }
        }
        
        model_mapping = language_mappings.get(model_type, {})
        mapped_language = model_mapping.get(language.lower())
        
        if mapped_language:
            return mapped_language
        
        # Check if language is in supported list
        if language in config.languages:
            return language
        
        # Default to English
        return model_mapping.get('english', 'en')



class EnsembleVoiceSynthesizer:
    """
    Main ensemble voice synthesis engine that coordinates multiple TTS models
    for optimal voice replication quality with parallel synthesis and intelligent fusion.
    
    This engine implements parallel synthesis across XTTS v2, StyleTTS2, and OpenVoice
    models, running them concurrently using asyncio.gather() for maximum efficiency.
    Results are then fused using an intelligent weighted combination algorithm
    based on per-segment quality scores.
    
    Key Features:
    - Parallel synthesis across XTTS, StyleTTS2, and OpenVoice models
    - Per-segment quality scoring for each model output
    - Intelligent weighted fusion based on quality metrics
    - Cross-fade at segment boundaries for seamless audio
    - Automatic fallback handling when models fail
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    
    def __init__(self):
        self.sample_rate = 22050
        self.models_dir = Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model_selector = ModelSelector()
        self.quality_assessor = QualityAssessment()
        self.cross_language_adapter = CrossLanguageAdapter()
        self.audio_preprocessor = AdvancedAudioPreprocessor()
        self.fusion_engine = IntelligentFusionEngine()
        self.parallel_executor = ParallelSynthesisExecutor()
        
        # Model configurations
        self.model_configs = {
            TTSModelType.XTTS_V2: ModelConfiguration(
                model_type=TTSModelType.XTTS_V2,
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                quality_score=0.95,
                supports_cloning=True,
                languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
                max_text_length=500,
                processing_time_factor=1.0,
                memory_usage="high"
            ),
            TTSModelType.STYLETTS2: ModelConfiguration(
                model_type=TTSModelType.STYLETTS2,
                model_name="styletts2",
                quality_score=0.95,
                supports_cloning=True,
                languages=["en"],
                max_text_length=400,
                processing_time_factor=0.8,
                memory_usage="high"
            ),
            TTSModelType.OPENVOICE: ModelConfiguration(
                model_type=TTSModelType.OPENVOICE,
                model_name="openvoice",
                quality_score=0.88,
                supports_cloning=True,
                languages=["en", "es", "fr", "zh", "ja", "ko"],
                max_text_length=400,
                processing_time_factor=1.2,
                memory_usage="medium"
            ),
            TTSModelType.BARK: ModelConfiguration(
                model_type=TTSModelType.BARK,
                model_name="tts_models/multilingual/multi-dataset/bark",
                quality_score=0.92,
                supports_cloning=True,
                languages=["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"],
                max_text_length=300,
                processing_time_factor=2.5,
                memory_usage="high"
            ),
        }

        # Loaded models cache
        self.loaded_models: Dict[TTSModelType, Any] = {}
        self.model_load_status: Dict[TTSModelType, bool] = {}
        
        # Synthesis parameters
        self.ensemble_weights = EnsembleWeights(
            model_weights={
                TTSModelType.XTTS_V2: 0.35,
                TTSModelType.STYLETTS2: 0.30,
                TTSModelType.OPENVOICE: 0.20,
                TTSModelType.BARK: 0.15
            },
            quality_threshold=0.8,
            confidence_threshold=0.75,
            similarity_threshold=0.85
        )
    
    async def initialize_models(self, progress_callback: Optional[Callable] = None) -> bool:
        """Initialize ensemble TTS models."""
        try:
            if progress_callback:
                progress_callback(5, "Initializing ensemble TTS models")
            
            logger.info("Loading ensemble TTS models...")
            
            # Load TTS library models
            tts_models = [
                (TTSModelType.XTTS_V2, self.model_configs[TTSModelType.XTTS_V2]),
            ]
            
            loaded_count = 0
            
            for i, (model_type, config) in enumerate(tts_models):
                progress = 10 + (i * 30)
                if progress_callback:
                    progress_callback(progress, f"Loading {config.model_name}")
                
                try:
                    success = await self._load_model_async(model_type, config)
                    if success:
                        loaded_count += 1
                        logger.info(f"Successfully loaded {model_type.value}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_type.value}: {e}")
            
            # Initialize StyleTTS2 and OpenVoice (they have their own initialization)
            try:
                from app.services.styletts2_synthesizer import get_styletts2_synthesizer
                styletts2 = get_styletts2_synthesizer()
                if styletts2.initialize():
                    self.loaded_models[TTSModelType.STYLETTS2] = styletts2
                    self.model_load_status[TTSModelType.STYLETTS2] = True
                    loaded_count += 1
                    logger.info("StyleTTS2 initialized")
            except Exception as e:
                logger.warning(f"StyleTTS2 not available: {e}")

            try:
                from app.services.openvoice_synthesizer import get_openvoice_synthesizer
                openvoice = get_openvoice_synthesizer()
                if openvoice.initialize():
                    self.loaded_models[TTSModelType.OPENVOICE] = openvoice
                    self.model_load_status[TTSModelType.OPENVOICE] = True
                    loaded_count += 1
                    logger.info("OpenVoice initialized")
            except Exception as e:
                logger.warning(f"OpenVoice not available: {e}")
            
            if progress_callback:
                progress_callback(90, f"Loaded {loaded_count} models")
            
            # Verify at least one model loaded
            if loaded_count == 0:
                raise RuntimeError("No TTS models could be loaded for ensemble synthesis")
            
            if progress_callback:
                progress_callback(100, f"Ensemble synthesis ready with {loaded_count} models")
            
            logger.info(f"Ensemble voice synthesizer initialized with {loaded_count} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble models: {e}")
            if progress_callback:
                progress_callback(0, f"Ensemble initialization failed: {str(e)}")
            raise
    
    async def _load_model_async(self, model_type: TTSModelType, config: ModelConfiguration) -> bool:
        """Load a single TTS model asynchronously."""
        try:
            model = await asyncio.to_thread(
                TTS, 
                model_name=config.model_name, 
                progress_bar=False
            )
            
            self.loaded_models[model_type] = model
            self.model_load_status[model_type] = True
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "invalid load key" in error_msg or "corrupted" in error_msg:
                logger.warning(f"Model {model_type.value} appears corrupted, clearing cache...")
                self._clear_model_cache(model_type)
            
            logger.error(f"Failed to load {model_type.value}: {e}")
            self.model_load_status[model_type] = False
            return False
    
    def _clear_model_cache(self, model_type: TTSModelType):
        """Clear corrupted model cache."""
        try:
            cache_locations = [
                Path.home() / ".cache" / "tts",
                Path.home() / "AppData" / "Local" / "tts"
            ]
            
            for cache_dir in cache_locations:
                if not cache_dir.exists():
                    continue
                model_cache_dirs = list(cache_dir.glob(f"*{model_type.value}*"))
                for cache_dir_path in model_cache_dirs:
                    if cache_dir_path.is_dir():
                        logger.info(f"Removing corrupted cache: {cache_dir_path}")
                        shutil.rmtree(cache_dir_path, ignore_errors=True)
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup model cache: {cleanup_error}")

    async def synthesize_speech_ensemble(
        self,
        text: str,
        voice_profile: VoiceProfileSchema,
        language: str = "en",
        voice_settings: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Synthesize speech using ensemble of TTS models with parallel synthesis
        and intelligent fusion for maximum quality.
        
        Requirements: 6.1, 6.2, 6.3
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile for cloning
            language: Target language
            voice_settings: Optional voice settings
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (success, output_path, metadata)
        """
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(5, "Preparing ensemble synthesis")
            
            # Validate inputs
            if not text or not text.strip():
                return False, None, {"error": "Empty text input"}
            
            if not self.loaded_models:
                return False, None, {"error": "No models loaded for ensemble synthesis"}
            
            # Select optimal models for this voice and text
            selected_models = self.model_selector.select_optimal_models(
                voice_profile, text, language
            )
            
            # Filter to only loaded models
            available_models = {
                m: self.loaded_models[m] 
                for m in selected_models 
                if m in self.loaded_models
            }
            
            if not available_models:
                # Fallback to any available model
                available_models = dict(list(self.loaded_models.items())[:3])
            
            if not available_models:
                return False, None, {"error": "No suitable models available"}
            
            if progress_callback:
                progress_callback(15, f"Using {len(available_models)} models for parallel synthesis")
            
            # Adapt voice profile for target language if needed
            if language != "en":
                adapted_profile = self.cross_language_adapter.adapt_voice_for_language(
                    voice_profile, language
                )
            else:
                adapted_profile = voice_profile
            
            # Prepare reference audio
            reference_audio_path = await self._prepare_reference_audio(adapted_profile)
            if not reference_audio_path:
                return await self._synthesize_without_reference(
                    text, language, available_models, progress_callback
                )

            if progress_callback:
                progress_callback(25, "Starting parallel synthesis across models")
            
            # Execute parallel synthesis across all selected models
            synthesis_results = await self.parallel_executor.synthesize_parallel(
                models=available_models,
                text=text,
                reference_audio_path=reference_audio_path,
                language=language,
                sample_rate=self.sample_rate,
                model_configs=self.model_configs,
                progress_callback=progress_callback
            )
            
            # Filter successful results
            valid_results = [r for r in synthesis_results if r.audio_data is not None]
            
            if not valid_results:
                # Fallback handling: try models one by one
                logger.warning("All parallel synthesis failed, attempting fallback")
                return await self._fallback_synthesis(
                    text, reference_audio_path, language, available_models, progress_callback
                )
            
            if progress_callback:
                progress_callback(70, "Assessing synthesis quality")
            
            # Load reference audio for quality assessment
            reference_audio, _ = librosa.load(reference_audio_path, sr=self.sample_rate)
            
            # Assess quality of each result
            for result in valid_results:
                quality_metrics = self.quality_assessor.assess_synthesis_quality(
                    result.audio_data, reference_audio, self.sample_rate
                )
                result.similarity_score = quality_metrics["similarity"]
                result.quality_score = quality_metrics["overall"]
                result.segment_scores = quality_metrics
            
            if progress_callback:
                progress_callback(80, "Performing intelligent fusion")
            
            # Perform intelligent fusion of results
            try:
                fused_audio, fusion_metadata = self.fusion_engine.fuse_synthesis_results(
                    valid_results, reference_audio, self.sample_rate
                )
            except Exception as fusion_error:
                logger.warning(f"Fusion failed, using best single result: {fusion_error}")
                best_result = max(valid_results, key=lambda r: r.quality_score)
                fused_audio = best_result.audio_data
                fusion_metadata = {"fusion_method": "best_single", "models_used": [best_result.model_type.value]}
            
            # Apply voice settings if provided
            if voice_settings:
                if progress_callback:
                    progress_callback(85, "Applying voice settings")
                fused_audio = await self._apply_voice_settings(fused_audio, voice_settings)
            
            if progress_callback:
                progress_callback(90, "Applying iterative refinement")
            
            # Apply iterative refinement
            refined_audio = await self._apply_iterative_refinement(
                fused_audio, reference_audio, valid_results
            )
            
            # Save final output
            output_path = await self._save_ensemble_output(refined_audio, voice_profile.id)
            
            processing_time = time.time() - start_time
            
            # Calculate final quality metrics
            final_quality = self.quality_assessor.assess_synthesis_quality(
                refined_audio, reference_audio, self.sample_rate
            )

            # Generate metadata
            metadata = {
                "text": text,
                "language": language,
                "voice_profile_id": voice_profile.id,
                "processing_time": processing_time,
                "sample_rate": self.sample_rate,
                "duration": len(refined_audio) / self.sample_rate,
                "ensemble_models": [r.model_type.value for r in valid_results],
                "model_count": len(valid_results),
                "quality_metrics": final_quality,
                "similarity_score": final_quality["similarity"],
                "synthesis_method": "parallel_ensemble_fusion",
                "fusion_metadata": fusion_metadata,
                "iterative_refinement": True
            }
            
            # Cleanup temporary files
            if os.path.exists(reference_audio_path):
                try:
                    os.remove(reference_audio_path)
                except Exception:
                    pass
            
            if progress_callback:
                progress_callback(100, f"Ensemble synthesis complete (similarity: {final_quality['similarity']:.1%})")
            
            logger.info(f"Ensemble synthesis completed: {output_path} (similarity: {final_quality['similarity']:.1%})")
            return True, output_path, metadata
            
        except Exception as e:
            logger.error(f"Ensemble synthesis failed: {e}")
            if progress_callback:
                progress_callback(0, f"Ensemble synthesis failed: {str(e)}")
            return False, None, {"error": str(e)}
    
    async def _synthesize_without_reference(
        self,
        text: str,
        language: str,
        available_models: Dict[TTSModelType, Any],
        progress_callback: Optional[Callable]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Synthesize without reference audio using basic TTS."""
        logger.warning("No reference audio available, attempting synthesis without reference")
        
        try:
            model_type = list(available_models.keys())[0]
            model = available_models[model_type]
            
            if progress_callback:
                progress_callback(50, f"Synthesizing with {model_type.value} (no reference)")
            
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"ensemble_synthesis_{int(time.time())}.wav"
            
            config = self.model_configs.get(model_type)
            language_code = self.parallel_executor._get_model_language_code(
                model_type, language, config
            ) if config else "en"
            
            synthesis_kwargs = {"text": text, "language": language_code}
            
            if hasattr(model, 'tts_to_file'):
                await asyncio.to_thread(model.tts_to_file, file_path=str(output_path), **synthesis_kwargs)
            else:
                wav = await asyncio.to_thread(model.tts, **synthesis_kwargs)
                sf.write(str(output_path), wav, self.sample_rate)
            
            if progress_callback:
                progress_callback(100, "Synthesis completed without reference")
            
            return True, str(output_path), {
                "processing_time": 0.0,
                "quality_metrics": {"overall_similarity": 0.5, "confidence_score": 0.6},
                "synthesis_method": "basic_tts",
                "models_used": [model_type.value],
                "note": "Synthesized without reference audio"
            }
            
        except Exception as e:
            logger.error(f"Basic synthesis failed: {e}")
            return False, None, {"error": f"Basic synthesis failed: {str(e)}"}

    async def _fallback_synthesis(
        self,
        text: str,
        reference_audio_path: str,
        language: str,
        available_models: Dict[TTSModelType, Any],
        progress_callback: Optional[Callable]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Fallback synthesis when parallel synthesis fails.
        
        Tries models one by one until one succeeds.
        Requirements: 6.4
        """
        logger.info("Attempting fallback synthesis with individual models")
        
        for model_type, model in available_models.items():
            try:
                if progress_callback:
                    progress_callback(50, f"Fallback: trying {model_type.value}")
                
                config = self.model_configs.get(model_type)
                if config is None:
                    continue
                
                result = await self.parallel_executor._synthesize_with_model_async(
                    model_type, model, text, reference_audio_path,
                    language, self.sample_rate, config
                )
                
                if result is not None and result.audio_data is not None:
                    # Save the result
                    output_path = await self._save_ensemble_output(
                        result.audio_data, f"fallback_{int(time.time())}"
                    )
                    
                    if progress_callback:
                        progress_callback(100, f"Fallback synthesis completed with {model_type.value}")
                    
                    return True, output_path, {
                        "synthesis_method": "fallback_single_model",
                        "models_used": [model_type.value],
                        "quality_score": result.quality_score,
                        "note": "Used fallback synthesis after parallel failure"
                    }
                    
            except Exception as e:
                logger.warning(f"Fallback with {model_type.value} failed: {e}")
                continue
        
        return False, None, {"error": "All fallback synthesis attempts failed"}
    
    async def _prepare_reference_audio(self, voice_profile: VoiceProfileSchema) -> Optional[str]:
        """Prepare reference audio for synthesis."""
        try:
            if hasattr(voice_profile, 'reference_audio_id') and voice_profile.reference_audio_id:
                reference_id = voice_profile.reference_audio_id
                
                # Check if reference_audio_id is already a file path
                if Path(reference_id).exists() and Path(reference_id).is_file():
                    logger.info(f"Using direct reference audio path: {reference_id}")
                    return reference_id
                
                # Look for reference audio in uploads directory
                possible_paths = [
                    Path(settings.UPLOAD_DIR) / f"{reference_id}.wav",
                    Path(settings.UPLOAD_DIR) / f"{reference_id}.mp3",
                    Path("uploads") / f"{reference_id}.wav",
                    Path("uploads") / f"{reference_id}.mp3",
                ]
                
                # Check session-based uploads
                upload_dir = Path(settings.UPLOAD_DIR) if hasattr(settings, 'UPLOAD_DIR') else Path("uploads")
                for session_dir in upload_dir.glob("session_*"):
                    possible_paths.extend([
                        session_dir / f"{reference_id}.wav",
                        session_dir / f"{reference_id}.mp3",
                    ])
                
                for path in possible_paths:
                    if path.exists() and path.is_file():
                        logger.info(f"Found reference audio: {path}")
                        return str(path)
                
                logger.warning(f"Reference audio not found for ID: {reference_id}")
            
            logger.warning("No reference audio available for voice profile")
            return None
            
        except Exception as e:
            logger.error(f"Failed to prepare reference audio: {e}")
            return None

    async def _apply_iterative_refinement(
        self, 
        audio: np.ndarray, 
        reference_audio: np.ndarray, 
        synthesis_results: List[SynthesisResult]
    ) -> np.ndarray:
        """Apply iterative refinement with quality feedback loops."""
        try:
            refined_audio = audio.copy()
            
            # Apply spectral matching
            refined_audio = self._apply_spectral_matching(refined_audio, reference_audio)
            
            # Apply prosody correction
            refined_audio = self._apply_prosody_correction(refined_audio, reference_audio)
            
            # Apply voice characteristic enhancement
            refined_audio = self._enhance_voice_characteristics(refined_audio, reference_audio)
            
            return refined_audio
            
        except Exception as e:
            logger.error(f"Iterative refinement failed: {e}")
            return audio
    
    def _apply_spectral_matching(self, audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """Apply spectral matching to align frequency characteristics."""
        try:
            audio_stft = librosa.stft(audio)
            ref_stft = librosa.stft(reference_audio)
            
            audio_magnitude = np.abs(audio_stft)
            ref_magnitude = np.abs(ref_stft)
            
            audio_envelope = np.mean(audio_magnitude, axis=1, keepdims=True)
            ref_envelope = np.mean(ref_magnitude, axis=1, keepdims=True)
            
            matching_strength = 0.3
            target_envelope = (1 - matching_strength) * audio_envelope + matching_strength * ref_envelope
            
            envelope_ratio = target_envelope / (audio_envelope + 1e-8)
            matched_magnitude = audio_magnitude * envelope_ratio
            
            matched_stft = matched_magnitude * np.exp(1j * np.angle(audio_stft))
            matched_audio = librosa.istft(matched_stft)
            
            return matched_audio
            
        except Exception as e:
            logger.error(f"Spectral matching failed: {e}")
            return audio
    
    def _apply_prosody_correction(self, audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """Apply prosody correction to match reference patterns."""
        try:
            f0_audio, _, _ = librosa.pyin(audio, fmin=80, fmax=400, sr=self.sample_rate)
            f0_ref, _, _ = librosa.pyin(reference_audio, fmin=80, fmax=400, sr=self.sample_rate)
            
            f0_audio_clean = f0_audio[~np.isnan(f0_audio)]
            f0_ref_clean = f0_ref[~np.isnan(f0_ref)]
            
            if len(f0_audio_clean) > 0 and len(f0_ref_clean) > 0:
                pitch_ratio = np.mean(f0_ref_clean) / np.mean(f0_audio_clean)
                
                if 0.8 <= pitch_ratio <= 1.2:
                    n_steps = 12 * np.log2(pitch_ratio)
                    corrected_audio = librosa.effects.pitch_shift(
                        audio, sr=self.sample_rate, n_steps=n_steps * 0.3
                    )
                    return corrected_audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Prosody correction failed: {e}")
            return audio

    def _enhance_voice_characteristics(self, audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """Enhance voice characteristics to match reference."""
        try:
            enhanced_audio = audio.copy()
            
            audio_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            ref_centroid = np.mean(librosa.feature.spectral_centroid(y=reference_audio, sr=self.sample_rate))
            
            if ref_centroid > 0 and audio_centroid > 0:
                tilt_ratio = ref_centroid / audio_centroid
                
                if 0.8 <= tilt_ratio <= 1.2:
                    nyquist = self.sample_rate / 2
                    high_freq = 3000 / nyquist
                    
                    if tilt_ratio > 1.05:
                        b, a = butter(2, high_freq, btype='high')
                        high_freq_component = filtfilt(b, a, enhanced_audio)
                        enhanced_audio += high_freq_component * 0.1 * (tilt_ratio - 1)
                    elif tilt_ratio < 0.95:
                        b, a = butter(2, high_freq, btype='low')
                        enhanced_audio = filtfilt(b, a, enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Voice characteristic enhancement failed: {e}")
            return audio
    
    async def _apply_voice_settings(self, audio: np.ndarray, voice_settings: Dict[str, Any]) -> np.ndarray:
        """Apply voice settings to the final combined audio."""
        try:
            processed_audio = audio.copy()
            
            pitch_shift = voice_settings.get("pitch_shift", 0.0)
            if pitch_shift != 0.0:
                processed_audio = librosa.effects.pitch_shift(
                    processed_audio, sr=self.sample_rate, n_steps=pitch_shift
                )
                logger.info(f"Applied pitch shift: {pitch_shift} semitones")
            
            speed_factor = voice_settings.get("speed_factor", 1.0)
            if speed_factor != 1.0:
                processed_audio = librosa.effects.time_stretch(processed_audio, rate=speed_factor)
                logger.info(f"Applied speed factor: {speed_factor}x")
            
            volume_gain = voice_settings.get("volume_gain", 0.0)
            if volume_gain != 0.0:
                gain_linear = 10 ** (volume_gain / 20.0)
                processed_audio = processed_audio * gain_linear
                if np.max(np.abs(processed_audio)) > 1.0:
                    processed_audio = processed_audio / np.max(np.abs(processed_audio)) * 0.95
                logger.info(f"Applied volume gain: {volume_gain} dB")
            
            emotion_intensity = voice_settings.get("emotion_intensity", 1.0)
            if emotion_intensity != 1.0:
                if emotion_intensity > 1.0:
                    processed_audio = np.sign(processed_audio) * np.power(np.abs(processed_audio), 1.0 / emotion_intensity)
                elif emotion_intensity < 1.0:
                    processed_audio = np.sign(processed_audio) * np.power(np.abs(processed_audio), emotion_intensity)
                logger.info(f"Applied emotion intensity: {emotion_intensity}")
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Failed to apply voice settings: {e}")
            return audio

    async def _save_ensemble_output(self, audio: np.ndarray, voice_profile_id: str) -> str:
        """Save ensemble synthesis output."""
        try:
            timestamp = int(time.time())
            filename = f"ensemble_synthesis_{voice_profile_id}_{timestamp}.wav"
            output_path = os.path.join(settings.RESULTS_DIR, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            sf.write(output_path, audio, self.sample_rate, format='WAV', subtype='PCM_16')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save ensemble output: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [m.value for m in self.loaded_models.keys()]
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models."""
        return {m.value: status for m, status in self.model_load_status.items()}


# Global service instance
ensemble_voice_synthesizer = EnsembleVoiceSynthesizer()


async def initialize_ensemble_synthesis_service():
    """Initialize the ensemble voice synthesis service."""
    try:
        logger.info("Initializing Ensemble Voice Synthesis Service...")
        success = await ensemble_voice_synthesizer.initialize_models()
        logger.info("Ensemble Voice Synthesis Service initialized successfully")
        return success
    except Exception as e:
        logger.error(f"Ensemble synthesis service initialization error: {str(e)}")
        raise RuntimeError(f"Failed to initialize ensemble synthesis service: {str(e)}")
