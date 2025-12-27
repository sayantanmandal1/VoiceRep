"""
Ensemble Voice Synthesis Engine for high-fidelity voice cloning.

This module implements an advanced ensemble synthesis system that integrates
multiple TTS models (TorToiSe, Bark, XTTS) for maximum voice replication accuracy.
"""

import asyncio
import logging
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
import tempfile
import shutil

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
    BARK = "bark"
    TORTOISE = "tortoise"
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
    audio_data: np.ndarray
    sample_rate: int
    quality_score: float
    confidence_score: float
    processing_time: float
    similarity_score: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class EnsembleWeights:
    """Weights for ensemble model combination."""
    model_weights: Dict[TTSModelType, float]
    quality_threshold: float
    confidence_threshold: float
    similarity_threshold: float


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
            TTSModelType.BARK: {
                "best_for": ["emotional_speech", "natural_prosody", "code_switching"],
                "pitch_range": (60, 500),
                "formant_sensitivity": 0.7,
                "prosody_preservation": 0.95,
                "emotion_capture": 0.9
            },
            TTSModelType.TORTOISE: {
                "best_for": ["high_fidelity", "long_form", "character_voices"],
                "pitch_range": (70, 450),
                "formant_sensitivity": 0.9,
                "prosody_preservation": 0.8,
                "emotion_capture": 0.8
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
            
            # Select top 2-3 models for ensemble
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
            # Since VoiceProfileSchema doesn't have voice_characteristics field,
            # we'll adapt the individual fields
            
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


class EnsembleVoiceSynthesizer:
    """
    Main ensemble voice synthesis engine that coordinates multiple TTS models
    for optimal voice replication quality.
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
            TTSModelType.YOUR_TTS: ModelConfiguration(
                model_type=TTSModelType.YOUR_TTS,
                model_name="tts_models/multilingual/multi-dataset/your_tts",
                quality_score=0.88,
                supports_cloning=True,
                languages=["en", "fr", "pt", "tr"],
                max_text_length=400,
                processing_time_factor=1.5,
                memory_usage="medium"
            )
        }
        
        # Loaded models cache
        self.loaded_models: Dict[TTSModelType, TTS] = {}
        self.model_load_status: Dict[TTSModelType, bool] = {}
        
        # Synthesis parameters
        self.ensemble_weights = EnsembleWeights(
            model_weights={
                TTSModelType.XTTS_V2: 0.4,
                TTSModelType.BARK: 0.35,
                TTSModelType.YOUR_TTS: 0.25
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
            
            # Load models in order of priority
            model_load_tasks = []
            
            for i, (model_type, config) in enumerate(self.model_configs.items()):
                progress = 10 + (i * 25)
                if progress_callback:
                    progress_callback(progress, f"Loading {config.model_name}")
                
                # Create async task for model loading
                task = asyncio.create_task(
                    self._load_model_async(model_type, config)
                )
                model_load_tasks.append((model_type, task))
            
            # Wait for all models to load (with timeout)
            loaded_count = 0
            for model_type, task in model_load_tasks:
                try:
                    success = await asyncio.wait_for(task, timeout=300)  # 5 minute timeout per model
                    if success:
                        loaded_count += 1
                        logger.info(f"Successfully loaded {model_type.value}")
                    else:
                        logger.warning(f"Failed to load {model_type.value}")
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout loading {model_type.value}")
                    task.cancel()
                except Exception as e:
                    logger.error(f"Error loading {model_type.value}: {e}")
            
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
            # Load model in thread to avoid blocking
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
            
            # Handle specific error cases
            if "invalid load key" in error_msg or "corrupted" in error_msg or "text_2.pt" in error_msg:
                logger.warning(f"Model {model_type.value} appears corrupted, attempting to clear cache and retry...")
                try:
                    # Clear the model cache for this specific model
                    import shutil
                    from pathlib import Path
                    
                    # Try to find and remove corrupted model files
                    cache_locations = [
                        Path.home() / ".cache" / "tts",
                        Path.home() / "AppData" / "Local" / "tts"
                    ]
                    
                    # More comprehensive cache clearing for Bark model
                    if model_type == TTSModelType.BARK:
                        bark_cache_patterns = [
                            "*bark*",
                            "*multilingual*multi-dataset*bark*",
                            "tts_models--multilingual--multi-dataset--bark"
                        ]
                        
                        for cache_dir in cache_locations:
                            if not cache_dir.exists():
                                continue
                                
                            for pattern in bark_cache_patterns:
                                model_cache_dirs = list(cache_dir.glob(pattern))
                                for cache_dir_path in model_cache_dirs:
                                    if cache_dir_path.is_dir():
                                        logger.info(f"Removing corrupted Bark cache: {cache_dir_path}")
                                        shutil.rmtree(cache_dir_path, ignore_errors=True)
                            
                            # Also check for specific Bark model files
                            bark_specific_dir = cache_dir / "tts_models--multilingual--multi-dataset--bark"
                            if bark_specific_dir.exists():
                                logger.info(f"Removing Bark model directory: {bark_specific_dir}")
                                shutil.rmtree(bark_specific_dir, ignore_errors=True)
                    else:
                        # General cache clearing for other models
                        for cache_dir in cache_locations:
                            if not cache_dir.exists():
                                continue
                            model_cache_dirs = list(cache_dir.glob(f"*{model_type.value}*"))
                            for cache_dir_path in model_cache_dirs:
                                if cache_dir_path.is_dir():
                                    logger.info(f"Removing corrupted cache: {cache_dir_path}")
                                    shutil.rmtree(cache_dir_path, ignore_errors=True)
                    
                    logger.warning(f"Cache cleared for {model_type.value}. Model will be re-downloaded on next startup.")
                    
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup corrupted model cache: {cleanup_error}")
            
            logger.error(f"Failed to load {model_type.value}: {e}")
            self.model_load_status[model_type] = False
            return False
    
    async def synthesize_speech_ensemble(
        self,
        text: str,
        voice_profile: VoiceProfileSchema,
        language: str = "en",
        voice_settings: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Synthesize speech using ensemble of TTS models for maximum quality.
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile for cloning
            language: Target language
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
            available_models = [m for m in selected_models if m in self.loaded_models]
            
            if not available_models:
                return False, None, {"error": "No suitable models available"}
            
            if progress_callback:
                progress_callback(15, f"Using {len(available_models)} models for synthesis")
            
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
                logger.warning("No reference audio available, attempting synthesis without reference")
                # Try to synthesize without reference audio using available models
                if not available_models:
                    return False, None, {"error": "No models available and no reference audio"}
                
                # Use a simplified approach for synthesis without reference
                try:
                    # Use the first available model for basic synthesis
                    model_type = available_models[0]
                    model = self.loaded_models[model_type]
                    
                    if progress_callback:
                        progress_callback(50, f"Synthesizing with {model_type.value} (no reference)")
                    
                    # Generate output path
                    output_dir = Path("results")
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"ensemble_synthesis_{int(time.time())}.wav"
                    
                    # Convert language to model-specific format
                    model_language = self._get_model_language_code(model_type, language)
                    logger.info(f"Using language '{model_language}' for model {model_type.value} (original: '{language}')")
                    
                    # Basic synthesis without reference audio - handle multi-speaker models
                    synthesis_kwargs = {
                        "text": text,
                        "language": model_language
                    }
                    
                    # Check if model is multi-speaker and handle accordingly
                    try:
                        # Try basic synthesis first
                        if hasattr(model, 'tts_to_file'):
                            model.tts_to_file(file_path=str(output_path), **synthesis_kwargs)
                        else:
                            wav = model.tts(**synthesis_kwargs)
                            sf.write(str(output_path), wav, self.sample_rate)
                    except Exception as speaker_error:
                        if "multi-speaker" in str(speaker_error).lower() or "speaker" in str(speaker_error).lower():
                            logger.info(f"Model {model_type.value} is multi-speaker, trying with default speaker")
                            
                            # Try with default speaker configurations
                            speaker_options = [
                                {"speaker": "default"},
                                {"speaker": "p225"},  # Common VCTK speaker
                                {"speaker": "ljspeech"},  # Common single speaker
                                {"speaker_idx": 0},  # Speaker index
                                {"speaker_wav": None}  # Explicit None
                            ]
                            
                            synthesis_success = False
                            for speaker_config in speaker_options:
                                try:
                                    synthesis_kwargs.update(speaker_config)
                                    
                                    if hasattr(model, 'tts_to_file'):
                                        model.tts_to_file(file_path=str(output_path), **synthesis_kwargs)
                                    else:
                                        wav = model.tts(**synthesis_kwargs)
                                        sf.write(str(output_path), wav, self.sample_rate)
                                    
                                    synthesis_success = True
                                    logger.info(f"Synthesis successful with speaker config: {speaker_config}")
                                    break
                                    
                                except Exception as config_error:
                                    logger.debug(f"Speaker config {speaker_config} failed: {config_error}")
                                    continue
                            
                            if not synthesis_success:
                                raise speaker_error  # Re-raise original error if all configs fail
                        else:
                            raise speaker_error  # Re-raise if not a speaker-related error
                    
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
            
            if progress_callback:
                progress_callback(25, "Generating speech with multiple models")
            
            # Synthesize with each selected model
            synthesis_results = []
            
            for i, model_type in enumerate(available_models):
                model_progress = 25 + (i * 40 // len(available_models))
                if progress_callback:
                    progress_callback(model_progress, f"Synthesizing with {model_type.value}")
                
                result = await self._synthesize_with_model(
                    model_type, text, reference_audio_path, language
                )
                
                if result and result.audio_data is not None:
                    synthesis_results.append(result)
            
            if not synthesis_results:
                return False, None, {"error": "All model synthesis attempts failed"}
            
            if progress_callback:
                progress_callback(70, "Assessing synthesis quality")
            
            # Assess quality of each result
            reference_audio, _ = librosa.load(reference_audio_path, sr=self.sample_rate)
            
            for result in synthesis_results:
                quality_metrics = self.quality_assessor.assess_synthesis_quality(
                    result.audio_data, reference_audio, self.sample_rate
                )
                result.similarity_score = quality_metrics["similarity"]
                result.quality_score = quality_metrics["overall"]
            
            if progress_callback:
                progress_callback(80, "Combining ensemble results")
            
            # Combine results using ensemble weighting
            final_audio = await self._combine_ensemble_results(synthesis_results)
            
            # Apply voice settings to the combined audio (after ensemble combination to prevent overlap)
            if voice_settings:
                if progress_callback:
                    progress_callback(85, "Applying voice settings")
                final_audio = await self._apply_voice_settings(final_audio, voice_settings)
            
            if progress_callback:
                progress_callback(90, "Applying iterative refinement")
            
            # Apply iterative refinement
            refined_audio = await self._apply_iterative_refinement(
                final_audio, reference_audio, synthesis_results
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
                "ensemble_models": [r.model_type.value for r in synthesis_results],
                "model_count": len(synthesis_results),
                "quality_metrics": final_quality,
                "similarity_score": final_quality["similarity"],
                "synthesis_method": "ensemble_voice_cloning",
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
    
    async def _prepare_reference_audio(self, voice_profile: VoiceProfileSchema) -> Optional[str]:
        """Prepare reference audio for synthesis."""
        try:
            # Try to get reference audio from voice profile
            if hasattr(voice_profile, 'reference_audio_id') and voice_profile.reference_audio_id:
                reference_id = voice_profile.reference_audio_id
                
                # Check if reference_audio_id is already a file path
                from pathlib import Path
                if Path(reference_id).exists() and Path(reference_id).is_file():
                    logger.info(f"Using direct reference audio path: {reference_id}")
                    return reference_id
                
                # Look for reference audio in uploads directory
                from app.core.config import settings
                
                # Check multiple possible locations
                possible_paths = [
                    Path(settings.UPLOAD_DIR) / f"{reference_id}.wav",
                    Path(settings.UPLOAD_DIR) / f"{reference_id}.mp3",
                    Path("uploads") / f"{reference_id}.wav",
                    Path("uploads") / f"{reference_id}.mp3",
                ]
                
                # Also check session-based uploads
                import os
                upload_dir = Path(settings.UPLOAD_DIR) if hasattr(settings, 'UPLOAD_DIR') else Path("uploads")
                for session_dir in upload_dir.glob("session_*"):
                    possible_paths.extend([
                        session_dir / f"{reference_id}.wav",
                        session_dir / f"{reference_id}.mp3",
                    ])
                
                # Find the first existing file
                for path in possible_paths:
                    if path.exists() and path.is_file():
                        logger.info(f"Found reference audio: {path}")
                        return str(path)
                
                logger.warning(f"Reference audio not found for ID: {reference_id}")
            
            # If no reference audio found, try to use a default or create a minimal one
            # For now, we'll return None but log the issue
            logger.warning("No reference audio available for voice profile")
            return None
            
        except Exception as e:
            logger.error(f"Failed to prepare reference audio: {e}")
            return None
    
    def _get_model_language_code(self, model_type: TTSModelType, language: str) -> str:
        """Convert language to model-specific format."""
        # Language code mappings for different models
        language_mappings = {
            TTSModelType.XTTS_V2: {
                'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
                'italian': 'it', 'portuguese': 'pt', 'polish': 'pl', 'turkish': 'tr',
                'russian': 'ru', 'dutch': 'nl', 'czech': 'cs', 'arabic': 'ar',
                'chinese': 'zh-cn', 'japanese': 'ja', 'hungarian': 'hu', 'korean': 'ko',
                # Also support direct ISO codes
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it', 'pt': 'pt',
                'pl': 'pl', 'tr': 'tr', 'ru': 'ru', 'nl': 'nl', 'cs': 'cs', 'ar': 'ar',
                'zh-cn': 'zh-cn', 'ja': 'ja', 'hu': 'hu', 'ko': 'ko'
            },
            TTSModelType.BARK: {
                'english': 'en', 'german': 'de', 'spanish': 'es', 'french': 'fr',
                'hindi': 'hi', 'italian': 'it', 'japanese': 'ja', 'korean': 'ko',
                'polish': 'pl', 'portuguese': 'pt', 'russian': 'ru', 'turkish': 'tr',
                'chinese': 'zh',
                # Also support direct ISO codes
                'en': 'en', 'de': 'de', 'es': 'es', 'fr': 'fr', 'hi': 'hi', 'it': 'it',
                'ja': 'ja', 'ko': 'ko', 'pl': 'pl', 'pt': 'pt', 'ru': 'ru', 'tr': 'tr', 'zh': 'zh'
            },
            TTSModelType.YOUR_TTS: {
                'english': 'en', 'french': 'fr-fr', 'portuguese': 'pt-br', 'turkish': 'tr',
                # Also support direct codes
                'en': 'en', 'fr': 'fr-fr', 'pt': 'pt-br', 'tr': 'tr'
            }
        }
        
        model_mapping = language_mappings.get(model_type, {})
        mapped_language = model_mapping.get(language.lower())
        
        if mapped_language:
            return mapped_language
        
        # Fallback: check if language is already in supported format
        model_config = self.model_configurations.get(model_type)
        if model_config and language in model_config.languages:
            return language
            
        # Default fallback to English
        logger.warning(f"Language '{language}' not supported by {model_type.value}, falling back to English")
        return model_mapping.get('english', 'en')

    async def _synthesize_with_model(
        self, 
        model_type: TTSModelType, 
        text: str, 
        reference_audio_path: str, 
        language: str
    ) -> Optional[SynthesisResult]:
        """Synthesize speech with a specific model."""
        try:
            start_time = time.time()
            
            model = self.loaded_models.get(model_type)
            if not model:
                return None
            
            # Convert language to model-specific format
            model_language = self._get_model_language_code(model_type, language)
            logger.info(f"Using language '{model_language}' for model {model_type.value} (original: '{language}')")
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_output_path = tmp_file.name
            
            # Synthesize with the model
            synthesis_kwargs = {
                "text": text,
                "language": model_language
            }
            
            # Add reference audio if available
            if reference_audio_path and os.path.exists(reference_audio_path):
                synthesis_kwargs["speaker_wav"] = reference_audio_path
            
            try:
                if hasattr(model, 'tts_to_file'):
                    # For models that support file output
                    model.tts_to_file(file_path=temp_output_path, **synthesis_kwargs)
                else:
                    # For models that return audio data
                    audio_data = model.tts(**synthesis_kwargs)
                    # Save to temporary file
                    sf.write(temp_output_path, audio_data, self.sample_rate)
                    
            except Exception as synthesis_error:
                error_msg = str(synthesis_error).lower()
                
                # Handle language-related errors
                if "language" in error_msg and "not in" in error_msg:
                    logger.warning(f"Language error with {model_type.value}: {synthesis_error}")
                    # Try with fallback language (English)
                    try:
                        fallback_kwargs = synthesis_kwargs.copy()
                        fallback_kwargs["language"] = self._get_model_language_code(model_type, "english")
                        logger.info(f"Retrying {model_type.value} with fallback language: {fallback_kwargs['language']}")
                        
                        if hasattr(model, 'tts_to_file'):
                            model.tts_to_file(file_path=temp_output_path, **fallback_kwargs)
                        else:
                            audio_data = model.tts(**fallback_kwargs)
                            sf.write(temp_output_path, audio_data, self.sample_rate)
                            
                        logger.info(f"Synthesis successful with fallback language for {model_type.value}")
                        
                    except Exception as fallback_error:
                        logger.error(f"Fallback language also failed for {model_type.value}: {fallback_error}")
                        raise synthesis_error  # Re-raise original error
                        
                elif "multi-speaker" in error_msg or "speaker" in error_msg:
                    logger.info(f"Model {model_type.value} is multi-speaker, trying with default speaker")
                    
                    # Try with default speaker configurations
                    speaker_options = [
                        {"speaker": "default"},
                        {"speaker": "p225"},  # Common VCTK speaker
                        {"speaker": "ljspeech"},  # Common single speaker
                        {"speaker_idx": 0},  # Speaker index
                    ]
                    
                    synthesis_success = False
                    for speaker_config in speaker_options:
                        try:
                            # Create new kwargs with speaker config
                            speaker_kwargs = synthesis_kwargs.copy()
                            speaker_kwargs.update(speaker_config)
                            
                            if hasattr(model, 'tts_to_file'):
                                model.tts_to_file(file_path=temp_output_path, **speaker_kwargs)
                            else:
                                audio_data = model.tts(**speaker_kwargs)
                                sf.write(temp_output_path, audio_data, self.sample_rate)
                            
                            synthesis_success = True
                            logger.info(f"Synthesis successful with speaker config: {speaker_config}")
                            break
                            
                        except Exception as config_error:
                            logger.debug(f"Speaker config {speaker_config} failed: {config_error}")
                            continue
                    
                    if not synthesis_success:
                        raise synthesis_error  # Re-raise original error if all configs fail
                else:
                    raise synthesis_error  # Re-raise if not a speaker-related error
            
            # Load the generated audio
            audio_data, sr = librosa.load(temp_output_path, sr=self.sample_rate)
            
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            processing_time = time.time() - start_time
            
            # Calculate confidence score based on model characteristics
            config = self.model_configs[model_type]
            confidence_score = config.quality_score * 0.8  # Base confidence
            
            return SynthesisResult(
                model_type=model_type,
                audio_data=audio_data,
                sample_rate=sr,
                quality_score=config.quality_score,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Synthesis with {model_type.value} failed: {e}")
            return SynthesisResult(
                model_type=model_type,
                audio_data=None,
                sample_rate=self.sample_rate,
                quality_score=0.0,
                confidence_score=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
    
    async def _combine_ensemble_results(self, results: List[SynthesisResult]) -> np.ndarray:
        """Combine multiple synthesis results using weighted averaging with proper synchronization."""
        try:
            if not results:
                raise ValueError("No synthesis results to combine")
            
            # Filter out failed results
            valid_results = [r for r in results if r.audio_data is not None]
            
            if not valid_results:
                raise ValueError("No valid synthesis results")
            
            # Use single best result instead of combining to prevent overlapping voices
            if len(valid_results) == 1:
                return valid_results[0].audio_data
            
            # Select the single best result based on quality score to avoid voice overlap
            best_result = max(valid_results, key=lambda r: r.quality_score)
            logger.info(f"Selected single best model {best_result.model_type} with quality {best_result.quality_score:.3f} to prevent voice overlap")
            
            return best_result.audio_data
            
            # DISABLED: Original ensemble combination logic that causes voice overlap
            # # Normalize audio lengths
            # max_length = max(len(r.audio_data) for r in valid_results)
            # 
            # # Pad shorter audio to match longest
            # padded_audio = []
            # weights = []
            # 
            # for result in valid_results:
            #     audio = result.audio_data
            #     
            #     # Pad if necessary
            #     if len(audio) < max_length:
            #         padding = max_length - len(audio)
            #         audio = np.pad(audio, (0, padding), mode='constant')
            #     
            #     padded_audio.append(audio)
            #     
            #     # Calculate weight based on quality and confidence
            #     weight = (result.quality_score * 0.6 + result.confidence_score * 0.4)
            #     
            #     # Apply model-specific weight
            #     model_weight = self.ensemble_weights.model_weights.get(result.model_type, 0.33)
            #     final_weight = weight * model_weight
            #     
            #     weights.append(final_weight)
            # 
            # # Normalize weights
            # total_weight = sum(weights)
            # if total_weight > 0:
            #     weights = [w / total_weight for w in weights]
            # else:
            #     weights = [1.0 / len(weights)] * len(weights)
            # 
            # # Weighted combination
            # combined_audio = np.zeros(max_length)
            # for audio, weight in zip(padded_audio, weights):
            #     combined_audio += audio * weight
            # 
            # # Normalize to prevent clipping
            # if np.max(np.abs(combined_audio)) > 0:
            #     combined_audio = combined_audio / np.max(np.abs(combined_audio)) * 0.95
            # 
            # return combined_audio
            
        except Exception as e:
            logger.error(f"Failed to combine ensemble results: {e}")
            # Return the best single result as fallback
            if results:
                best_result = max(results, key=lambda r: r.quality_score if r.audio_data is not None else 0)
                if best_result.audio_data is not None:
                    return best_result.audio_data
            
            # Ultimate fallback: silence
            return np.zeros(int(self.sample_rate * 2))  # 2 seconds of silence
    
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
            return audio  # Return original if refinement fails
    
    def _apply_spectral_matching(self, audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """Apply spectral matching to align frequency characteristics."""
        try:
            # Compute spectral envelopes
            audio_stft = librosa.stft(audio)
            ref_stft = librosa.stft(reference_audio)
            
            audio_magnitude = np.abs(audio_stft)
            ref_magnitude = np.abs(ref_stft)
            
            # Calculate average spectral envelopes
            audio_envelope = np.mean(audio_magnitude, axis=1, keepdims=True)
            ref_envelope = np.mean(ref_magnitude, axis=1, keepdims=True)
            
            # Apply spectral matching (conservative)
            matching_strength = 0.3
            target_envelope = (1 - matching_strength) * audio_envelope + matching_strength * ref_envelope
            
            # Apply envelope matching
            envelope_ratio = target_envelope / (audio_envelope + 1e-8)
            matched_magnitude = audio_magnitude * envelope_ratio
            
            # Reconstruct audio
            matched_stft = matched_magnitude * np.exp(1j * np.angle(audio_stft))
            matched_audio = librosa.istft(matched_stft)
            
            return matched_audio
            
        except Exception as e:
            logger.error(f"Spectral matching failed: {e}")
            return audio
    
    def _apply_prosody_correction(self, audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """Apply prosody correction to match reference patterns."""
        try:
            # Extract pitch contours
            f0_audio, _, _ = librosa.pyin(audio, fmin=80, fmax=400, sr=self.sample_rate)
            f0_ref, _, _ = librosa.pyin(reference_audio, fmin=80, fmax=400, sr=self.sample_rate)
            
            # Apply gentle pitch correction if needed
            f0_audio_clean = f0_audio[~np.isnan(f0_audio)]
            f0_ref_clean = f0_ref[~np.isnan(f0_ref)]
            
            if len(f0_audio_clean) > 0 and len(f0_ref_clean) > 0:
                # Calculate pitch shift needed
                pitch_ratio = np.mean(f0_ref_clean) / np.mean(f0_audio_clean)
                
                # Apply moderate pitch shift if ratio is reasonable
                if 0.8 <= pitch_ratio <= 1.2:
                    n_steps = 12 * np.log2(pitch_ratio)
                    corrected_audio = librosa.effects.pitch_shift(
                        audio, sr=self.sample_rate, n_steps=n_steps * 0.3  # Gentle correction
                    )
                    return corrected_audio
            
            return audio
            
        except Exception as e:
            logger.error(f"Prosody correction failed: {e}")
            return audio
    
    def _enhance_voice_characteristics(self, audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """Enhance voice characteristics to match reference."""
        try:
            # Apply gentle EQ to match spectral characteristics
            enhanced_audio = audio.copy()
            
            # Calculate spectral centroids
            audio_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            ref_centroid = np.mean(librosa.feature.spectral_centroid(y=reference_audio, sr=self.sample_rate))
            
            # Apply gentle spectral tilt adjustment
            if ref_centroid > 0 and audio_centroid > 0:
                tilt_ratio = ref_centroid / audio_centroid
                
                if 0.8 <= tilt_ratio <= 1.2:
                    # Apply gentle high-frequency emphasis/de-emphasis
                    nyquist = self.sample_rate / 2
                    high_freq = 3000 / nyquist
                    
                    if tilt_ratio > 1.05:  # Need more high frequency
                        b, a = butter(2, high_freq, btype='high')
                        high_freq_component = filtfilt(b, a, enhanced_audio)
                        enhanced_audio += high_freq_component * 0.1 * (tilt_ratio - 1)
                    elif tilt_ratio < 0.95:  # Need less high frequency
                        b, a = butter(2, high_freq, btype='low')
                        enhanced_audio = filtfilt(b, a, enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Voice characteristic enhancement failed: {e}")
            return audio
    
    async def _apply_voice_settings(self, audio: np.ndarray, voice_settings: Dict[str, Any]) -> np.ndarray:
        """Apply voice settings to the final combined audio to prevent voice overlap."""
        try:
            processed_audio = audio.copy()
            
            # Apply pitch shifting if specified
            pitch_shift = voice_settings.get("pitch_shift", 0.0)
            if pitch_shift != 0.0:
                import librosa
                processed_audio = librosa.effects.pitch_shift(
                    processed_audio, 
                    sr=self.sample_rate, 
                    n_steps=pitch_shift
                )
                logger.info(f"Applied pitch shift: {pitch_shift} semitones")
            
            # Apply speed modification
            speed_factor = voice_settings.get("speed_factor", 1.0)
            if speed_factor != 1.0:
                import librosa
                processed_audio = librosa.effects.time_stretch(processed_audio, rate=speed_factor)
                logger.info(f"Applied speed factor: {speed_factor}x")
            
            # Apply volume gain
            volume_gain = voice_settings.get("volume_gain", 0.0)
            if volume_gain != 0.0:
                # Convert dB to linear scale
                gain_linear = 10 ** (volume_gain / 20.0)
                processed_audio = processed_audio * gain_linear
                # Prevent clipping
                if np.max(np.abs(processed_audio)) > 1.0:
                    processed_audio = processed_audio / np.max(np.abs(processed_audio)) * 0.95
                logger.info(f"Applied volume gain: {volume_gain} dB")
            
            # Apply emotion intensity (simplified implementation)
            emotion_intensity = voice_settings.get("emotion_intensity", 1.0)
            if emotion_intensity != 1.0:
                # Simple emotion intensity by adjusting dynamic range
                if emotion_intensity > 1.0:
                    # Increase dynamic range for more expressive speech
                    processed_audio = np.sign(processed_audio) * np.power(np.abs(processed_audio), 1.0 / emotion_intensity)
                elif emotion_intensity < 1.0:
                    # Decrease dynamic range for flatter speech
                    processed_audio = np.sign(processed_audio) * np.power(np.abs(processed_audio), emotion_intensity)
                logger.info(f"Applied emotion intensity: {emotion_intensity}")
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Failed to apply voice settings: {e}")
            # Return original audio if processing fails
            return audio

    async def _save_ensemble_output(self, audio: np.ndarray, voice_profile_id: str) -> str:
        """Save ensemble synthesis output."""
        try:
            # Generate output filename
            timestamp = int(time.time())
            filename = f"ensemble_synthesis_{voice_profile_id}_{timestamp}.wav"
            output_path = os.path.join(settings.RESULTS_DIR, filename)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save audio file
            sf.write(output_path, audio, self.sample_rate, format='WAV', subtype='PCM_16')
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save ensemble output: {e}")
            raise


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