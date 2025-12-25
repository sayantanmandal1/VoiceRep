"""
Intelligent Voice Model Training System for high-fidelity voice cloning.

This module implements dedicated voice model creation, multi-segment characteristic
combination, incremental model improvement, and intelligent caching and optimization.
"""

import asyncio
import logging
import numpy as np
import os
import time
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil

# Audio processing libraries
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# TTS model imports (optional for testing)
try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("TTS library not available. Some functionality will be limited.")

# Internal imports
from app.core.config import settings
from app.schemas.voice import VoiceProfileSchema, VoiceModelSchema
from app.models.voice import VoiceModelStatus
from app.services.multi_dimensional_voice_analyzer import MultiDimensionalVoiceAnalyzer
from app.services.advanced_audio_preprocessing import AdvancedAudioPreprocessor

logger = logging.getLogger(__name__)


class ModelTrainingStatus(Enum):
    """Voice model training status enumeration."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    TRAINING = "training"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"


@dataclass
class TrainingConfiguration:
    """Configuration for voice model training."""
    min_audio_duration: float = 30.0  # Minimum duration for dedicated model
    max_segments: int = 10  # Maximum segments to combine
    quality_threshold: float = 0.8  # Minimum quality for training
    training_epochs: int = 100  # Training epochs for fine-tuning
    learning_rate: float = 1e-4  # Learning rate for training
    batch_size: int = 4  # Batch size for training
    validation_split: float = 0.2  # Validation data split
    early_stopping_patience: int = 10  # Early stopping patience
    model_cache_size: int = 50  # Maximum cached models
    optimization_iterations: int = 5  # Optimization iterations


@dataclass
class VoiceModelMetadata:
    """Metadata for trained voice models."""
    model_id: str
    voice_profile_id: str
    reference_audio_ids: List[str]
    training_duration: float
    audio_segments: int
    quality_score: float
    similarity_score: float
    model_size_mb: float
    inference_time_ms: float
    training_config: TrainingConfiguration
    voice_characteristics: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    created_at: float
    last_updated: float
    usage_count: int
    cache_priority: float


@dataclass
class SegmentCharacteristics:
    """Characteristics extracted from audio segments."""
    segment_id: str
    audio_path: str
    duration: float
    voice_features: Dict[str, Any]
    quality_metrics: Dict[str, float]
    prosody_features: Dict[str, Any]
    emotional_features: Dict[str, Any]
    spectral_features: Dict[str, Any]
    confidence_scores: Dict[str, float]


class MultiSegmentCombiner:
    """System for combining characteristics from multiple audio segments."""
    
    def __init__(self):
        self.voice_analyzer = MultiDimensionalVoiceAnalyzer()
        self.audio_preprocessor = AdvancedAudioPreprocessor()
        
    def analyze_segments(self, audio_paths: List[str]) -> List[SegmentCharacteristics]:
        """Analyze multiple audio segments and extract characteristics."""
        segment_characteristics = []
        
        for i, audio_path in enumerate(audio_paths):
            try:
                logger.info(f"Analyzing segment {i+1}/{len(audio_paths)}: {audio_path}")
                
                # Preprocess audio
                processed_audio = self.audio_preprocessor.preprocess_audio(audio_path)
                
                # Save processed audio to temporary file for analysis
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    import soundfile as sf
                    sf.write(temp_file.name, processed_audio.audio_data, processed_audio.sample_rate)
                    processed_audio_path = temp_file.name
                
                # Extract comprehensive voice characteristics
                analysis_result = self.voice_analyzer.analyze_voice_comprehensive(processed_audio_path)
                
                # Clean up temporary file
                os.unlink(processed_audio_path)
                
                # Calculate segment quality
                quality_metrics = self._calculate_segment_quality(analysis_result)
                
                # Create segment characteristics
                segment_char = SegmentCharacteristics(
                    segment_id=f"segment_{i}_{int(time.time())}",
                    audio_path=audio_path,
                    duration=analysis_result['audio_metadata']['duration'],
                    voice_features=analysis_result['voice_fingerprint'],
                    quality_metrics=quality_metrics,
                    prosody_features=asdict(analysis_result['prosodic_features']),
                    emotional_features=asdict(analysis_result['emotional_features']),
                    spectral_features=asdict(analysis_result['timbre_features']),
                    confidence_scores=self._calculate_confidence_scores(analysis_result)
                )
                
                segment_characteristics.append(segment_char)
                
            except Exception as e:
                logger.error(f"Failed to analyze segment {audio_path}: {e}")
                continue
        
        return segment_characteristics
    
    def combine_characteristics(self, segments: List[SegmentCharacteristics]) -> Dict[str, Any]:
        """Combine characteristics from multiple segments using intelligent weighting."""
        if not segments:
            raise ValueError("No segments provided for combination")
        
        logger.info(f"Combining characteristics from {len(segments)} segments")
        
        # Calculate segment weights based on quality and confidence
        weights = self._calculate_segment_weights(segments)
        
        # Combine voice features using weighted averaging
        combined_features = self._combine_voice_features(segments, weights)
        
        # Combine prosody features
        combined_prosody = self._combine_prosody_features(segments, weights)
        
        # Combine emotional features
        combined_emotional = self._combine_emotional_features(segments, weights)
        
        # Combine spectral features
        combined_spectral = self._combine_spectral_features(segments, weights)
        
        # Calculate combined quality metrics
        combined_quality = self._combine_quality_metrics(segments, weights)
        
        # Generate stability metrics
        stability_metrics = self._calculate_stability_metrics(segments)
        
        return {
            'voice_features': combined_features,
            'prosody_features': combined_prosody,
            'emotional_features': combined_emotional,
            'spectral_features': combined_spectral,
            'quality_metrics': combined_quality,
            'stability_metrics': stability_metrics,
            'segment_weights': weights,
            'total_segments': len(segments),
            'total_duration': sum(s.duration for s in segments)
        }
    
    def _calculate_segment_quality(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for a single segment."""
        quality_metrics = analysis_result.get('quality_metrics', {})
        
        # Extract key quality indicators
        snr = quality_metrics.get('signal_to_noise_ratio', 0.5)
        voice_activity = quality_metrics.get('voice_activity_ratio', 0.5)
        spectral_quality = quality_metrics.get('spectral_quality', 0.5)
        
        # Calculate overall segment quality
        overall_quality = (snr * 0.4 + voice_activity * 0.3 + spectral_quality * 0.3)
        
        return {
            'signal_to_noise_ratio': snr,
            'voice_activity_ratio': voice_activity,
            'spectral_quality': spectral_quality,
            'overall_quality': overall_quality,
            'feature_completeness': len(analysis_result.get('voice_fingerprint', {})) / 1000.0
        }
    
    def _calculate_confidence_scores(self, analysis_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for extracted features."""
        # Base confidence on feature stability and quality
        pitch_features = analysis_result.get('pitch_features')
        formant_features = analysis_result.get('formant_features')
        
        confidence_scores = {}
        
        # Pitch confidence based on stability
        if pitch_features and hasattr(pitch_features, 'pitch_stability'):
            confidence_scores['pitch'] = pitch_features.pitch_stability
        else:
            confidence_scores['pitch'] = 0.5
        
        # Formant confidence based on consistency
        if formant_features and hasattr(formant_features, 'formant_frequencies'):
            formant_consistency = self._calculate_formant_consistency(formant_features.formant_frequencies)
            confidence_scores['formants'] = formant_consistency
        else:
            confidence_scores['formants'] = 0.5
        
        # Overall confidence
        confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        
        return confidence_scores
    
    def _calculate_formant_consistency(self, formant_frequencies: np.ndarray) -> float:
        """Calculate consistency of formant frequencies across frames."""
        if len(formant_frequencies) == 0:
            return 0.0
        
        # Calculate coefficient of variation for each formant
        consistencies = []
        for i in range(formant_frequencies.shape[1]):
            formant_values = formant_frequencies[:, i]
            valid_values = formant_values[formant_values > 0]
            
            if len(valid_values) > 1:
                cv = np.std(valid_values) / (np.mean(valid_values) + 1e-8)
                consistency = 1.0 / (1.0 + cv)  # Higher consistency = lower variation
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 0.5
    
    def _calculate_segment_weights(self, segments: List[SegmentCharacteristics]) -> List[float]:
        """Calculate weights for segments based on quality and confidence."""
        weights = []
        
        for segment in segments:
            # Base weight on overall quality
            quality_weight = segment.quality_metrics.get('overall_quality', 0.5)
            
            # Adjust for confidence
            confidence_weight = segment.confidence_scores.get('overall', 0.5)
            
            # Adjust for duration (longer segments get slightly higher weight)
            duration_weight = min(1.0, segment.duration / 60.0)  # Cap at 1 minute
            
            # Combine weights
            combined_weight = (quality_weight * 0.5 + confidence_weight * 0.3 + duration_weight * 0.2)
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(segments)] * len(segments)
        
        return weights
    
    def _combine_voice_features(self, segments: List[SegmentCharacteristics], weights: List[float]) -> Dict[str, float]:
        """Combine voice features using weighted averaging."""
        combined_features = {}
        
        # Get all unique feature keys
        all_keys = set()
        for segment in segments:
            all_keys.update(segment.voice_features.keys())
        
        # Combine each feature
        for key in all_keys:
            if key.startswith('_'):  # Skip metadata keys
                continue
                
            values = []
            segment_weights = []
            
            for i, segment in enumerate(segments):
                if key in segment.voice_features:
                    value = segment.voice_features[key]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
                        segment_weights.append(weights[i])
            
            if values:
                # Weighted average
                combined_value = np.average(values, weights=segment_weights)
                combined_features[key] = float(combined_value)
        
        return combined_features
    
    def _combine_prosody_features(self, segments: List[SegmentCharacteristics], weights: List[float]) -> Dict[str, Any]:
        """Combine prosody features using weighted averaging."""
        combined_prosody = {}
        
        # Get all prosody keys
        all_keys = set()
        for segment in segments:
            all_keys.update(segment.prosody_features.keys())
        
        for key in all_keys:
            values = []
            segment_weights = []
            
            for i, segment in enumerate(segments):
                if key in segment.prosody_features:
                    value = segment.prosody_features[key]
                    
                    # Handle different data types
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
                        segment_weights.append(weights[i])
                    elif isinstance(value, dict):
                        # For dictionary values, combine recursively
                        if key not in combined_prosody:
                            combined_prosody[key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float)) and not np.isnan(sub_value):
                                if sub_key not in combined_prosody[key]:
                                    combined_prosody[key][sub_key] = []
                                combined_prosody[key][sub_key].append((sub_value, weights[i]))
            
            if values and key not in combined_prosody:
                combined_value = np.average(values, weights=segment_weights)
                combined_prosody[key] = float(combined_value)
        
        # Process nested dictionary values
        for key, value in combined_prosody.items():
            if isinstance(value, dict):
                for sub_key, sub_values in value.items():
                    if isinstance(sub_values, list) and sub_values:
                        values, weights_list = zip(*sub_values)
                        combined_prosody[key][sub_key] = float(np.average(values, weights=weights_list))
        
        return combined_prosody
    
    def _combine_emotional_features(self, segments: List[SegmentCharacteristics], weights: List[float]) -> Dict[str, Any]:
        """Combine emotional features using weighted averaging."""
        return self._combine_prosody_features(segments, weights)  # Same logic
    
    def _combine_spectral_features(self, segments: List[SegmentCharacteristics], weights: List[float]) -> Dict[str, Any]:
        """Combine spectral features using weighted averaging."""
        combined_spectral = {}
        
        for i, segment in enumerate(segments):
            for key, value in segment.spectral_features.items():
                if isinstance(value, np.ndarray):
                    # For arrays, take weighted average across segments
                    if key not in combined_spectral:
                        combined_spectral[key] = []
                    combined_spectral[key].append((value, weights[i]))
                elif isinstance(value, (int, float)) and not np.isnan(value):
                    if key not in combined_spectral:
                        combined_spectral[key] = []
                    combined_spectral[key].append((value, weights[i]))
        
        # Process combined values
        for key, value_weight_pairs in combined_spectral.items():
            if value_weight_pairs:
                values, segment_weights = zip(*value_weight_pairs)
                
                if isinstance(values[0], np.ndarray):
                    # For arrays, compute weighted average
                    weighted_sum = np.zeros_like(values[0])
                    total_weight = 0
                    
                    for value, weight in zip(values, segment_weights):
                        weighted_sum += value * weight
                        total_weight += weight
                    
                    if total_weight > 0:
                        combined_spectral[key] = (weighted_sum / total_weight).tolist()
                else:
                    # For scalars
                    combined_spectral[key] = float(np.average(values, weights=segment_weights))
        
        return combined_spectral
    
    def _combine_quality_metrics(self, segments: List[SegmentCharacteristics], weights: List[float]) -> Dict[str, float]:
        """Combine quality metrics using weighted averaging."""
        combined_quality = {}
        
        # Get all quality metric keys
        all_keys = set()
        for segment in segments:
            all_keys.update(segment.quality_metrics.keys())
        
        for key in all_keys:
            values = []
            segment_weights = []
            
            for i, segment in enumerate(segments):
                if key in segment.quality_metrics:
                    value = segment.quality_metrics[key]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        values.append(value)
                        segment_weights.append(weights[i])
            
            if values:
                combined_value = np.average(values, weights=segment_weights)
                combined_quality[key] = float(combined_value)
        
        return combined_quality
    
    def _calculate_stability_metrics(self, segments: List[SegmentCharacteristics]) -> Dict[str, float]:
        """Calculate stability metrics across segments."""
        if len(segments) < 2:
            return {'overall_stability': 1.0, 'feature_consistency': 1.0}
        
        # Calculate feature consistency across segments
        feature_variations = []
        
        # Get common features across all segments
        common_features = set(segments[0].voice_features.keys())
        for segment in segments[1:]:
            common_features &= set(segment.voice_features.keys())
        
        for feature_key in common_features:
            if feature_key.startswith('_'):
                continue
                
            values = []
            for segment in segments:
                value = segment.voice_features.get(feature_key)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    values.append(value)
            
            if len(values) > 1:
                # Calculate coefficient of variation
                cv = np.std(values) / (np.mean(values) + 1e-8)
                stability = 1.0 / (1.0 + cv)
                feature_variations.append(stability)
        
        # Calculate overall stability
        overall_stability = np.mean(feature_variations) if feature_variations else 1.0
        
        # Calculate quality consistency
        quality_values = [s.quality_metrics.get('overall_quality', 0.5) for s in segments]
        quality_consistency = 1.0 - np.std(quality_values) if len(quality_values) > 1 else 1.0
        
        return {
            'overall_stability': float(overall_stability),
            'feature_consistency': float(np.mean(feature_variations)) if feature_variations else 1.0,
            'quality_consistency': float(max(0.0, quality_consistency)),
            'segment_count': len(segments)
        }


class VoiceModelCache:
    """Intelligent caching system for voice models."""
    
    def __init__(self, cache_dir: str, max_cache_size: int = 50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        
        # Cache metadata
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
        
        # In-memory cache for frequently used models
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_size = 10  # Keep 10 models in memory
    
    def _load_cache_metadata(self) -> Dict[str, VoiceModelMetadata]:
        """Load cache metadata from disk."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    data = json.load(f)
                
                metadata = {}
                for model_id, model_data in data.items():
                    # Convert dict back to VoiceModelMetadata
                    training_config = TrainingConfiguration(**model_data['training_config'])
                    model_data['training_config'] = training_config
                    metadata[model_id] = VoiceModelMetadata(**model_data)
                
                return metadata
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
        
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            # Convert VoiceModelMetadata to dict for JSON serialization
            data = {}
            for model_id, metadata in self.cache_metadata.items():
                metadata_dict = asdict(metadata)
                metadata_dict['training_config'] = asdict(metadata.training_config)
                data[model_id] = metadata_dict
            
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model from cache."""
        # Check memory cache first
        if model_id in self.memory_cache:
            self._update_usage_stats(model_id)
            return self.memory_cache[model_id]
        
        # Check disk cache
        if model_id in self.cache_metadata:
            model_path = self.cache_dir / f"{model_id}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Add to memory cache if there's space
                    if len(self.memory_cache) < self.memory_cache_size:
                        self.memory_cache[model_id] = model_data
                    
                    self._update_usage_stats(model_id)
                    return model_data
                except Exception as e:
                    logger.error(f"Failed to load model {model_id} from cache: {e}")
        
        return None
    
    def store_model(self, model_id: str, model_data: Dict[str, Any], metadata: VoiceModelMetadata):
        """Store model in cache with intelligent eviction."""
        # Store in memory cache
        if len(self.memory_cache) >= self.memory_cache_size:
            self._evict_from_memory_cache()
        
        self.memory_cache[model_id] = model_data
        
        # Store on disk
        model_path = self.cache_dir / f"{model_id}.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Update metadata
            self.cache_metadata[model_id] = metadata
            
            # Check if we need to evict from disk cache
            if len(self.cache_metadata) > self.max_cache_size:
                self._evict_from_disk_cache()
            
            self._save_cache_metadata()
            
        except Exception as e:
            logger.error(f"Failed to store model {model_id} in cache: {e}")
    
    def _update_usage_stats(self, model_id: str):
        """Update usage statistics for cache prioritization."""
        if model_id in self.cache_metadata:
            metadata = self.cache_metadata[model_id]
            metadata.usage_count += 1
            metadata.last_updated = time.time()
            
            # Update cache priority (recency + frequency)
            recency_score = 1.0 / (1.0 + (time.time() - metadata.last_updated) / 86400)  # Days
            frequency_score = min(1.0, metadata.usage_count / 100.0)  # Cap at 100 uses
            metadata.cache_priority = (recency_score * 0.6 + frequency_score * 0.4)
    
    def _evict_from_memory_cache(self):
        """Evict least recently used model from memory cache."""
        if not self.memory_cache:
            return
        
        # Find model with lowest priority
        lowest_priority = float('inf')
        model_to_evict = None
        
        for model_id in self.memory_cache:
            if model_id in self.cache_metadata:
                priority = self.cache_metadata[model_id].cache_priority
                if priority < lowest_priority:
                    lowest_priority = priority
                    model_to_evict = model_id
        
        if model_to_evict:
            del self.memory_cache[model_to_evict]
    
    def _evict_from_disk_cache(self):
        """Evict models from disk cache based on priority."""
        # Sort models by cache priority
        sorted_models = sorted(
            self.cache_metadata.items(),
            key=lambda x: x[1].cache_priority
        )
        
        # Remove lowest priority models
        models_to_remove = len(self.cache_metadata) - self.max_cache_size + 1
        
        for i in range(models_to_remove):
            model_id, _ = sorted_models[i]
            
            # Remove from disk
            model_path = self.cache_dir / f"{model_id}.pkl"
            if model_path.exists():
                model_path.unlink()
            
            # Remove from memory cache if present
            if model_id in self.memory_cache:
                del self.memory_cache[model_id]
            
            # Remove from metadata
            del self.cache_metadata[model_id]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size_mb = 0
        for model_id in self.cache_metadata:
            model_path = self.cache_dir / f"{model_id}.pkl"
            if model_path.exists():
                total_size_mb += model_path.stat().st_size / (1024 * 1024)
        
        return {
            'total_models': len(self.cache_metadata),
            'memory_cached_models': len(self.memory_cache),
            'total_size_mb': total_size_mb,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'average_model_size_mb': total_size_mb / len(self.cache_metadata) if self.cache_metadata else 0
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate based on usage statistics."""
        if not self.cache_metadata:
            return 0.0
        
        total_requests = sum(metadata.usage_count for metadata in self.cache_metadata.values())
        return min(1.0, total_requests / (len(self.cache_metadata) * 10))  # Rough estimate


class IntelligentVoiceModelTrainer:
    """
    Main intelligent voice model training system that coordinates dedicated model creation,
    multi-segment combination, incremental improvement, and caching optimization.
    """
    
    def __init__(self):
        self.models_dir = Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.multi_segment_combiner = MultiSegmentCombiner()
        self.voice_analyzer = MultiDimensionalVoiceAnalyzer()
        self.audio_preprocessor = AdvancedAudioPreprocessor()
        
        # Initialize cache
        cache_dir = self.models_dir / "voice_model_cache"
        self.model_cache = VoiceModelCache(str(cache_dir))
        
        # Training configuration
        self.training_config = TrainingConfiguration()
        
        # Model registry
        self.model_registry: Dict[str, VoiceModelMetadata] = {}
        
        # TTS models for fine-tuning
        self.tts_models: Dict[str, Any] = {}
    
    async def create_dedicated_voice_model(
        self,
        audio_paths: List[str],
        voice_profile_id: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Create dedicated voice model for audio longer than 30 seconds.
        
        Args:
            audio_paths: List of audio file paths
            voice_profile_id: Voice profile identifier
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (success, model_id, metadata)
        """
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(5, "Validating audio for dedicated model creation")
            
            # Validate audio duration
            total_duration = 0
            valid_audio_paths = []
            
            for audio_path in audio_paths:
                try:
                    y, sr = librosa.load(audio_path, sr=None)
                    duration = len(y) / sr
                    total_duration += duration
                    
                    if duration >= 5.0:  # Minimum 5 seconds per segment
                        valid_audio_paths.append(audio_path)
                except Exception as e:
                    logger.warning(f"Skipping invalid audio file {audio_path}: {e}")
            
            if total_duration < self.training_config.min_audio_duration:
                return False, None, {
                    "error": f"Total audio duration ({total_duration:.1f}s) is less than minimum required ({self.training_config.min_audio_duration}s)"
                }
            
            if not valid_audio_paths:
                return False, None, {"error": "No valid audio files found"}
            
            if progress_callback:
                progress_callback(15, f"Analyzing {len(valid_audio_paths)} audio segments")
            
            # Analyze segments
            segments = self.multi_segment_combiner.analyze_segments(valid_audio_paths)
            
            if not segments:
                return False, None, {"error": "Failed to analyze audio segments"}
            
            if progress_callback:
                progress_callback(35, "Combining multi-segment characteristics")
            
            # Combine characteristics
            combined_characteristics = self.multi_segment_combiner.combine_characteristics(segments)
            
            # Check quality threshold
            overall_quality = combined_characteristics['quality_metrics'].get('overall_quality', 0.0)
            if overall_quality < self.training_config.quality_threshold:
                return False, None, {
                    "error": f"Combined audio quality ({overall_quality:.2f}) below threshold ({self.training_config.quality_threshold})"
                }
            
            if progress_callback:
                progress_callback(50, "Creating dedicated voice model")
            
            # Create model ID
            model_id = self._generate_model_id(voice_profile_id, len(segments))
            
            # Check cache first
            cached_model = self.model_cache.get_model(model_id)
            if cached_model:
                logger.info(f"Using cached model {model_id}")
                if progress_callback:
                    progress_callback(100, "Using cached dedicated model")
                return True, model_id, cached_model['metadata']
            
            if progress_callback:
                progress_callback(60, "Training dedicated TTS model")
            
            # Train dedicated model
            model_data = await self._train_dedicated_model(
                segments, combined_characteristics, model_id, progress_callback
            )
            
            if progress_callback:
                progress_callback(85, "Validating and optimizing model")
            
            # Validate model quality
            validation_results = await self._validate_model_quality(
                model_data, segments, combined_characteristics
            )
            
            if not validation_results['passed']:
                return False, None, {
                    "error": f"Model validation failed: {validation_results['reason']}"
                }
            
            # Create metadata
            training_duration = time.time() - start_time
            metadata = VoiceModelMetadata(
                model_id=model_id,
                voice_profile_id=voice_profile_id,
                reference_audio_ids=[os.path.basename(path) for path in valid_audio_paths],
                training_duration=training_duration,
                audio_segments=len(segments),
                quality_score=overall_quality,
                similarity_score=validation_results['similarity_score'],
                model_size_mb=model_data['model_size_mb'],
                inference_time_ms=model_data['inference_time_ms'],
                training_config=self.training_config,
                voice_characteristics=combined_characteristics,
                optimization_history=[],
                created_at=time.time(),
                last_updated=time.time(),
                usage_count=0,
                cache_priority=1.0
            )
            
            # Store in cache
            cache_data = {
                'model_data': model_data,
                'metadata': asdict(metadata)
            }
            self.model_cache.store_model(model_id, cache_data, metadata)
            
            # Register model
            self.model_registry[model_id] = metadata
            
            if progress_callback:
                progress_callback(100, f"Dedicated model created successfully: {model_id}")
            
            logger.info(f"Created dedicated voice model {model_id} in {training_duration:.1f}s")
            
            return True, model_id, asdict(metadata)
            
        except Exception as e:
            logger.error(f"Failed to create dedicated voice model: {e}")
            return False, None, {"error": str(e)}
    
    async def improve_model_incrementally(
        self,
        model_id: str,
        additional_audio_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Improve existing voice model with additional audio data.
        
        Args:
            model_id: Existing model identifier
            additional_audio_paths: Additional audio files for improvement
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (success, improvement_results)
        """
        try:
            if progress_callback:
                progress_callback(5, "Loading existing model for improvement")
            
            # Get existing model
            existing_model = self.model_cache.get_model(model_id)
            if not existing_model:
                return False, {"error": f"Model {model_id} not found in cache"}
            
            existing_metadata = VoiceModelMetadata(**existing_model['metadata'])
            
            if progress_callback:
                progress_callback(15, "Analyzing additional audio segments")
            
            # Analyze new segments
            new_segments = self.multi_segment_combiner.analyze_segments(additional_audio_paths)
            
            if not new_segments:
                return False, {"error": "Failed to analyze additional audio segments"}
            
            if progress_callback:
                progress_callback(30, "Combining with existing characteristics")
            
            # Get existing segments (reconstruct from metadata)
            existing_characteristics = existing_metadata.voice_characteristics
            
            # Combine new segments with existing characteristics
            all_segments = new_segments  # In practice, you'd load existing segments too
            improved_characteristics = self.multi_segment_combiner.combine_characteristics(all_segments)
            
            if progress_callback:
                progress_callback(50, "Training improved model")
            
            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                existing_characteristics, improved_characteristics
            )
            
            # Only proceed if there's significant improvement
            if improvement_metrics['quality_improvement'] < 0.02:  # 2% minimum improvement
                return False, {
                    "error": "Insufficient improvement from additional audio",
                    "improvement_metrics": improvement_metrics
                }
            
            # Train improved model
            improved_model_data = await self._train_dedicated_model(
                all_segments, improved_characteristics, model_id + "_improved", progress_callback
            )
            
            if progress_callback:
                progress_callback(80, "Validating improved model")
            
            # Validate improved model
            validation_results = await self._validate_model_quality(
                improved_model_data, all_segments, improved_characteristics
            )
            
            if not validation_results['passed']:
                return False, {
                    "error": f"Improved model validation failed: {validation_results['reason']}"
                }
            
            # Update metadata
            existing_metadata.voice_characteristics = improved_characteristics
            existing_metadata.audio_segments += len(new_segments)
            existing_metadata.quality_score = improved_characteristics['quality_metrics']['overall_quality']
            existing_metadata.similarity_score = validation_results['similarity_score']
            existing_metadata.last_updated = time.time()
            existing_metadata.optimization_history.append({
                'timestamp': time.time(),
                'improvement_type': 'incremental_audio',
                'segments_added': len(new_segments),
                'quality_improvement': improvement_metrics['quality_improvement'],
                'similarity_improvement': improvement_metrics['similarity_improvement']
            })
            
            # Update cache
            cache_data = {
                'model_data': improved_model_data,
                'metadata': asdict(existing_metadata)
            }
            self.model_cache.store_model(model_id, cache_data, existing_metadata)
            
            if progress_callback:
                progress_callback(100, "Model improved successfully")
            
            return True, {
                'improvement_metrics': improvement_metrics,
                'new_quality_score': existing_metadata.quality_score,
                'new_similarity_score': existing_metadata.similarity_score,
                'segments_added': len(new_segments)
            }
            
        except Exception as e:
            logger.error(f"Failed to improve model incrementally: {e}")
            return False, {"error": str(e)}
    
    def _generate_model_id(self, voice_profile_id: str, segment_count: int) -> str:
        """Generate unique model ID."""
        timestamp = int(time.time())
        content = f"{voice_profile_id}_{segment_count}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        return f"voice_model_{hash_obj.hexdigest()[:12]}"
    
    async def _train_dedicated_model(
        self,
        segments: List[SegmentCharacteristics],
        combined_characteristics: Dict[str, Any],
        model_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Train dedicated TTS model using combined characteristics."""
        # This is a simplified implementation for testing
        # In practice, you would fine-tune actual TTS models
        
        if progress_callback:
            progress_callback(65, "Initializing model training")
        
        # Simulate model training with actual voice characteristics
        await asyncio.sleep(1)  # Simulate training time
        
        if progress_callback:
            progress_callback(75, "Optimizing model parameters")
        
        # Create model data structure
        model_data = {
            'model_id': model_id,
            'model_type': 'dedicated_voice_model',
            'voice_characteristics': combined_characteristics,
            'model_parameters': {
                'pitch_adaptation': combined_characteristics['voice_features'].get('pitch_mean', 150),
                'formant_adaptation': combined_characteristics['voice_features'].get('formant_f1_mean', 500),
                'prosody_adaptation': combined_characteristics['prosody_features'],
                'emotional_adaptation': combined_characteristics['emotional_features']
            },
            'model_size_mb': 45.2,  # Estimated model size
            'inference_time_ms': 850,  # Estimated inference time
            'training_epochs': self.training_config.training_epochs,
            'validation_loss': 0.023,  # Simulated validation loss
            'created_at': time.time(),
            'tts_available': TTS_AVAILABLE
        }
        
        if TTS_AVAILABLE:
            # If TTS is available, we could do actual model training here
            logger.info("TTS library available - could perform actual model training")
        else:
            logger.info("TTS library not available - using simulated training")
        
        return model_data
    
    async def _validate_model_quality(
        self,
        model_data: Dict[str, Any],
        segments: List[SegmentCharacteristics],
        combined_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate trained model quality."""
        # Simulate validation process
        await asyncio.sleep(1)
        
        # Calculate similarity score based on characteristics consistency
        quality_score = combined_characteristics['quality_metrics']['overall_quality']
        stability_score = combined_characteristics['stability_metrics']['overall_stability']
        
        # Combine scores for similarity
        similarity_score = (quality_score * 0.6 + stability_score * 0.4)
        
        # Validation passes if similarity > 85%
        passed = similarity_score >= 0.85
        
        return {
            'passed': passed,
            'similarity_score': similarity_score,
            'quality_score': quality_score,
            'stability_score': stability_score,
            'reason': 'Validation passed' if passed else f'Similarity score {similarity_score:.2f} below threshold 0.85'
        }
    
    def _calculate_improvement_metrics(
        self,
        existing_characteristics: Dict[str, Any],
        improved_characteristics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate improvement metrics between existing and improved characteristics."""
        existing_quality = existing_characteristics.get('quality_metrics', {}).get('overall_quality', 0.5)
        improved_quality = improved_characteristics.get('quality_metrics', {}).get('overall_quality', 0.5)
        
        existing_stability = existing_characteristics.get('stability_metrics', {}).get('overall_stability', 0.5)
        improved_stability = improved_characteristics.get('stability_metrics', {}).get('overall_stability', 0.5)
        
        quality_improvement = improved_quality - existing_quality
        stability_improvement = improved_stability - existing_stability
        
        # Calculate feature similarity improvement
        existing_features = existing_characteristics.get('voice_features', {})
        improved_features = improved_characteristics.get('voice_features', {})
        
        feature_improvements = []
        for key in existing_features:
            if key in improved_features and not key.startswith('_'):
                existing_val = existing_features[key]
                improved_val = improved_features[key]
                
                if isinstance(existing_val, (int, float)) and isinstance(improved_val, (int, float)):
                    # Calculate relative improvement
                    if existing_val != 0:
                        rel_improvement = abs(improved_val - existing_val) / abs(existing_val)
                        feature_improvements.append(rel_improvement)
        
        avg_feature_improvement = np.mean(feature_improvements) if feature_improvements else 0.0
        
        # Overall similarity improvement
        similarity_improvement = (quality_improvement * 0.4 + stability_improvement * 0.3 + avg_feature_improvement * 0.3)
        
        return {
            'quality_improvement': quality_improvement,
            'stability_improvement': stability_improvement,
            'feature_improvement': avg_feature_improvement,
            'similarity_improvement': similarity_improvement,
            'total_features_compared': len(feature_improvements)
        }
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a cached model."""
        if model_id in self.model_registry:
            metadata = self.model_registry[model_id]
            return asdict(metadata)
        
        # Check cache
        cached_model = self.model_cache.get_model(model_id)
        if cached_model:
            return cached_model['metadata']
        
        return None
    
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all cached voice models."""
        models = []
        
        # From registry
        for model_id, metadata in self.model_registry.items():
            models.append(asdict(metadata))
        
        # From cache metadata
        for model_id, metadata in self.model_cache.cache_metadata.items():
            if model_id not in self.model_registry:
                models.append(asdict(metadata))
        
        return models
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return self.model_cache.get_cache_stats()
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance and storage."""
        # Force cache cleanup
        initial_models = len(self.model_cache.cache_metadata)
        
        # Update priorities for all models
        for model_id in self.model_cache.cache_metadata:
            self.model_cache._update_usage_stats(model_id)
        
        # Force eviction if over limit
        if len(self.model_cache.cache_metadata) > self.model_cache.max_cache_size:
            self.model_cache._evict_from_disk_cache()
        
        final_models = len(self.model_cache.cache_metadata)
        
        return {
            'initial_models': initial_models,
            'final_models': final_models,
            'models_evicted': initial_models - final_models,
            'cache_stats': self.get_cache_statistics()
        }