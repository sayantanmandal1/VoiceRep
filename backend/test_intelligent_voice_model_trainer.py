"""
Unit tests for Intelligent Voice Model Training System.
"""

import pytest
import asyncio
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

from app.services.intelligent_voice_model_trainer import (
    IntelligentVoiceModelTrainer,
    MultiSegmentCombiner,
    VoiceModelCache,
    TrainingConfiguration,
    VoiceModelMetadata,
    SegmentCharacteristics
)


class TestMultiSegmentCombiner:
    """Test multi-segment characteristic combination system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.combiner = MultiSegmentCombiner()
    
    @patch('app.services.intelligent_voice_model_trainer.librosa.load')
    @patch.object(MultiSegmentCombiner, '_calculate_segment_quality')
    @patch.object(MultiSegmentCombiner, '_calculate_confidence_scores')
    def test_analyze_segments_success(self, mock_confidence, mock_quality, mock_load):
        """Test successful analysis of multiple audio segments."""
        # Mock audio loading
        mock_load.return_value = (np.random.randn(22050), 22050)  # 1 second audio
        
        # Mock quality and confidence calculations
        mock_quality.return_value = {
            'signal_to_noise_ratio': 0.8,
            'voice_activity_ratio': 0.9,
            'spectral_quality': 0.85,
            'overall_quality': 0.85,
            'feature_completeness': 0.9
        }
        
        mock_confidence.return_value = {
            'pitch': 0.8,
            'formants': 0.85,
            'overall': 0.825
        }
        
        # Mock voice analyzer
        with patch.object(self.combiner.voice_analyzer, 'analyze_voice_comprehensive') as mock_analyze:
            mock_analyze.return_value = {
                'voice_fingerprint': {'feature_1': 0.5, 'feature_2': 0.7},
                'prosodic_features': Mock(rhythm_patterns={'regularity': 0.8}),
                'emotional_features': Mock(emotional_dimensions={'valence': 0.6}),
                'timbre_features': Mock(spectral_centroid=np.array([1000, 1100])),
                'audio_metadata': {'duration': 1.0}
            }
            
            # Mock audio preprocessor
            with patch.object(self.combiner.audio_preprocessor, 'preprocess_audio') as mock_preprocess:
                mock_preprocess.return_value = "processed_audio.wav"
                
                # Test segment analysis
                audio_paths = ["test1.wav", "test2.wav"]
                segments = self.combiner.analyze_segments(audio_paths)
                
                assert len(segments) == 2
                assert all(isinstance(seg, SegmentCharacteristics) for seg in segments)
                assert segments[0].duration == 1.0
                assert segments[0].quality_metrics['overall_quality'] == 0.85
    
    def test_combine_characteristics_empty_segments(self):
        """Test combining characteristics with empty segments list."""
        with pytest.raises(ValueError, match="No segments provided"):
            self.combiner.combine_characteristics([])
    
    def test_calculate_segment_weights(self):
        """Test segment weight calculation based on quality and confidence."""
        # Create mock segments
        segments = [
            Mock(
                quality_metrics={'overall_quality': 0.9},
                confidence_scores={'overall': 0.8},
                duration=60.0
            ),
            Mock(
                quality_metrics={'overall_quality': 0.7},
                confidence_scores={'overall': 0.6},
                duration=30.0
            )
        ]
        
        weights = self.combiner._calculate_segment_weights(segments)
        
        assert len(weights) == 2
        assert sum(weights) == pytest.approx(1.0, rel=1e-6)
        assert weights[0] > weights[1]  # Higher quality segment should have higher weight
    
    def test_combine_voice_features(self):
        """Test voice feature combination using weighted averaging."""
        segments = [
            Mock(voice_features={'feature_1': 0.8, 'feature_2': 0.6}),
            Mock(voice_features={'feature_1': 0.6, 'feature_2': 0.8})
        ]
        weights = [0.6, 0.4]
        
        combined = self.combiner._combine_voice_features(segments, weights)
        
        expected_feature_1 = 0.8 * 0.6 + 0.6 * 0.4  # 0.72
        expected_feature_2 = 0.6 * 0.6 + 0.8 * 0.4  # 0.68
        
        assert combined['feature_1'] == pytest.approx(expected_feature_1, rel=1e-6)
        assert combined['feature_2'] == pytest.approx(expected_feature_2, rel=1e-6)
    
    def test_calculate_stability_metrics(self):
        """Test stability metrics calculation across segments."""
        segments = [
            Mock(voice_features={'feature_1': 0.8, 'feature_2': 0.6}),
            Mock(voice_features={'feature_1': 0.82, 'feature_2': 0.58}),
            Mock(voice_features={'feature_1': 0.78, 'feature_2': 0.62})
        ]
        
        stability = self.combiner._calculate_stability_metrics(segments)
        
        assert 'overall_stability' in stability
        assert 'feature_consistency' in stability
        assert 'quality_consistency' in stability
        assert 'segment_count' in stability
        assert stability['segment_count'] == 3
        assert 0 <= stability['overall_stability'] <= 1


class TestVoiceModelCache:
    """Test voice model caching and optimization system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = VoiceModelCache(self.temp_dir, max_cache_size=3)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initialization and metadata loading."""
        assert self.cache.cache_dir.exists()
        assert self.cache.max_cache_size == 3
        assert isinstance(self.cache.cache_metadata, dict)
    
    def test_store_and_retrieve_model(self):
        """Test storing and retrieving models from cache."""
        model_id = "test_model_1"
        model_data = {"model_type": "test", "parameters": {"param1": 0.5}}
        
        metadata = VoiceModelMetadata(
            model_id=model_id,
            voice_profile_id="profile_1",
            reference_audio_ids=["audio_1"],
            training_duration=120.0,
            audio_segments=2,
            quality_score=0.85,
            similarity_score=0.9,
            model_size_mb=45.2,
            inference_time_ms=850,
            training_config=TrainingConfiguration(),
            voice_characteristics={},
            optimization_history=[],
            created_at=1234567890.0,
            last_updated=1234567890.0,
            usage_count=0,
            cache_priority=1.0
        )
        
        # Store model
        self.cache.store_model(model_id, model_data, metadata)
        
        # Retrieve model
        retrieved = self.cache.get_model(model_id)
        
        assert retrieved is not None
        assert retrieved["model_type"] == "test"
        assert retrieved["parameters"]["param1"] == 0.5
    
    def test_cache_eviction(self):
        """Test cache eviction when exceeding maximum size."""
        # Store models exceeding cache size
        for i in range(5):  # More than max_cache_size (3)
            model_id = f"test_model_{i}"
            model_data = {"model_type": "test", "id": i}
            
            metadata = VoiceModelMetadata(
                model_id=model_id,
                voice_profile_id=f"profile_{i}",
                reference_audio_ids=[f"audio_{i}"],
                training_duration=120.0,
                audio_segments=2,
                quality_score=0.85,
                similarity_score=0.9,
                model_size_mb=45.2,
                inference_time_ms=850,
                training_config=TrainingConfiguration(),
                voice_characteristics={},
                optimization_history=[],
                created_at=1234567890.0 + i,
                last_updated=1234567890.0 + i,
                usage_count=i,  # Different usage counts for priority
                cache_priority=float(i)
            )
            
            self.cache.store_model(model_id, model_data, metadata)
        
        # Check that cache size is within limit
        assert len(self.cache.cache_metadata) <= self.cache.max_cache_size
    
    def test_usage_stats_update(self):
        """Test usage statistics update for cache prioritization."""
        model_id = "test_model_usage"
        model_data = {"model_type": "test"}
        
        metadata = VoiceModelMetadata(
            model_id=model_id,
            voice_profile_id="profile_1",
            reference_audio_ids=["audio_1"],
            training_duration=120.0,
            audio_segments=2,
            quality_score=0.85,
            similarity_score=0.9,
            model_size_mb=45.2,
            inference_time_ms=850,
            training_config=TrainingConfiguration(),
            voice_characteristics={},
            optimization_history=[],
            created_at=1234567890.0,
            last_updated=1234567890.0,
            usage_count=0,
            cache_priority=1.0
        )
        
        self.cache.store_model(model_id, model_data, metadata)
        initial_usage = self.cache.cache_metadata[model_id].usage_count
        
        # Retrieve model multiple times
        for _ in range(3):
            self.cache.get_model(model_id)
        
        final_usage = self.cache.cache_metadata[model_id].usage_count
        assert final_usage > initial_usage
    
    def test_cache_stats(self):
        """Test cache statistics calculation."""
        # Add some models
        for i in range(2):
            model_id = f"stats_model_{i}"
            model_data = {"model_type": "test", "size": 1000}
            
            metadata = VoiceModelMetadata(
                model_id=model_id,
                voice_profile_id=f"profile_{i}",
                reference_audio_ids=[f"audio_{i}"],
                training_duration=120.0,
                audio_segments=2,
                quality_score=0.85,
                similarity_score=0.9,
                model_size_mb=45.2,
                inference_time_ms=850,
                training_config=TrainingConfiguration(),
                voice_characteristics={},
                optimization_history=[],
                created_at=1234567890.0,
                last_updated=1234567890.0,
                usage_count=5,
                cache_priority=1.0
            )
            
            self.cache.store_model(model_id, model_data, metadata)
        
        stats = self.cache.get_cache_stats()
        
        assert 'total_models' in stats
        assert 'memory_cached_models' in stats
        assert 'total_size_mb' in stats
        assert 'cache_hit_rate' in stats
        assert stats['total_models'] == 2


class TestIntelligentVoiceModelTrainer:
    """Test main intelligent voice model training system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = IntelligentVoiceModelTrainer()
    
    @pytest.mark.asyncio
    async def test_create_dedicated_voice_model_insufficient_duration(self):
        """Test model creation with insufficient audio duration."""
        with patch('app.services.intelligent_voice_model_trainer.librosa.load') as mock_load:
            # Mock short audio (less than 30 seconds)
            mock_load.return_value = (np.random.randn(22050 * 10), 22050)  # 10 seconds
            
            success, model_id, metadata = await self.trainer.create_dedicated_voice_model(
                audio_paths=["short_audio.wav"],
                voice_profile_id="test_profile"
            )
            
            assert not success
            assert model_id is None
            assert "Total audio duration" in metadata["error"]
    
    @pytest.mark.asyncio
    async def test_create_dedicated_voice_model_success(self):
        """Test successful dedicated voice model creation."""
        with patch('app.services.intelligent_voice_model_trainer.librosa.load') as mock_load:
            # Mock sufficient audio duration
            mock_load.return_value = (np.random.randn(22050 * 60), 22050)  # 60 seconds
            
            # Mock multi-segment combiner
            with patch.object(self.trainer.multi_segment_combiner, 'analyze_segments') as mock_analyze:
                mock_segments = [
                    Mock(
                        duration=60.0,
                        quality_metrics={'overall_quality': 0.85},
                        voice_features={'feature_1': 0.8}
                    )
                ]
                mock_analyze.return_value = mock_segments
                
                with patch.object(self.trainer.multi_segment_combiner, 'combine_characteristics') as mock_combine:
                    mock_combine.return_value = {
                        'quality_metrics': {'overall_quality': 0.85},
                        'voice_features': {'feature_1': 0.8},
                        'prosody_features': {},
                        'emotional_features': {},
                        'spectral_features': {},
                        'stability_metrics': {'overall_stability': 0.9}
                    }
                    
                    # Mock model training and validation
                    with patch.object(self.trainer, '_train_dedicated_model') as mock_train:
                        mock_train.return_value = {
                            'model_id': 'test_model',
                            'model_size_mb': 45.2,
                            'inference_time_ms': 850
                        }
                        
                        with patch.object(self.trainer, '_validate_model_quality') as mock_validate:
                            mock_validate.return_value = {
                                'passed': True,
                                'similarity_score': 0.92
                            }
                            
                            success, model_id, metadata = await self.trainer.create_dedicated_voice_model(
                                audio_paths=["long_audio.wav"],
                                voice_profile_id="test_profile"
                            )
                            
                            assert success
                            assert model_id is not None
                            assert metadata['quality_score'] == 0.85
                            assert metadata['similarity_score'] == 0.92
    
    @pytest.mark.asyncio
    async def test_improve_model_incrementally_success(self):
        """Test successful incremental model improvement."""
        # First create a model in cache
        model_id = "existing_model"
        existing_metadata = {
            'model_id': model_id,
            'voice_profile_id': 'test_profile',
            'reference_audio_ids': ['audio_1'],
            'training_duration': 120.0,
            'audio_segments': 1,
            'quality_score': 0.8,
            'similarity_score': 0.85,
            'model_size_mb': 45.2,
            'inference_time_ms': 850,
            'training_config': TrainingConfiguration().__dict__,
            'voice_characteristics': {'quality_metrics': {'overall_quality': 0.8}},
            'optimization_history': [],
            'created_at': 1234567890.0,
            'last_updated': 1234567890.0,
            'usage_count': 0,
            'cache_priority': 1.0
        }
        
        # Mock cache retrieval
        with patch.object(self.trainer.model_cache, 'get_model') as mock_get:
            mock_get.return_value = {'metadata': existing_metadata}
            
            # Mock segment analysis
            with patch.object(self.trainer.multi_segment_combiner, 'analyze_segments') as mock_analyze:
                mock_segments = [
                    Mock(
                        duration=30.0,
                        quality_metrics={'overall_quality': 0.9}
                    )
                ]
                mock_analyze.return_value = mock_segments
                
                # Mock characteristic combination
                with patch.object(self.trainer.multi_segment_combiner, 'combine_characteristics') as mock_combine:
                    mock_combine.return_value = {
                        'quality_metrics': {'overall_quality': 0.88}  # Improved quality
                    }
                    
                    # Mock improvement calculation
                    with patch.object(self.trainer, '_calculate_improvement_metrics') as mock_calc:
                        mock_calc.return_value = {
                            'quality_improvement': 0.08,  # 8% improvement
                            'similarity_improvement': 0.05
                        }
                        
                        # Mock model training and validation
                        with patch.object(self.trainer, '_train_dedicated_model') as mock_train:
                            mock_train.return_value = {'model_id': model_id + '_improved'}
                            
                            with patch.object(self.trainer, '_validate_model_quality') as mock_validate:
                                mock_validate.return_value = {
                                    'passed': True,
                                    'similarity_score': 0.9
                                }
                                
                                success, results = await self.trainer.improve_model_incrementally(
                                    model_id=model_id,
                                    additional_audio_paths=["additional_audio.wav"]
                                )
                                
                                assert success
                                assert results['improvement_metrics']['quality_improvement'] == 0.08
                                assert results['new_similarity_score'] == 0.9
    
    @pytest.mark.asyncio
    async def test_improve_model_insufficient_improvement(self):
        """Test incremental improvement with insufficient improvement."""
        model_id = "existing_model"
        existing_metadata = {
            'model_id': model_id,
            'voice_profile_id': 'test_profile',
            'voice_characteristics': {'quality_metrics': {'overall_quality': 0.8}},
            'training_config': TrainingConfiguration().__dict__
        }
        
        with patch.object(self.trainer.model_cache, 'get_model') as mock_get:
            mock_get.return_value = {'metadata': existing_metadata}
            
            with patch.object(self.trainer.multi_segment_combiner, 'analyze_segments') as mock_analyze:
                mock_analyze.return_value = [Mock()]
                
                with patch.object(self.trainer.multi_segment_combiner, 'combine_characteristics') as mock_combine:
                    mock_combine.return_value = {'quality_metrics': {'overall_quality': 0.8}}
                    
                    with patch.object(self.trainer, '_calculate_improvement_metrics') as mock_calc:
                        mock_calc.return_value = {
                            'quality_improvement': 0.01  # Only 1% improvement (below 2% threshold)
                        }
                        
                        success, results = await self.trainer.improve_model_incrementally(
                            model_id=model_id,
                            additional_audio_paths=["additional_audio.wav"]
                        )
                        
                        assert not success
                        assert "Insufficient improvement" in results["error"]
    
    def test_generate_model_id(self):
        """Test model ID generation."""
        voice_profile_id = "test_profile"
        segment_count = 3
        
        model_id = self.trainer._generate_model_id(voice_profile_id, segment_count)
        
        assert model_id.startswith("voice_model_")
        assert len(model_id) == len("voice_model_") + 12  # 12-character hash
    
    def test_calculate_improvement_metrics(self):
        """Test improvement metrics calculation."""
        existing_characteristics = {
            'quality_metrics': {'overall_quality': 0.8},
            'stability_metrics': {'overall_stability': 0.75},
            'voice_features': {'feature_1': 0.6, 'feature_2': 0.7}
        }
        
        improved_characteristics = {
            'quality_metrics': {'overall_quality': 0.85},
            'stability_metrics': {'overall_stability': 0.8},
            'voice_features': {'feature_1': 0.65, 'feature_2': 0.75}
        }
        
        metrics = self.trainer._calculate_improvement_metrics(
            existing_characteristics, improved_characteristics
        )
        
        assert metrics['quality_improvement'] == 0.05
        assert metrics['stability_improvement'] == 0.05
        assert metrics['similarity_improvement'] > 0
        assert 'total_features_compared' in metrics
    
    def test_get_model_info_not_found(self):
        """Test getting model info for non-existent model."""
        model_info = self.trainer.get_model_info("non_existent_model")
        assert model_info is None
    
    def test_list_cached_models_empty(self):
        """Test listing cached models when cache is empty."""
        models = self.trainer.list_cached_models()
        assert isinstance(models, list)
        assert len(models) == 0
    
    def test_get_cache_statistics(self):
        """Test cache statistics retrieval."""
        stats = self.trainer.get_cache_statistics()
        
        assert 'total_models' in stats
        assert 'memory_cached_models' in stats
        assert 'total_size_mb' in stats
        assert 'cache_hit_rate' in stats
        assert 'average_model_size_mb' in stats
    
    def test_optimize_cache(self):
        """Test cache optimization."""
        optimization_results = self.trainer.optimize_cache()
        
        assert 'initial_models' in optimization_results
        assert 'final_models' in optimization_results
        assert 'models_evicted' in optimization_results
        assert 'cache_stats' in optimization_results


if __name__ == "__main__":
    pytest.main([__file__])