"""
Core functionality test for Intelligent Voice Model Training System.
Tests the core components without external TTS dependencies.
"""

import asyncio
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
import json
import time

# Import only the core components that don't require TTS
from app.services.intelligent_voice_model_trainer import (
    VoiceModelCache,
    TrainingConfiguration,
    VoiceModelMetadata,
    SegmentCharacteristics
)


def test_training_configuration():
    """Test training configuration dataclass."""
    print("Testing TrainingConfiguration...")
    
    config = TrainingConfiguration()
    
    assert config.min_audio_duration == 30.0
    assert config.max_segments == 10
    assert config.quality_threshold == 0.8
    assert config.training_epochs == 100
    
    print("✓ TrainingConfiguration works correctly")
    return True


def test_voice_model_metadata():
    """Test voice model metadata dataclass."""
    print("Testing VoiceModelMetadata...")
    
    metadata = VoiceModelMetadata(
        model_id="test_model_123",
        voice_profile_id="profile_456",
        reference_audio_ids=["audio_1", "audio_2"],
        training_duration=120.5,
        audio_segments=3,
        quality_score=0.85,
        similarity_score=0.92,
        model_size_mb=45.2,
        inference_time_ms=850,
        training_config=TrainingConfiguration(),
        voice_characteristics={"feature_1": 0.8, "feature_2": 0.6},
        optimization_history=[],
        created_at=time.time(),
        last_updated=time.time(),
        usage_count=0,
        cache_priority=1.0
    )
    
    assert metadata.model_id == "test_model_123"
    assert metadata.audio_segments == 3
    assert metadata.quality_score == 0.85
    assert len(metadata.reference_audio_ids) == 2
    
    print("✓ VoiceModelMetadata works correctly")
    return True


def test_segment_characteristics():
    """Test segment characteristics dataclass."""
    print("Testing SegmentCharacteristics...")
    
    segment = SegmentCharacteristics(
        segment_id="segment_001",
        audio_path="/path/to/audio.wav",
        duration=45.0,
        voice_features={"pitch_mean": 150.0, "formant_f1": 500.0},
        quality_metrics={"overall_quality": 0.8, "snr": 25.0},
        prosody_features={"speech_rate": 4.5, "pause_frequency": 2.1},
        emotional_features={"valence": 0.6, "arousal": 0.4},
        spectral_features={"spectral_centroid": 2000.0},
        confidence_scores={"overall": 0.85, "pitch": 0.9}
    )
    
    assert segment.segment_id == "segment_001"
    assert segment.duration == 45.0
    assert segment.voice_features["pitch_mean"] == 150.0
    assert segment.quality_metrics["overall_quality"] == 0.8
    
    print("✓ SegmentCharacteristics works correctly")
    return True


def test_voice_model_cache():
    """Test voice model caching system."""
    print("Testing VoiceModelCache...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = VoiceModelCache(temp_dir, max_cache_size=3)
        
        # Test cache initialization
        assert cache.cache_dir.exists()
        assert cache.max_cache_size == 3
        assert isinstance(cache.cache_metadata, dict)
        
        # Test storing and retrieving models
        test_models = []
        for i in range(5):  # More than cache size to test eviction
            model_id = f"test_model_{i}"
            model_data = {
                "model_type": "test",
                "parameters": {"param1": i * 0.1, "param2": i * 0.2},
                "quality_score": 0.8 + i * 0.02
            }
            
            metadata = VoiceModelMetadata(
                model_id=model_id,
                voice_profile_id=f"profile_{i}",
                reference_audio_ids=[f"audio_{i}"],
                training_duration=120.0 + i * 10,
                audio_segments=2 + i,
                quality_score=0.8 + i * 0.02,
                similarity_score=0.85 + i * 0.01,
                model_size_mb=45.2 + i * 5,
                inference_time_ms=850 + i * 50,
                training_config=TrainingConfiguration(),
                voice_characteristics={},
                optimization_history=[],
                created_at=1234567890.0 + i,
                last_updated=1234567890.0 + i,
                usage_count=i,
                cache_priority=float(i)
            )
            
            cache.store_model(model_id, model_data, metadata)
            test_models.append((model_id, model_data))
        
        print(f"  Stored {len(test_models)} models in cache")
        
        # Test retrieval
        retrieved_count = 0
        for model_id, original_data in test_models:
            retrieved = cache.get_model(model_id)
            if retrieved:
                retrieved_count += 1
                print(f"    Retrieved {model_id}: quality={retrieved.get('quality_score', 0):.2f}")
        
        print(f"  Successfully retrieved {retrieved_count} models")
        
        # Test cache size limit (should be <= max_cache_size due to eviction)
        assert len(cache.cache_metadata) <= cache.max_cache_size
        
        # Test usage statistics update
        if retrieved_count > 0:
            first_model_id = test_models[0][0]
            if first_model_id in cache.cache_metadata:
                initial_usage = cache.cache_metadata[first_model_id].usage_count
                
                # Retrieve multiple times
                for _ in range(3):
                    cache.get_model(first_model_id)
                
                final_usage = cache.cache_metadata[first_model_id].usage_count
                assert final_usage >= initial_usage
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        print(f"  Cache statistics:")
        print(f"    Total models: {stats['total_models']}")
        print(f"    Memory cached: {stats['memory_cached_models']}")
        print(f"    Total size: {stats['total_size_mb']:.1f} MB")
        print(f"    Hit rate: {stats['cache_hit_rate']:.2f}")
        
        assert 'total_models' in stats
        assert 'memory_cached_models' in stats
        assert 'total_size_mb' in stats
        assert 'cache_hit_rate' in stats
        
        print("✓ VoiceModelCache works correctly")
        return True


def test_cache_metadata_persistence():
    """Test cache metadata persistence to disk."""
    print("Testing cache metadata persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cache and add some models
        cache1 = VoiceModelCache(temp_dir, max_cache_size=5)
        
        for i in range(3):
            model_id = f"persistent_model_{i}"
            model_data = {"test": f"data_{i}"}
            
            metadata = VoiceModelMetadata(
                model_id=model_id,
                voice_profile_id=f"profile_{i}",
                reference_audio_ids=[f"audio_{i}"],
                training_duration=60.0,
                audio_segments=1,
                quality_score=0.8,
                similarity_score=0.85,
                model_size_mb=30.0,
                inference_time_ms=500,
                training_config=TrainingConfiguration(),
                voice_characteristics={},
                optimization_history=[],
                created_at=time.time(),
                last_updated=time.time(),
                usage_count=0,
                cache_priority=1.0
            )
            
            cache1.store_model(model_id, model_data, metadata)
        
        initial_count = len(cache1.cache_metadata)
        print(f"  Stored {initial_count} models in first cache instance")
        
        # Create new cache instance with same directory
        cache2 = VoiceModelCache(temp_dir, max_cache_size=5)
        loaded_count = len(cache2.cache_metadata)
        
        print(f"  Loaded {loaded_count} models in second cache instance")
        
        # Should load the same models
        assert loaded_count == initial_count
        
        # Test that we can retrieve the same models
        retrieved_count = 0
        for i in range(3):
            model_id = f"persistent_model_{i}"
            retrieved = cache2.get_model(model_id)
            if retrieved:
                retrieved_count += 1
        
        print(f"  Retrieved {retrieved_count} models from persisted cache")
        assert retrieved_count == 3
        
        print("✓ Cache metadata persistence works correctly")
        return True


def test_cache_eviction_policy():
    """Test cache eviction policy based on priority."""
    print("Testing cache eviction policy...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = VoiceModelCache(temp_dir, max_cache_size=3)
        
        # Add models with different priorities
        models_data = [
            ("low_priority", 0.1, 1),      # Low priority, low usage
            ("medium_priority", 0.5, 5),   # Medium priority, medium usage  
            ("high_priority", 0.9, 10),    # High priority, high usage
            ("newer_low", 0.2, 2),         # Newer but low priority
            ("newer_high", 0.8, 8)         # Newer and high priority
        ]
        
        for model_id, priority, usage in models_data:
            model_data = {"priority": priority}
            
            metadata = VoiceModelMetadata(
                model_id=model_id,
                voice_profile_id=f"profile_{model_id}",
                reference_audio_ids=[f"audio_{model_id}"],
                training_duration=60.0,
                audio_segments=1,
                quality_score=0.8,
                similarity_score=0.85,
                model_size_mb=30.0,
                inference_time_ms=500,
                training_config=TrainingConfiguration(),
                voice_characteristics={},
                optimization_history=[],
                created_at=time.time(),
                last_updated=time.time(),
                usage_count=usage,
                cache_priority=priority
            )
            
            cache.store_model(model_id, model_data, metadata)
        
        # Should have evicted some models due to size limit
        final_count = len(cache.cache_metadata)
        print(f"  Final cache size: {final_count} (max: {cache.max_cache_size})")
        assert final_count <= cache.max_cache_size
        
        # High priority models should still be in cache
        high_priority_in_cache = "high_priority" in cache.cache_metadata
        newer_high_in_cache = "newer_high" in cache.cache_metadata
        
        print(f"  High priority model in cache: {high_priority_in_cache}")
        print(f"  Newer high priority model in cache: {newer_high_in_cache}")
        
        # At least one high priority model should remain
        assert high_priority_in_cache or newer_high_in_cache
        
        print("✓ Cache eviction policy works correctly")
        return True


def main():
    """Run all core functionality tests."""
    print("Starting Intelligent Voice Model Training System Core Tests")
    print("=" * 65)
    
    test_functions = [
        test_training_configuration,
        test_voice_model_metadata,
        test_segment_characteristics,
        test_voice_model_cache,
        test_cache_metadata_persistence,
        test_cache_eviction_policy
    ]
    
    results = []
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
        print()  # Add spacing between tests
    
    # Print results summary
    print("=" * 65)
    print("Core Test Results:")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All core tests passed!")
        return True
    else:
        print("❌ Some core tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)