"""
Integration test for Intelligent Voice Model Training System using real audio.
"""

import asyncio
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from app.services.intelligent_voice_model_trainer import (
    IntelligentVoiceModelTrainer,
    MultiSegmentCombiner,
    VoiceModelCache,
    TrainingConfiguration
)


async def test_multi_segment_combiner():
    """Test multi-segment characteristic combination with real audio."""
    print("Testing MultiSegmentCombiner with real audio...")
    
    combiner = MultiSegmentCombiner()
    
    # Use real audio file from downloads folder
    downloads_dir = Path("../downloads")
    audio_file = downloads_dir / "Taylor Swift - The Fate of Ophelia (Official Music Video).mp3"
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return False
    
    # Use the real audio file
    audio_paths = [str(audio_file)]
    
    try:
        # Analyze segments
        print(f"Analyzing {len(audio_paths)} audio segments...")
        segments = combiner.analyze_segments(audio_paths)
        
        print(f"Successfully analyzed {len(segments)} segments")
        for i, segment in enumerate(segments):
            print(f"  Segment {i}: duration={segment.duration:.1f}s, "
                  f"quality={segment.quality_metrics.get('overall_quality', 0):.2f}")
        
        if segments:
            # Combine characteristics
            print("Combining segment characteristics...")
            combined = combiner.combine_characteristics(segments)
            
            print(f"Combined characteristics:")
            print(f"  Total segments: {combined['total_segments']}")
            print(f"  Total duration: {combined['total_duration']:.1f}s")
            print(f"  Overall quality: {combined['quality_metrics'].get('overall_quality', 0):.2f}")
            print(f"  Stability: {combined['stability_metrics'].get('overall_stability', 0):.2f}")
            print(f"  Voice features: {len(combined['voice_features'])} features")
            
            return True
        else:
            print("No segments were successfully analyzed")
            return False
            
    except Exception as e:
        print(f"Error in multi-segment combination: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_voice_model_cache():
    """Test voice model caching system."""
    print("\nTesting VoiceModelCache...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = VoiceModelCache(temp_dir, max_cache_size=3)
        
        # Test storing and retrieving models
        test_models = []
        for i in range(5):  # More than cache size to test eviction
            model_id = f"test_model_{i}"
            model_data = {
                "model_type": "test",
                "parameters": {"param1": i * 0.1, "param2": i * 0.2},
                "quality_score": 0.8 + i * 0.02
            }
            
            from app.services.intelligent_voice_model_trainer import VoiceModelMetadata
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
        
        print(f"Stored {len(test_models)} models in cache")
        
        # Test retrieval
        retrieved_count = 0
        for model_id, original_data in test_models:
            retrieved = cache.get_model(model_id)
            if retrieved:
                retrieved_count += 1
                print(f"  Retrieved {model_id}: quality={retrieved.get('quality_score', 0):.2f}")
        
        print(f"Successfully retrieved {retrieved_count} models")
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        print(f"Cache statistics:")
        print(f"  Total models: {stats['total_models']}")
        print(f"  Memory cached: {stats['memory_cached_models']}")
        print(f"  Total size: {stats['total_size_mb']:.1f} MB")
        print(f"  Hit rate: {stats['cache_hit_rate']:.2f}")
        
        return retrieved_count > 0


async def test_intelligent_voice_model_trainer():
    """Test main intelligent voice model trainer with real audio."""
    print("\nTesting IntelligentVoiceModelTrainer with real audio...")
    
    trainer = IntelligentVoiceModelTrainer()
    
    # Use real audio file from downloads folder
    downloads_dir = Path("../downloads")
    audio_file = downloads_dir / "Taylor Swift - The Fate of Ophelia (Official Music Video).mp3"
    
    if not audio_file.exists():
        print(f"Audio file not found: {audio_file}")
        return False
    
    audio_paths = [str(audio_file)]
    print(f"Using real audio file: {audio_file.name}")
    
    try:
        # Test dedicated model creation
        print("Creating dedicated voice model...")
        
        def progress_callback(progress, message):
            print(f"  Progress: {progress}% - {message}")
        
        success, model_id, metadata = await trainer.create_dedicated_voice_model(
            audio_paths=audio_paths,
            voice_profile_id="test_profile_real_audio",
            progress_callback=progress_callback
        )
        
        if success:
            print(f"Successfully created model: {model_id}")
            print(f"  Quality score: {metadata.get('quality_score', 0):.2f}")
            print(f"  Similarity score: {metadata.get('similarity_score', 0):.2f}")
            print(f"  Training duration: {metadata.get('training_duration', 0):.1f}s")
            print(f"  Audio segments: {metadata.get('audio_segments', 0)}")
            
            # Test model info retrieval
            model_info = trainer.get_model_info(model_id)
            if model_info:
                print(f"Model info retrieved successfully")
            
            # Test cache statistics
            cache_stats = trainer.get_cache_statistics()
            print(f"Cache contains {cache_stats['total_models']} models")
            
            return True
        else:
            print(f"Model creation failed: {metadata}")
            return False
            
    except Exception as e:
        print(f"Error in voice model training: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests with real audio."""
    print("Starting Intelligent Voice Model Training System Integration Tests (Real Audio)")
    print("=" * 80)
    
    results = []
    
    # Test 1: Multi-segment combiner
    try:
        result1 = await test_multi_segment_combiner()
        results.append(("MultiSegmentCombiner", result1))
    except Exception as e:
        print(f"MultiSegmentCombiner test failed: {e}")
        results.append(("MultiSegmentCombiner", False))
    
    # Test 2: Voice model cache
    try:
        result2 = await test_voice_model_cache()
        results.append(("VoiceModelCache", result2))
    except Exception as e:
        print(f"VoiceModelCache test failed: {e}")
        results.append(("VoiceModelCache", False))
    
    # Test 3: Intelligent voice model trainer
    try:
        result3 = await test_intelligent_voice_model_trainer()
        results.append(("IntelligentVoiceModelTrainer", result3))
    except Exception as e:
        print(f"IntelligentVoiceModelTrainer test failed: {e}")
        results.append(("IntelligentVoiceModelTrainer", False))
    
    # Print results summary
    print("\n" + "=" * 80)
    print("Integration Test Results:")
    print("-" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<35} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All integration tests passed!")
        return True
    else:
        print("❌ Some integration tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)