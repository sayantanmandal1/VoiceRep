#!/usr/bin/env python3
"""
Test script for Advanced Audio Post-Processing Engine.

This script tests the core functionality of the advanced post-processing system
to ensure it meets the requirements for high-fidelity voice cloning.
"""

import os
import sys
import numpy as np
import librosa
from pathlib import Path
import tempfile
import logging
import time

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.advanced_audio_post_processor import AdvancedAudioPostProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_audio(duration=5.0, sample_rate=22050, audio_type="clean"):
    """Create synthetic test audio with voice-like characteristics."""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Create a voice-like signal with fundamental frequency and harmonics
    f0 = 150  # Fundamental frequency (Hz)
    
    # Generate harmonics
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):  # First 5 harmonics
        amplitude = 1.0 / harmonic  # Decreasing amplitude
        signal += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
    
    # Add formant-like resonances
    formants = [500, 1500, 2500]  # Typical formant frequencies
    for formant in formants:
        signal += 0.3 * np.sin(2 * np.pi * formant * t) * np.exp(-t * 0.5)
    
    # Add prosodic variation (pitch modulation)
    pitch_modulation = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Slow pitch variation
    signal = signal * pitch_modulation
    
    # Apply different characteristics based on audio type
    if audio_type == "synthesized":
        # Add synthesis artifacts
        # Spectral discontinuities
        for i in range(0, len(signal), len(signal)//10):
            if i + 100 < len(signal):
                signal[i:i+100] *= 0.5  # Create discontinuity
        
        # Add some aliasing-like artifacts
        signal += 0.05 * np.sin(2 * np.pi * (sample_rate * 0.45) * t)
        
    elif audio_type == "noisy":
        # Add noise
        noise_level = 0.15
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise
        
    elif audio_type == "compressed":
        # Simulate compression artifacts
        signal = np.tanh(signal * 2) * 0.7  # Soft compression
        
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    return signal, sample_rate


def create_reference_audio(duration=5.0, sample_rate=22050):
    """Create high-quality reference audio."""
    return create_test_audio(duration, sample_rate, "clean")


def test_spectral_matching():
    """Test spectral matching and frequency alignment (Requirement 6.1)."""
    logger.info("Testing Requirement 6.1: Spectral matching and frequency alignment")
    
    # Create test audio
    synthesized_audio, sr = create_test_audio(duration=3.0, audio_type="synthesized")
    reference_audio, _ = create_reference_audio(duration=3.0, sample_rate=sr)
    
    # Initialize post-processor
    post_processor = AdvancedAudioPostProcessor(sample_rate=sr)
    
    # Test spectral matching
    start_time = time.time()
    spectral_result = post_processor.apply_spectral_matching(synthesized_audio, reference_audio)
    processing_time = time.time() - start_time
    
    # Verify results
    assert spectral_result.matched_audio is not None, "Matched audio should be generated"
    assert len(spectral_result.matched_audio) > 0, "Matched audio should not be empty"
    assert 0 <= spectral_result.frequency_alignment_score <= 1, "Alignment score should be between 0 and 1"
    assert spectral_result.spectral_distance >= 0, "Spectral distance should be non-negative"
    assert len(spectral_result.enhancement_applied) > 0, "Enhancements should be applied"
    assert spectral_result.processing_time > 0, "Processing time should be recorded"
    
    logger.info(f"✓ Spectral matching completed in {processing_time:.3f}s")
    logger.info(f"  Frequency alignment score: {spectral_result.frequency_alignment_score:.3f}")
    logger.info(f"  Spectral distance: {spectral_result.spectral_distance:.3f}")
    logger.info(f"  Enhancements applied: {spectral_result.enhancement_applied}")
    
    return True


def test_artifact_removal():
    """Test artifact removal and audio smoothing (Requirement 6.2)."""
    logger.info("Testing Requirement 6.2: Artifact removal and audio smoothing")
    
    # Create audio with artifacts
    synthesized_audio, sr = create_test_audio(duration=3.0, audio_type="synthesized")
    
    # Add some specific artifacts
    # Add clipping
    synthesized_audio[1000:1100] = 0.99
    synthesized_audio[2000:2010] = -0.99
    
    # Add noise bursts
    synthesized_audio[3000:3050] += np.random.normal(0, 0.5, 50)
    
    # Initialize post-processor
    post_processor = AdvancedAudioPostProcessor(sample_rate=sr)
    
    # Test artifact removal
    start_time = time.time()
    artifact_result = post_processor.remove_synthesis_artifacts(synthesized_audio)
    processing_time = time.time() - start_time
    
    # Verify results
    assert artifact_result.cleaned_audio is not None, "Cleaned audio should be generated"
    assert len(artifact_result.cleaned_audio) > 0, "Cleaned audio should not be empty"
    assert isinstance(artifact_result.artifacts_detected, list), "Artifacts detected should be a list"
    assert isinstance(artifact_result.artifacts_removed, list), "Artifacts removed should be a list"
    assert artifact_result.processing_time > 0, "Processing time should be recorded"
    
    # Check that some artifacts were detected
    assert len(artifact_result.artifacts_detected) > 0, "Some artifacts should be detected"
    
    logger.info(f"✓ Artifact removal completed in {processing_time:.3f}s")
    logger.info(f"  Artifacts detected: {artifact_result.artifacts_detected}")
    logger.info(f"  Artifacts removed: {artifact_result.artifacts_removed}")
    logger.info(f"  Quality improvement: {artifact_result.quality_improvement:.3f}")
    
    return True


def test_voice_characteristic_preservation():
    """Test voice characteristic preservation during enhancement (Requirement 6.3)."""
    logger.info("Testing Requirement 6.3: Voice characteristic preservation during enhancement")
    
    # Create test audio
    enhanced_audio, sr = create_test_audio(duration=3.0, audio_type="clean")
    original_synthesized, _ = create_test_audio(duration=3.0, audio_type="synthesized")
    reference_audio, _ = create_reference_audio(duration=3.0, sample_rate=sr)
    
    # Initialize post-processor
    post_processor = AdvancedAudioPostProcessor(sample_rate=sr)
    
    # Test voice characteristic preservation
    start_time = time.time()
    preservation_result = post_processor.preserve_voice_characteristics(
        enhanced_audio, original_synthesized, reference_audio
    )
    processing_time = time.time() - start_time
    
    # Verify results
    assert 'enhanced_audio' in preservation_result, "Enhanced audio should be returned"
    assert 'metrics' in preservation_result, "Preservation metrics should be returned"
    
    enhanced_audio_result = preservation_result['enhanced_audio']
    metrics = preservation_result['metrics']
    
    assert enhanced_audio_result is not None, "Enhanced audio should not be None"
    assert len(enhanced_audio_result) > 0, "Enhanced audio should not be empty"
    
    # Check preservation metrics
    assert 0 <= metrics.fundamental_frequency_preserved <= 1, "F0 preservation should be between 0 and 1"
    assert 0 <= metrics.formant_preservation_score <= 1, "Formant preservation should be between 0 and 1"
    assert 0 <= metrics.prosody_preservation_score <= 1, "Prosody preservation should be between 0 and 1"
    assert 0 <= metrics.timbre_preservation_score <= 1, "Timbre preservation should be between 0 and 1"
    assert 0 <= metrics.overall_preservation_score <= 1, "Overall preservation should be between 0 and 1"
    
    logger.info(f"✓ Voice characteristic preservation completed in {processing_time:.3f}s")
    logger.info(f"  F0 preservation: {metrics.fundamental_frequency_preserved:.3f}")
    logger.info(f"  Formant preservation: {metrics.formant_preservation_score:.3f}")
    logger.info(f"  Prosody preservation: {metrics.prosody_preservation_score:.3f}")
    logger.info(f"  Timbre preservation: {metrics.timbre_preservation_score:.3f}")
    logger.info(f"  Overall preservation: {metrics.overall_preservation_score:.3f}")
    
    return True


def test_dynamic_range_matching():
    """Test dynamic range compression matching (Requirement 6.4)."""
    logger.info("Testing Requirement 6.4: Dynamic range compression matching")
    
    # Create test audio with different dynamic ranges
    audio, sr = create_test_audio(duration=3.0, audio_type="clean")
    reference_audio, _ = create_test_audio(duration=3.0, audio_type="compressed")
    
    # Initialize post-processor
    post_processor = AdvancedAudioPostProcessor(sample_rate=sr)
    
    # Test dynamic range matching
    start_time = time.time()
    compression_result = post_processor.match_dynamic_range_compression(audio, reference_audio)
    processing_time = time.time() - start_time
    
    # Verify results
    assert compression_result.compressed_audio is not None, "Compressed audio should be generated"
    assert len(compression_result.compressed_audio) > 0, "Compressed audio should not be empty"
    assert compression_result.compression_ratio > 0, "Compression ratio should be positive"
    assert compression_result.dynamic_range_before >= 0, "Dynamic range before should be non-negative"
    assert compression_result.dynamic_range_after >= 0, "Dynamic range after should be non-negative"
    assert 0 <= compression_result.reference_match_score <= 1, "Reference match score should be between 0 and 1"
    
    logger.info(f"✓ Dynamic range matching completed in {processing_time:.3f}s")
    logger.info(f"  Compression ratio: {compression_result.compression_ratio:.3f}")
    logger.info(f"  Dynamic range before: {compression_result.dynamic_range_before:.1f} dB")
    logger.info(f"  Dynamic range after: {compression_result.dynamic_range_after:.1f} dB")
    logger.info(f"  Reference match score: {compression_result.reference_match_score:.3f}")
    
    return True


def test_consistency_maintenance():
    """Test consistency maintenance for volume and quality (Requirement 6.5)."""
    logger.info("Testing Requirement 6.5: Consistency maintenance for volume and quality")
    
    # Create test audio with inconsistencies
    audio, sr = create_test_audio(duration=3.0, audio_type="clean")
    reference_audio, _ = create_reference_audio(duration=3.0, sample_rate=sr)
    
    # Add volume inconsistencies
    audio[0:len(audio)//3] *= 0.5  # Quiet section
    audio[2*len(audio)//3:] *= 1.5  # Loud section
    
    # Initialize post-processor
    post_processor = AdvancedAudioPostProcessor(sample_rate=sr)
    
    # Test consistency maintenance
    start_time = time.time()
    consistency_result = post_processor.maintain_consistency(audio, reference_audio)
    processing_time = time.time() - start_time
    
    # Verify results
    assert 'consistent_audio' in consistency_result, "Consistent audio should be returned"
    assert 'metrics' in consistency_result, "Consistency metrics should be returned"
    
    consistent_audio = consistency_result['consistent_audio']
    metrics = consistency_result['metrics']
    
    assert consistent_audio is not None, "Consistent audio should not be None"
    assert len(consistent_audio) > 0, "Consistent audio should not be empty"
    
    # Check consistency metrics
    assert 0 <= metrics.volume_consistency <= 1, "Volume consistency should be between 0 and 1"
    assert 0 <= metrics.quality_consistency <= 1, "Quality consistency should be between 0 and 1"
    assert 0 <= metrics.spectral_consistency <= 1, "Spectral consistency should be between 0 and 1"
    assert 0 <= metrics.temporal_consistency <= 1, "Temporal consistency should be between 0 and 1"
    assert 0 <= metrics.overall_consistency <= 1, "Overall consistency should be between 0 and 1"
    
    logger.info(f"✓ Consistency maintenance completed in {processing_time:.3f}s")
    logger.info(f"  Volume consistency: {metrics.volume_consistency:.3f}")
    logger.info(f"  Quality consistency: {metrics.quality_consistency:.3f}")
    logger.info(f"  Spectral consistency: {metrics.spectral_consistency:.3f}")
    logger.info(f"  Temporal consistency: {metrics.temporal_consistency:.3f}")
    logger.info(f"  Overall consistency: {metrics.overall_consistency:.3f}")
    
    return True


def test_similarity_calculation():
    """Test similarity score calculation."""
    logger.info("Testing similarity score calculation")
    
    # Create test audio
    synthesized_audio, sr = create_test_audio(duration=3.0, audio_type="synthesized")
    reference_audio, _ = create_reference_audio(duration=3.0, sample_rate=sr)
    
    # Initialize post-processor
    post_processor = AdvancedAudioPostProcessor(sample_rate=sr)
    
    # Test similarity calculation
    start_time = time.time()
    similarity_score = post_processor.calculate_similarity_score(synthesized_audio, reference_audio)
    processing_time = time.time() - start_time
    
    # Verify results
    assert 0 <= similarity_score <= 1, "Similarity score should be between 0 and 1"
    
    logger.info(f"✓ Similarity calculation completed in {processing_time:.3f}s")
    logger.info(f"  Similarity score: {similarity_score:.3f}")
    
    return True


def test_complete_post_processing_pipeline():
    """Test the complete post-processing pipeline."""
    logger.info("Testing complete post-processing pipeline")
    
    # Create test audio
    synthesized_audio, sr = create_test_audio(duration=3.0, audio_type="synthesized")
    reference_audio, _ = create_reference_audio(duration=3.0, sample_rate=sr)
    
    # Initialize post-processor
    post_processor = AdvancedAudioPostProcessor(sample_rate=sr)
    
    # Test complete pipeline
    start_time = time.time()
    enhanced_audio, processing_metrics = post_processor.enhance_synthesis_quality(
        synthesized_audio, reference_audio, sr, preserve_characteristics=True
    )
    processing_time = time.time() - start_time
    
    # Verify results
    assert enhanced_audio is not None, "Enhanced audio should be generated"
    assert len(enhanced_audio) > 0, "Enhanced audio should not be empty"
    assert isinstance(processing_metrics, dict), "Processing metrics should be a dictionary"
    
    # Check required metrics
    required_metrics = [
        'spectral_matching', 'artifact_removal', 'characteristic_preservation',
        'dynamic_range_matching', 'consistency_maintenance', 'final_similarity_score'
    ]
    
    for metric in required_metrics:
        assert metric in processing_metrics, f"Metric '{metric}' should be present"
    
    # Check similarity score target (>95%)
    similarity_score = processing_metrics['final_similarity_score']
    assert 0 <= similarity_score <= 1, "Final similarity score should be between 0 and 1"
    
    logger.info(f"✓ Complete post-processing pipeline completed in {processing_time:.3f}s")
    logger.info(f"  Final similarity score: {similarity_score:.3f}")
    logger.info(f"  Target achieved (>0.95): {'✓' if similarity_score > 0.95 else '✗'}")
    
    # Log detailed metrics
    for metric_name, metric_data in processing_metrics.items():
        if isinstance(metric_data, dict):
            logger.info(f"  {metric_name}:")
            for key, value in metric_data.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {key}: {value:.3f}")
                else:
                    logger.info(f"    {key}: {value}")
        elif isinstance(metric_data, (int, float)):
            logger.info(f"  {metric_name}: {metric_data:.3f}")
        else:
            logger.info(f"  {metric_name}: {metric_data}")
    
    return True


def run_all_tests():
    """Run all post-processing tests."""
    logger.info("Starting Advanced Audio Post-Processing Engine Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Spectral Matching", test_spectral_matching),
        ("Artifact Removal", test_artifact_removal),
        ("Voice Characteristic Preservation", test_voice_characteristic_preservation),
        ("Dynamic Range Matching", test_dynamic_range_matching),
        ("Consistency Maintenance", test_consistency_maintenance),
        ("Similarity Calculation", test_similarity_calculation),
        ("Complete Pipeline", test_complete_post_processing_pipeline),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- {test_name} ---")
            result = test_func()
            if result:
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Advanced Audio Post-Processing Engine is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)