#!/usr/bin/env python3
"""
Test script for Advanced Audio Preprocessing Engine.

This script tests the core functionality of the advanced preprocessing system
to ensure it meets the requirements for high-fidelity voice cloning.
"""

import os
import sys
import numpy as np
import librosa
from pathlib import Path
import tempfile
import logging

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.advanced_audio_preprocessing import AdvancedAudioPreprocessor
from app.services.audio_quality_assessment import audio_quality_assessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_audio(duration=5.0, sample_rate=22050, add_noise=True, add_distortion=False):
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
    
    # Add noise if requested
    if add_noise:
        noise_level = 0.1
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise
    
    # Add distortion if requested
    if add_distortion:
        # Simple clipping distortion
        signal = np.clip(signal, -0.8, 0.8)
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    return signal, sample_rate


def test_preprocessing_requirements():
    """Test that preprocessing meets all requirements."""
    logger.info("Testing Advanced Audio Preprocessing Engine Requirements")
    
    # Create test audio
    test_audio, sample_rate = create_test_audio(duration=3.0, add_noise=True)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
        import soundfile as sf
        sf.write(temp_path, test_audio, sample_rate)
    
    try:
        # Initialize preprocessor
        preprocessor = AdvancedAudioPreprocessor(target_sample_rate=22050)
        
        # Test Requirement 1.1: Normalize audio levels without altering voice characteristics
        logger.info("Testing Requirement 1.1: Audio level normalization")
        processed = preprocessor.preprocess_audio(temp_path, preserve_characteristics=True)
        
        assert processed.sample_rate >= 22050, "Sample rate should be at least 22kHz (Req 1.5)"
        assert len(processed.enhancement_applied) > 0, "Enhancements should be applied"
        assert "level_normalization" in processed.enhancement_applied, "Level normalization should be applied"
        
        logger.info(f"✓ Level normalization applied. Quality score: {processed.quality_score:.3f}")
        
        # Test Requirement 1.2: Noise reduction with voice preservation
        logger.info("Testing Requirement 1.2: Noise reduction with voice preservation")
        assert "advanced_noise_reduction" in processed.enhancement_applied, "Noise reduction should be applied"
        assert processed.noise_level >= 0, "Noise level should be estimated"
        
        logger.info(f"✓ Noise reduction applied. Estimated noise level: {processed.noise_level:.3f}")
        
        # Test Requirement 1.3: Quality enhancement for degraded audio
        logger.info("Testing Requirement 1.3: Quality enhancement")
        assert "quality_enhancement" in processed.enhancement_applied, "Quality enhancement should be applied"
        
        logger.info("✓ Quality enhancement applied")
        
        # Test Requirement 1.4: Compression restoration
        logger.info("Testing Requirement 1.4: Compression restoration")
        assert "compression_restoration" in processed.enhancement_applied, "Compression restoration should be applied"
        
        logger.info("✓ Compression restoration applied")
        
        # Test Requirement 1.5: Sample rate maintenance (minimum 22kHz)
        logger.info("Testing Requirement 1.5: Sample rate maintenance")
        assert processed.sample_rate >= 22050, f"Sample rate {processed.sample_rate} should be at least 22050 Hz"
        
        logger.info(f"✓ Sample rate maintained at {processed.sample_rate} Hz")
        
        # Test quality assessment
        logger.info("Testing quality assessment system")
        initial_quality = audio_quality_assessor.assess_audio_quality(test_audio, sample_rate)
        final_quality = audio_quality_assessor.assess_audio_quality(processed.audio_data, processed.sample_rate)
        
        logger.info(f"Quality improvement: {initial_quality.overall_score:.3f} -> {final_quality.overall_score:.3f}")
        
        # Test spectral analysis
        logger.info("Testing spectral analysis")
        assert 'spectral_centroid' in processed.spectral_analysis, "Spectral centroid should be calculated"
        assert 'harmonic_ratio' in processed.spectral_analysis, "Harmonic ratio should be calculated"
        
        logger.info(f"✓ Spectral analysis complete. Harmonic ratio: {processed.spectral_analysis['harmonic_ratio']:.3f}")
        
        # Test frequency response analysis
        logger.info("Testing frequency response analysis")
        assert len(processed.frequency_response) > 0, "Frequency response should be analyzed"
        
        logger.info("✓ Frequency response analysis complete")
        
        logger.info("🎉 All preprocessing requirements tests passed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_quality_assessment():
    """Test the quality assessment system."""
    logger.info("Testing Audio Quality Assessment System")
    
    # Test with different quality levels
    test_cases = [
        ("High Quality", create_test_audio(duration=5.0, add_noise=False, add_distortion=False)),
        ("Noisy Audio", create_test_audio(duration=5.0, add_noise=True, add_distortion=False)),
        ("Distorted Audio", create_test_audio(duration=5.0, add_noise=False, add_distortion=True)),
        ("Poor Quality", create_test_audio(duration=2.0, add_noise=True, add_distortion=True))
    ]
    
    for test_name, (test_audio, sample_rate) in test_cases:
        logger.info(f"Testing {test_name}")
        
        assessment = audio_quality_assessor.assess_audio_quality(test_audio, sample_rate)
        
        logger.info(f"  Overall Score: {assessment.overall_score:.3f}")
        logger.info(f"  Voice Suitability: {assessment.voice_suitability_score:.3f}")
        logger.info(f"  Issues Detected: {len(assessment.issues_detected)}")
        logger.info(f"  Recommendations: {len(assessment.enhancement_recommendations)}")
        
        # Verify assessment structure
        assert 0 <= assessment.overall_score <= 1, "Overall score should be between 0 and 1"
        assert 0 <= assessment.voice_suitability_score <= 1, "Voice suitability should be between 0 and 1"
        assert isinstance(assessment.technical_metrics, dict), "Technical metrics should be a dictionary"
        
        logger.info(f"✓ {test_name} assessment complete")
    
    logger.info("🎉 Quality assessment tests passed!")
    return True


def test_enhancement_recommendations():
    """Test the enhancement recommendation system."""
    logger.info("Testing Enhancement Recommendation System")
    
    # Create poor quality audio
    poor_audio, sample_rate = create_test_audio(duration=1.5, add_noise=True, add_distortion=True)
    
    assessment = audio_quality_assessor.assess_audio_quality(poor_audio, sample_rate)
    
    logger.info(f"Issues detected: {len(assessment.issues_detected)}")
    for issue in assessment.issues_detected:
        logger.info(f"  - {issue.issue_type.value}: {issue.description} (severity: {issue.severity:.2f})")
    
    logger.info(f"Recommendations: {len(assessment.enhancement_recommendations)}")
    for rec in assessment.enhancement_recommendations:
        logger.info(f"  - Priority {rec.priority}: {rec.enhancement_type} - {rec.description}")
    
    # Verify recommendations are provided for poor quality audio
    assert len(assessment.issues_detected) > 0, "Issues should be detected in poor quality audio"
    assert len(assessment.enhancement_recommendations) > 0, "Recommendations should be provided"
    
    logger.info("✓ Enhancement recommendations working correctly")
    return True


def main():
    """Run all tests."""
    logger.info("Starting Advanced Audio Preprocessing Engine Tests")
    
    tests = [
        test_preprocessing_requirements,
        test_quality_assessment,
        test_enhancement_recommendations
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_func.__name__} PASSED")
            else:
                logger.error(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_func.__name__} FAILED with exception: {str(e)}")
    
    logger.info(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Advanced Audio Preprocessing Engine is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)