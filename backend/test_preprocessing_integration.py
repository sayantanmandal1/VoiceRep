#!/usr/bin/env python3
"""
Integration test for Advanced Audio Preprocessing Engine with Celery tasks.

This script tests the integration between the preprocessing engine and the task system.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
import logging

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.tasks.audio_processing import audio_extraction_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_audio(duration=3.0, sample_rate=22050):
    """Create synthetic test audio."""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Create a voice-like signal
    f0 = 150  # Fundamental frequency
    signal = np.sin(2 * np.pi * f0 * t)
    
    # Add harmonics
    for harmonic in range(2, 5):
        signal += 0.5 / harmonic * np.sin(2 * np.pi * f0 * harmonic * t)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(signal))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    return signal, sample_rate


def test_preprocessing_service():
    """Test the preprocessing service integration."""
    logger.info("Testing Advanced Preprocessing Service Integration")
    
    # Create test audio
    test_audio, sample_rate = create_test_audio()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
        input_path = temp_input.name
        sf.write(input_path, test_audio, sample_rate)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
        output_path = temp_output.name
    
    try:
        # Test advanced preprocessing
        logger.info("Testing advanced preprocessing method")
        success, error_msg, metadata = audio_extraction_service.preprocess_audio_advanced(
            input_path, 
            output_path
        )
        
        assert success, f"Preprocessing should succeed: {error_msg}"
        assert os.path.exists(output_path), "Output file should be created"
        assert metadata is not None, "Metadata should be returned"
        
        logger.info("✓ Advanced preprocessing service working correctly")
        
        # Test quality assessment
        logger.info("Testing quality assessment method")
        assessment = audio_extraction_service.assess_audio_quality_detailed(input_path)
        
        assert 'overall_score' in assessment, "Assessment should include overall score"
        assert 'voice_suitability_score' in assessment, "Assessment should include voice suitability"
        assert 'technical_metrics' in assessment, "Assessment should include technical metrics"
        
        logger.info("✓ Quality assessment service working correctly")
        
        # Log results
        logger.info(f"Quality Score: {assessment['overall_score']:.3f}")
        logger.info(f"Voice Suitability: {assessment['voice_suitability_score']:.3f}")
        logger.info(f"Issues Detected: {len(assessment.get('issues_detected', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {str(e)}")
        return False
        
    finally:
        # Clean up
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.unlink(path)


def main():
    """Run integration tests."""
    logger.info("Starting Advanced Audio Preprocessing Integration Tests")
    
    try:
        if test_preprocessing_service():
            logger.info("🎉 All integration tests passed!")
            return True
        else:
            logger.error("❌ Integration tests failed!")
            return False
    except Exception as e:
        logger.error(f"❌ Integration tests failed with exception: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)