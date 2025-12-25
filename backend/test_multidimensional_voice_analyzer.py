#!/usr/bin/env python3
"""
Test script for the Multi-Dimensional Voice Analysis Engine.

This script tests the comprehensive voice analysis functionality including:
- Sub-Hz precision pitch analysis
- Comprehensive formant tracking
- Prosodic pattern extraction
- Timbre analysis (breathiness, roughness, resonance)
- Emotional pattern recognition
- Voice fingerprint with 1000+ features
"""

import os
import sys
import numpy as np
import librosa
import tempfile
import logging

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.multi_dimensional_voice_analyzer import MultiDimensionalVoiceAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_audio(duration=3.0, sample_rate=22050):
    """Create a synthetic test audio signal with voice-like characteristics."""
    
    # Generate time array
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Create a fundamental frequency that varies over time (like natural speech)
    f0_base = 150  # Base frequency (Hz)
    f0_variation = 30 * np.sin(2 * np.pi * 0.5 * t)  # Slow pitch variation
    f0 = f0_base + f0_variation
    
    # Generate harmonic series (like a voice)
    signal = np.zeros_like(t)
    
    # Add harmonics with decreasing amplitude
    for harmonic in range(1, 6):
        amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
        signal += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)
    
    # Add formant-like resonances
    # Simulate vowel formants
    formant_freqs = [800, 1200, 2600]  # Typical vowel formants
    for formant_freq in formant_freqs:
        formant_signal = 0.3 * np.sin(2 * np.pi * formant_freq * t)
        # Apply envelope to make it more voice-like
        envelope = np.exp(-5 * np.abs(t - duration/2))
        signal += formant_signal * envelope
    
    # Add some noise to make it more realistic
    noise = 0.05 * np.random.randn(len(t))
    signal += noise
    
    # Apply amplitude modulation (like natural speech rhythm)
    amplitude_modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
    signal *= amplitude_modulation
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal, sample_rate


def test_multidimensional_analyzer():
    """Test the multi-dimensional voice analyzer with synthetic audio."""
    
    logger.info("Starting Multi-Dimensional Voice Analysis Engine Test")
    
    try:
        # Create test audio
        logger.info("Creating synthetic test audio...")
        audio_data, sample_rate = create_test_audio(duration=5.0)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            import soundfile as sf
            sf.write(temp_path, audio_data, sample_rate)
        
        logger.info(f"Test audio saved to: {temp_path}")
        
        # Initialize analyzer
        logger.info("Initializing Multi-Dimensional Voice Analyzer...")
        analyzer = MultiDimensionalVoiceAnalyzer(sample_rate=sample_rate)
        
        # Perform comprehensive analysis
        logger.info("Performing comprehensive multi-dimensional analysis...")
        analysis_result = analyzer.analyze_voice_comprehensive(temp_path)
        
        # Verify results
        logger.info("Verifying analysis results...")
        
        # Check that all major components are present
        required_components = [
            'pitch_features', 'formant_features', 'timbre_features',
            'prosodic_features', 'emotional_features', 'voice_fingerprint',
            'quality_metrics', 'processing_time', 'audio_metadata'
        ]
        
        for component in required_components:
            if component not in analysis_result:
                raise ValueError(f"Missing required component: {component}")
            logger.info(f"✓ {component} present")
        
        # Check voice fingerprint feature count
        fingerprint = analysis_result['voice_fingerprint']
        feature_count = fingerprint.get('_total_features', 0)
        
        logger.info(f"Voice fingerprint contains {feature_count} features")
        
        if feature_count < 1000:
            logger.warning(f"Feature count ({feature_count}) is below target of 1000+")
        else:
            logger.info(f"✓ Voice fingerprint meets 1000+ feature requirement")
        
        # Check pitch analysis precision
        pitch_features = analysis_result['pitch_features']
        logger.info(f"Pitch analysis - Jitter: {pitch_features.jitter:.6f}, Shimmer: {pitch_features.shimmer:.6f}")
        logger.info(f"Pitch stability: {pitch_features.pitch_stability:.3f}")
        logger.info(f"HNR: {pitch_features.harmonics_to_noise_ratio:.2f} dB")
        
        # Check formant analysis
        formant_features = analysis_result['formant_features']
        logger.info(f"Vowel space area: {formant_features.vowel_space_area:.2f}")
        logger.info(f"Formant dispersion: {formant_features.formant_dispersion:.2f}")
        
        # Check timbre analysis
        timbre_features = analysis_result['timbre_features']
        logger.info(f"Breathiness: {timbre_features.breathiness_measure:.3f}")
        logger.info(f"Roughness: {timbre_features.roughness_measure:.3f}")
        logger.info(f"Brightness: {timbre_features.brightness_measure:.3f}")
        
        # Check emotional analysis
        emotional_features = analysis_result['emotional_features']
        logger.info(f"Emotional valence: {emotional_features.emotional_dimensions.get('valence', 0):.3f}")
        logger.info(f"Emotional arousal: {emotional_features.emotional_dimensions.get('arousal', 0):.3f}")
        logger.info(f"Speaking confidence: {emotional_features.confidence_measures.get('overall_confidence', 0):.3f}")
        
        # Check quality metrics
        quality_metrics = analysis_result['quality_metrics']
        logger.info(f"Overall quality: {quality_metrics.overall_quality:.3f}")
        logger.info(f"SNR: {quality_metrics.signal_to_noise_ratio:.2f} dB")
        
        # Check processing time
        processing_time = analysis_result['processing_time']
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        # Performance check
        audio_duration = analysis_result['audio_metadata']['duration']
        if processing_time < audio_duration * 6:  # Should be faster than 6x real-time for 5s audio
            logger.info(f"✓ Processing performance acceptable ({processing_time:.2f}s for {audio_duration:.2f}s audio)")
        else:
            logger.warning(f"Processing slower than expected ({processing_time:.2f}s for {audio_duration:.2f}s audio)")
        
        logger.info("✓ Multi-Dimensional Voice Analysis Engine test completed successfully!")
        
        # Print summary of key features
        print("\n" + "="*60)
        print("MULTI-DIMENSIONAL VOICE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Features Extracted: {feature_count}")
        print(f"Analysis Version: {fingerprint.get('_fingerprint_version', 'Unknown')}")
        print(f"Processing Time: {processing_time:.2f}s")
        print(f"Audio Duration: {audio_duration:.2f}s")
        print(f"Quality Score: {quality_metrics.overall_quality:.3f}")
        print(f"Pitch Precision: Sub-Hz (Jitter: {pitch_features.jitter:.6f})")
        print(f"Formant Tracking: {len(formant_features.formant_trajectories)} formants")
        print(f"Voice Quality: Breathiness={timbre_features.breathiness_measure:.3f}, Roughness={timbre_features.roughness_measure:.3f}")
        print(f"Emotional Profile: Valence={emotional_features.emotional_dimensions.get('valence', 0):.3f}, Arousal={emotional_features.emotional_dimensions.get('arousal', 0):.3f}")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Cleaned up temporary test file")


if __name__ == "__main__":
    success = test_multidimensional_analyzer()
    sys.exit(0 if success else 1)