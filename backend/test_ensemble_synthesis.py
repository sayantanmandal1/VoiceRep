#!/usr/bin/env python3
"""
Test script for Ensemble Voice Synthesis Engine.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.ensemble_voice_synthesis_engine import (
    EnsembleVoiceSynthesizer, 
    ModelSelector, 
    QualityAssessment,
    CrossLanguageAdapter,
    TTSModelType
)
from app.schemas.voice import VoiceProfileSchema

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_model_selector():
    """Test the model selection functionality."""
    logger.info("Testing Model Selector...")
    
    selector = ModelSelector()
    
    # Create a test voice profile
    from app.schemas.voice import FrequencyRange
    
    voice_profile = VoiceProfileSchema(
        id="test_profile",
        reference_audio_id="test_audio_ref",
        fundamental_frequency=FrequencyRange(
            min_hz=100, max_hz=200, mean_hz=150, std_hz=20
        ),
        formant_frequencies=[500, 1500, 2500],
        speech_rate=4.5,
        pause_frequency=0.3,
        energy_variance=0.2,
        pitch_variance=0.15,
        quality_score=0.85,
        created_at="2024-01-01T00:00:00Z"
    )
    
    # Test model selection
    selected_models = selector.select_optimal_models(voice_profile, "Hello world", "en")
    
    logger.info(f"Selected models: {[m.value for m in selected_models]}")
    assert len(selected_models) > 0, "At least one model should be selected"
    
    logger.info("✓ Model Selector test passed")


async def test_quality_assessment():
    """Test the quality assessment functionality."""
    logger.info("Testing Quality Assessment...")
    
    assessor = QualityAssessment()
    
    # Create dummy audio data
    import numpy as np
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    
    # Generate synthetic audio signals
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Reference audio (sine wave at 150 Hz)
    reference_audio = np.sin(2 * np.pi * 150 * t) * 0.5
    
    # Synthesized audio (similar but slightly different)
    synthesized_audio = np.sin(2 * np.pi * 155 * t) * 0.45
    
    # Assess quality
    quality_metrics = assessor.assess_synthesis_quality(
        synthesized_audio, reference_audio, sample_rate
    )
    
    logger.info(f"Quality metrics: {quality_metrics}")
    
    # Verify all metrics are present
    expected_metrics = ["similarity", "naturalness", "clarity", "prosody", "overall"]
    for metric in expected_metrics:
        assert metric in quality_metrics, f"Missing quality metric: {metric}"
        assert 0 <= quality_metrics[metric] <= 1, f"Quality metric {metric} out of range: {quality_metrics[metric]}"
    
    logger.info("✓ Quality Assessment test passed")


async def test_cross_language_adapter():
    """Test the cross-language adaptation functionality."""
    logger.info("Testing Cross-Language Adapter...")
    
    adapter = CrossLanguageAdapter()
    
    # Create a test voice profile
    from app.schemas.voice import FrequencyRange
    
    voice_profile = VoiceProfileSchema(
        id="test_profile",
        reference_audio_id="test_audio_ref",
        fundamental_frequency=FrequencyRange(
            min_hz=100, max_hz=200, mean_hz=150, std_hz=20
        ),
        formant_frequencies=[500, 1500, 2500],
        quality_score=0.85,
        created_at="2024-01-01T00:00:00Z"
    )
    
    # Test adaptation to Spanish
    adapted_profile = adapter.adapt_voice_for_language(voice_profile, "es")
    
    logger.info(f"Original profile ID: {voice_profile.id}")
    logger.info(f"Adapted profile ID: {adapted_profile.id}")
    
    # Verify adaptation occurred
    assert adapted_profile.id != voice_profile.id, "Profile ID should be different after adaptation"
    assert "es" in adapted_profile.id, "Adapted profile ID should contain target language"
    
    # Test adaptation to unsupported language (should return original)
    unchanged_profile = adapter.adapt_voice_for_language(voice_profile, "unsupported")
    assert unchanged_profile.id == voice_profile.id, "Profile should be unchanged for unsupported language"
    
    logger.info("✓ Cross-Language Adapter test passed")


async def test_ensemble_synthesizer_initialization():
    """Test the ensemble synthesizer initialization."""
    logger.info("Testing Ensemble Synthesizer Initialization...")
    
    synthesizer = EnsembleVoiceSynthesizer()
    
    # Test configuration
    assert len(synthesizer.model_configs) > 0, "Should have model configurations"
    
    # Verify model configurations
    for model_type, config in synthesizer.model_configs.items():
        assert isinstance(model_type, TTSModelType), f"Invalid model type: {model_type}"
        assert config.quality_score > 0, f"Invalid quality score for {model_type}"
        assert len(config.languages) > 0, f"No languages specified for {model_type}"
    
    # Test ensemble weights
    weights = synthesizer.ensemble_weights
    assert len(weights.model_weights) > 0, "Should have model weights"
    
    total_weight = sum(weights.model_weights.values())
    assert abs(total_weight - 1.0) < 0.1, f"Model weights should sum to ~1.0, got {total_weight}"
    
    logger.info("✓ Ensemble Synthesizer initialization test passed")


async def run_all_tests():
    """Run all tests."""
    logger.info("Starting Ensemble Voice Synthesis Engine tests...")
    
    try:
        await test_model_selector()
        await test_quality_assessment()
        await test_cross_language_adapter()
        await test_ensemble_synthesizer_initialization()
        
        logger.info("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)