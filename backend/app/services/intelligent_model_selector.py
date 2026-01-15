"""
Intelligent Model Selector for Perfect Voice Cloning.

This module implements an intelligent model selection system that analyzes
voice characteristics and selects the optimal combination of TTS models
for maximum voice cloning fidelity.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TTSModel(Enum):
    """Available TTS models for voice cloning."""
    XTTS_V2 = "xtts_v2"
    STYLETTS2 = "styletts2"
    OPENVOICE = "openvoice"
    YOUR_TTS = "your_tts"
    BARK = "bark"


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a TTS model."""
    model: TTSModel
    pitch_range: Tuple[float, float]  # Hz
    prosody_preservation: float  # 0-1
    emotion_capture: float  # 0-1
    naturalness: float  # 0-1
    speed: float  # Relative speed (1.0 = baseline)
    languages: List[str]
    best_for: List[str]
    memory_usage: str  # "low", "medium", "high"


@dataclass
class ModelScore:
    """Score for a model based on voice characteristics."""
    model: TTSModel
    overall_score: float
    pitch_compatibility: float
    prosody_match: float
    emotion_match: float
    naturalness_score: float
    reasons: List[str]


@dataclass
class ModelSelection:
    """Result of model selection."""
    primary_model: TTSModel
    secondary_models: List[TTSModel]
    model_weights: Dict[TTSModel, float]
    selection_confidence: float
    selection_reasons: List[str]


class IntelligentModelSelector:
    """
    Intelligent model selector that chooses optimal TTS models
    based on voice characteristics and synthesis requirements.
    """
    
    # Model capability profiles
    MODEL_PROFILES = {
        TTSModel.XTTS_V2: ModelCapabilities(
            model=TTSModel.XTTS_V2,
            pitch_range=(80, 400),
            prosody_preservation=0.90,
            emotion_capture=0.75,
            naturalness=0.92,
            speed=1.0,
            languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
            best_for=["multilingual", "high_quality", "general_purpose"],
            memory_usage="high"
        ),
        TTSModel.STYLETTS2: ModelCapabilities(
            model=TTSModel.STYLETTS2,
            pitch_range=(70, 450),
            prosody_preservation=0.95,
            emotion_capture=0.90,
            naturalness=0.95,
            speed=0.8,
            languages=["en"],
            best_for=["expressive_speech", "prosody_transfer", "human_level"],
            memory_usage="high"
        ),
        TTSModel.OPENVOICE: ModelCapabilities(
            model=TTSModel.OPENVOICE,
            pitch_range=(80, 400),
            prosody_preservation=0.85,
            emotion_capture=0.80,
            naturalness=0.88,
            speed=1.2,
            languages=["en", "es", "fr", "zh", "ja", "ko"],
            best_for=["instant_cloning", "style_control", "cross_lingual"],
            memory_usage="medium"
        ),
        TTSModel.YOUR_TTS: ModelCapabilities(
            model=TTSModel.YOUR_TTS,
            pitch_range=(90, 350),
            prosody_preservation=0.80,
            emotion_capture=0.70,
            naturalness=0.85,
            speed=1.1,
            languages=["en", "fr", "pt", "tr"],
            best_for=["few_shot", "speaker_adaptation"],
            memory_usage="medium"
        ),
        TTSModel.BARK: ModelCapabilities(
            model=TTSModel.BARK,
            pitch_range=(60, 500),
            prosody_preservation=0.88,
            emotion_capture=0.92,
            naturalness=0.90,
            speed=0.5,
            languages=["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"],
            best_for=["emotional_speech", "natural_prosody", "non_speech_sounds"],
            memory_usage="high"
        )
    }
    
    def __init__(self):
        """Initialize intelligent model selector."""
        self.available_models: List[TTSModel] = []
        self._check_available_models()
        
        logger.info(f"Model selector initialized with {len(self.available_models)} models")
    
    def _check_available_models(self):
        """Check which models are available."""
        # XTTS v2 is always available through TTS library
        self.available_models.append(TTSModel.XTTS_V2)
        self.available_models.append(TTSModel.YOUR_TTS)
        
        # Check for StyleTTS2
        try:
            from app.services.styletts2_synthesizer import STYLETTS2_AVAILABLE
            if STYLETTS2_AVAILABLE:
                self.available_models.append(TTSModel.STYLETTS2)
        except:
            pass
        
        # Check for OpenVoice
        try:
            from app.services.openvoice_synthesizer import OPENVOICE_AVAILABLE
            if OPENVOICE_AVAILABLE:
                self.available_models.append(TTSModel.OPENVOICE)
        except:
            pass
        
        # Bark is available through TTS library
        self.available_models.append(TTSModel.BARK)
    
    def select_models(
        self,
        voice_characteristics: Dict[str, Any],
        text: str,
        language: str = "en",
        priority: str = "quality"  # "quality", "speed", "balanced"
    ) -> ModelSelection:
        """
        Select optimal models based on voice characteristics.
        
        Args:
            voice_characteristics: Extracted voice characteristics
            text: Text to synthesize
            language: Target language
            priority: Selection priority
            
        Returns:
            ModelSelection with recommended models
        """
        # Score each available model
        model_scores = []
        
        for model in self.available_models:
            score = self._score_model(
                model, voice_characteristics, text, language, priority
            )
            model_scores.append(score)
        
        # Sort by overall score
        model_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Select primary and secondary models
        primary = model_scores[0].model
        secondary = [s.model for s in model_scores[1:3] if s.overall_score > 0.5]
        
        # Calculate weights
        total_score = sum(s.overall_score for s in model_scores[:3])
        weights = {}
        for score in model_scores[:3]:
            weights[score.model] = score.overall_score / total_score if total_score > 0 else 0.33
        
        # Compile selection reasons
        reasons = model_scores[0].reasons[:3]
        
        return ModelSelection(
            primary_model=primary,
            secondary_models=secondary,
            model_weights=weights,
            selection_confidence=model_scores[0].overall_score,
            selection_reasons=reasons
        )
    
    def _score_model(
        self,
        model: TTSModel,
        voice_characteristics: Dict[str, Any],
        text: str,
        language: str,
        priority: str
    ) -> ModelScore:
        """Score a model based on voice characteristics."""
        profile = self.MODEL_PROFILES[model]
        reasons = []
        
        # Language compatibility
        if language not in profile.languages:
            return ModelScore(
                model=model,
                overall_score=0.0,
                pitch_compatibility=0.0,
                prosody_match=0.0,
                emotion_match=0.0,
                naturalness_score=0.0,
                reasons=[f"Language {language} not supported"]
            )
        
        # Pitch compatibility
        pitch_mean = voice_characteristics.get('pitch_mean', 150)
        pitch_min, pitch_max = profile.pitch_range
        
        if pitch_min <= pitch_mean <= pitch_max:
            pitch_score = 1.0
            reasons.append(f"Pitch {pitch_mean:.0f}Hz within optimal range")
        else:
            # Calculate penalty for out-of-range pitch
            if pitch_mean < pitch_min:
                pitch_score = max(0.3, 1.0 - (pitch_min - pitch_mean) / 100)
            else:
                pitch_score = max(0.3, 1.0 - (pitch_mean - pitch_max) / 100)
            reasons.append(f"Pitch {pitch_mean:.0f}Hz outside optimal range")
        
        # Prosody match
        prosody_complexity = voice_characteristics.get('prosody_complexity', 0.5)
        prosody_score = 1.0 - abs(profile.prosody_preservation - prosody_complexity)
        
        if prosody_score > 0.8:
            reasons.append("Excellent prosody preservation match")
        
        # Emotion match
        emotional_intensity = voice_characteristics.get('emotional_intensity', 0.5)
        emotion_score = 1.0 - abs(profile.emotion_capture - emotional_intensity)
        
        if emotion_score > 0.8:
            reasons.append("Strong emotion capture capability")
        
        # Naturalness
        naturalness_score = profile.naturalness
        
        # Priority adjustments
        if priority == "quality":
            weight_pitch = 0.25
            weight_prosody = 0.25
            weight_emotion = 0.20
            weight_natural = 0.30
        elif priority == "speed":
            weight_pitch = 0.20
            weight_prosody = 0.20
            weight_emotion = 0.15
            weight_natural = 0.20
            # Add speed bonus
            naturalness_score *= profile.speed
        else:  # balanced
            weight_pitch = 0.25
            weight_prosody = 0.25
            weight_emotion = 0.20
            weight_natural = 0.30
        
        # Calculate overall score
        overall = (
            pitch_score * weight_pitch +
            prosody_score * weight_prosody +
            emotion_score * weight_emotion +
            naturalness_score * weight_natural
        )
        
        return ModelScore(
            model=model,
            overall_score=overall,
            pitch_compatibility=pitch_score,
            prosody_match=prosody_score,
            emotion_match=emotion_score,
            naturalness_score=naturalness_score,
            reasons=reasons
        )
    
    def get_model_info(self, model: TTSModel) -> ModelCapabilities:
        """Get capabilities info for a model."""
        return self.MODEL_PROFILES[model]
    
    def get_available_models(self) -> List[TTSModel]:
        """Get list of available models."""
        return self.available_models.copy()


# Global instance
_model_selector: Optional[IntelligentModelSelector] = None


def get_model_selector() -> IntelligentModelSelector:
    """Get or create global model selector."""
    global _model_selector
    if _model_selector is None:
        _model_selector = IntelligentModelSelector()
    return _model_selector
