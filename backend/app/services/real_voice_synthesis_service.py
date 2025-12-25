"""
Real Voice Synthesis Service - Production-ready TTS implementation
"""

import os
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np
from datetime import datetime

# Required audio processing libraries
import librosa
import soundfile as sf

# Required TTS libraries
import torch

# Handle PyTorch compatibility for TTS
try:
    # Try to import weight_norm from the new location first
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    try:
        # Fallback to old location
        from torch.nn.utils import weight_norm
        # Monkey patch it to the new location for TTS compatibility
        import torch.nn.utils.parametrizations
        torch.nn.utils.parametrizations.weight_norm = weight_norm
    except ImportError:
        pass

from TTS.api import TTS

from app.core.config import settings

logger = logging.getLogger(__name__)

class RealVoiceSynthesisService:
    """Production-ready voice synthesis service using Coqui TTS."""
    
    def __init__(self):
        self.sample_rate = 22050
        self.models_dir = Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # TTS model configuration
        self.tts_model = None
        self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"  # Fast, good quality model
        self.vocoder_name = "vocoder_models/en/ljspeech/hifigan_v2"
        
        # Voice cloning model (if available)
        self.voice_clone_model = None
        self.clone_model_name = "tts_models/multilingual/multi-dataset/your_tts"
        
        logger.info(f"Initializing Production Voice Synthesis Service")
        
    async def initialize_model(self, progress_callback: Optional[Callable] = None) -> bool:
        """Initialize the TTS models."""
        try:
            if progress_callback:
                progress_callback(10, "Initializing TTS models")
            
            logger.info("Loading TTS model...")
            
            if progress_callback:
                progress_callback(30, "Loading primary TTS model")
            
            # Initialize primary TTS model
            self.tts_model = TTS(model_name=self.model_name, progress_bar=False)
            logger.info(f"Successfully loaded TTS model: {self.model_name}")
            
            if progress_callback:
                progress_callback(70, "Loading voice cloning model")
            
            # Initialize voice cloning model
            try:
                self.voice_clone_model = TTS(model_name=self.clone_model_name, progress_bar=False)
                logger.info(f"Successfully loaded voice cloning model: {self.clone_model_name}")
            except Exception as e:
                logger.warning(f"Voice cloning model failed to load: {e}")
                # Try alternative cloning models
                alternative_models = [
                    "tts_models/multilingual/multi-dataset/xtts_v2",
                    "tts_models/en/vctk/vits"
                ]
                for model in alternative_models:
                    try:
                        self.voice_clone_model = TTS(model_name=model, progress_bar=False)
                        logger.info(f"Successfully loaded alternative cloning model: {model}")
                        break
                    except Exception as alt_e:
                        logger.warning(f"Alternative model {model} failed: {alt_e}")
                        continue
                
                if not self.voice_clone_model:
                    logger.error("No voice cloning model could be loaded")
            
            if progress_callback:
                progress_callback(100, "TTS service ready")
            
            logger.info("TTS service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {str(e)}")
            if progress_callback:
                progress_callback(0, f"TTS initialization failed: {str(e)}")
            raise RuntimeError(f"TTS initialization failed: {str(e)}")
    
    async def analyze_voice_characteristics(
        self, 
        audio_path: str, 
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Analyze voice characteristics from reference audio."""
        try:
            if progress_callback:
                progress_callback(10, "Loading reference audio")
            
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if progress_callback:
                progress_callback(50, "Analyzing voice features")
            
            # Extract comprehensive voice characteristics
            characteristics = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "audio_length": len(audio),
                "rms_energy": float(np.sqrt(np.mean(audio**2))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
            }
            
            # Extract pitch information
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            pitch_values = pitch_values[pitch_values > 0]
            
            if len(pitch_values) > 0:
                characteristics["fundamental_frequency"] = {
                    "min": float(np.min(pitch_values)),
                    "max": float(np.max(pitch_values)),
                    "mean": float(np.mean(pitch_values)),
                    "std": float(np.std(pitch_values))
                }
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            characteristics["spectral_features"] = {
                "centroid_mean": float(np.mean(spectral_centroids)),
                "rolloff_mean": float(np.mean(spectral_rolloff)),
                "mfcc_means": [float(np.mean(mfcc)) for mfcc in mfccs]
            }
            
            if progress_callback:
                progress_callback(100, "Voice analysis complete")
            
            logger.info(f"Voice analysis completed for {audio_path}")
            return characteristics
            
        except Exception as e:
            logger.error(f"Voice analysis failed: {str(e)}")
            if progress_callback:
                progress_callback(0, f"Voice analysis failed: {str(e)}")
            raise
    
    async def synthesize_speech(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        language: str = "en",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Synthesize speech using TTS with voice cloning."""
        try:
            if progress_callback:
                progress_callback(20, "Preparing synthesis")
            
            logger.info(f"Synthesizing speech: '{text[:50]}...' using reference: {reference_audio_path}")
            
            # Ensure models are initialized
            if not self.tts_model and not self.voice_clone_model:
                raise RuntimeError("No TTS models are initialized")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if progress_callback:
                progress_callback(40, "Generating synthetic speech")
            
            # Try voice cloning first if reference audio is provided and model is available
            if self.voice_clone_model and os.path.exists(reference_audio_path):
                if progress_callback:
                    progress_callback(60, "Cloning voice characteristics")
                
                # Use voice cloning
                self.voice_clone_model.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio_path,
                    language=language,
                    file_path=output_path
                )
                logger.info("Voice cloning synthesis completed")
                synthesis_method = "voice_cloning"
                quality_score = 0.9
                
            elif self.tts_model:
                if progress_callback:
                    progress_callback(60, "Generating speech with TTS")
                
                # Use regular TTS
                self.tts_model.tts_to_file(
                    text=text,
                    file_path=output_path
                )
                logger.info("Regular TTS synthesis completed")
                synthesis_method = "tts"
                quality_score = 0.75
                
            else:
                raise RuntimeError("No TTS model available for synthesis")
            
            # Verify the output file was created and has content
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("TTS failed to generate audio file")
            
            if progress_callback:
                progress_callback(80, "Processing audio output")
            
            # Load and analyze the generated audio
            audio, sr = librosa.load(output_path, sr=self.sample_rate)
            duration = len(audio) / sr
            
            # Ensure audio is not silent
            if np.max(np.abs(audio)) < 0.001:
                raise RuntimeError("Generated audio is silent")
            
            # Resave with consistent format
            sf.write(output_path, audio, sr, format='WAV', subtype='PCM_16')
            
            result = {
                "output_path": output_path,
                "duration": duration,
                "sample_rate": self.sample_rate,
                "quality_score": quality_score,
                "language": language,
                "text_length": len(text),
                "synthesis_method": synthesis_method
            }
            
            if progress_callback:
                progress_callback(100, "Speech synthesis complete")
            
            logger.info(f"Speech synthesis completed: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            if progress_callback:
                progress_callback(0, f"Synthesis failed: {str(e)}")
            raise
    
    async def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        if self.tts_model:
            try:
                # Try to get languages from the model
                if hasattr(self.tts_model, 'languages'):
                    return list(self.tts_model.languages)
                elif hasattr(self.tts_model, 'config') and hasattr(self.tts_model.config, 'languages'):
                    return list(self.tts_model.config.languages)
            except Exception as e:
                logger.warning(f"Could not get languages from model: {e}")
        
        # Default supported languages for most TTS models
        return ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    
    def is_model_ready(self) -> bool:
        """Check if the synthesis service is ready."""
        return self.tts_model is not None or self.voice_clone_model is not None


# Global service instance
real_voice_synthesis_service = RealVoiceSynthesisService()


async def initialize_voice_synthesis_service():
    """Initialize the voice synthesis service."""
    try:
        logger.info("Initializing Production Voice Synthesis Service...")
        success = await real_voice_synthesis_service.initialize_model()
        logger.info("Production Voice Synthesis Service initialized successfully")
        return success
    except Exception as e:
        logger.error(f"Voice synthesis service initialization error: {str(e)}")
        raise RuntimeError(f"Failed to initialize voice synthesis service: {str(e)}")