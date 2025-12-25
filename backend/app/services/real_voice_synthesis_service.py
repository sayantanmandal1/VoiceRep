"""
Real Voice Synthesis Service using Coqui TTS for actual voice cloning.
This service provides real voice synthesis capabilities using state-of-the-art models.
"""

import os
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from datetime import datetime

# Import TTS libraries
try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.utils.generic_utils import get_user_data_dir
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("TTS library not available. Install with: pip install TTS")

from app.core.config import settings
from app.utils.progress_tracker import progress_tracker

logger = logging.getLogger(__name__)

class RealVoiceSynthesisService:
    """Real voice synthesis service using Coqui TTS."""
    
    def __init__(self):
        self.tts_model = None
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 22050
        self.models_dir = Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing Real Voice Synthesis Service on device: {self.device}")
        
    async def initialize_model(self, progress_callback: Optional[Callable] = None) -> bool:
        """Initialize the TTS model."""
        try:
            if not TTS_AVAILABLE:
                raise ImportError("TTS library not available")
            
            if progress_callback:
                progress_callback(10, "Checking TTS model availability")
            
            # Initialize TTS model
            logger.info(f"Loading TTS model: {self.model_name}")
            
            if progress_callback:
                progress_callback(30, "Downloading TTS model (this may take a while on first run)")
            
            # Fix PyTorch weights loading issue by allowing unsafe globals
            import torch.serialization
            from TTS.tts.configs.xtts_config import XttsConfig
            torch.serialization.add_safe_globals([XttsConfig])
            
            # Use XTTS v2 for voice cloning
            self.tts_model = TTS(self.model_name).to(self.device)
            
            if progress_callback:
                progress_callback(80, "Model loaded successfully")
            
            logger.info("TTS model loaded successfully")
            
            if progress_callback:
                progress_callback(100, "Voice synthesis service ready")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {str(e)}")
            if progress_callback:
                progress_callback(0, f"Model initialization failed: {str(e)}")
            return False
    
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
                progress_callback(30, "Extracting voice features")
            
            # Extract voice characteristics
            characteristics = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "audio_length": len(audio),
                "rms_energy": float(np.sqrt(np.mean(audio**2))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
            }
            
            if progress_callback:
                progress_callback(60, "Analyzing pitch and formants")
            
            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                characteristics["fundamental_frequency"] = {
                    "min": float(np.min(pitch_values)),
                    "max": float(np.max(pitch_values)),
                    "mean": float(np.mean(pitch_values)),
                    "std": float(np.std(pitch_values))
                }
            
            if progress_callback:
                progress_callback(80, "Computing spectral features")
            
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
        """Synthesize speech using voice cloning."""
        try:
            if not self.tts_model:
                if progress_callback:
                    progress_callback(5, "Initializing TTS model")
                await self.initialize_model(progress_callback)
            
            if progress_callback:
                progress_callback(20, "Preparing reference audio")
            
            # Ensure reference audio is in the correct format
            reference_audio, sr = librosa.load(reference_audio_path, sr=self.sample_rate)
            
            # Save reference audio in correct format for TTS
            temp_ref_path = tempfile.mktemp(suffix=".wav")
            sf.write(temp_ref_path, reference_audio, self.sample_rate)
            
            if progress_callback:
                progress_callback(40, "Processing text input")
            
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            if progress_callback:
                progress_callback(60, "Generating synthetic speech")
            
            logger.info(f"Synthesizing speech: '{cleaned_text[:50]}...' using reference: {reference_audio_path}")
            
            # Perform voice cloning synthesis
            # XTTS v2 supports voice cloning with reference audio
            wav = self.tts_model.tts(
                text=cleaned_text,
                speaker_wav=temp_ref_path,
                language=self._map_language_code(language)
            )
            
            if progress_callback:
                progress_callback(80, "Post-processing audio")
            
            # Convert to numpy array if needed
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            
            # Ensure audio is in the right format
            if wav.ndim > 1:
                wav = wav.squeeze()
            
            # Normalize audio
            wav = wav / np.max(np.abs(wav))
            
            # Save synthesized audio
            sf.write(output_path, wav, self.sample_rate)
            
            if progress_callback:
                progress_callback(95, "Finalizing output")
            
            # Clean up temporary file
            try:
                os.unlink(temp_ref_path)
            except:
                pass
            
            # Calculate metadata
            duration = len(wav) / self.sample_rate
            quality_score = self._calculate_quality_score(wav)
            
            result = {
                "output_path": output_path,
                "duration": duration,
                "sample_rate": self.sample_rate,
                "quality_score": quality_score,
                "language": language,
                "text_length": len(text),
                "audio_length": len(wav)
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
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for synthesis."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Ensure text ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."
        
        return text
    
    def _map_language_code(self, language: str) -> str:
        """Map language codes to TTS model format."""
        language_map = {
            "english": "en",
            "en": "en",
            "spanish": "es",
            "es": "es",
            "french": "fr",
            "fr": "fr",
            "german": "de",
            "de": "de",
            "italian": "it",
            "it": "it",
            "portuguese": "pt",
            "pt": "pt",
            "polish": "pl",
            "pl": "pl",
            "turkish": "tr",
            "tr": "tr",
            "russian": "ru",
            "ru": "ru",
            "dutch": "nl",
            "nl": "nl",
            "czech": "cs",
            "cs": "cs",
            "arabic": "ar",
            "ar": "ar",
            "chinese": "zh-cn",
            "zh": "zh-cn",
            "japanese": "ja",
            "ja": "ja",
            "hungarian": "hu",
            "hu": "hu",
            "korean": "ko",
            "ko": "ko"
        }
        
        return language_map.get(language.lower(), "en")
    
    def _calculate_quality_score(self, audio: np.ndarray) -> float:
        """Calculate a quality score for the synthesized audio."""
        try:
            # Calculate various quality metrics
            
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio ** 2)
            noise_floor = np.percentile(np.abs(audio), 10) ** 2
            snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
            
            # Dynamic range
            dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
            
            # Spectral flatness (measure of how noise-like vs. tone-like)
            fft = np.fft.fft(audio)
            magnitude = np.abs(fft[:len(fft)//2])
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
            
            # Combine metrics into a quality score (0-1)
            snr_score = np.clip(snr / 30.0, 0, 1)  # Normalize SNR
            dr_score = np.clip(dynamic_range / 40.0, 0, 1)  # Normalize dynamic range
            sf_score = 1 - spectral_flatness  # Lower spectral flatness is better for speech
            
            quality_score = (snr_score * 0.4 + dr_score * 0.3 + sf_score * 0.3)
            
            return float(np.clip(quality_score, 0, 1))
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {str(e)}")
            return 0.75  # Default reasonable score
    
    async def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", 
            "cs", "ar", "zh-cn", "ja", "hu", "ko"
        ]
    
    def is_model_ready(self) -> bool:
        """Check if the TTS model is ready."""
        return self.tts_model is not None and TTS_AVAILABLE


# Global service instance
real_voice_synthesis_service = RealVoiceSynthesisService()


async def initialize_voice_synthesis_service():
    """Initialize the voice synthesis service."""
    try:
        logger.info("Initializing Real Voice Synthesis Service...")
        success = await real_voice_synthesis_service.initialize_model()
        if success:
            logger.info("Real Voice Synthesis Service initialized successfully")
        else:
            logger.error("Failed to initialize Real Voice Synthesis Service")
        return success
    except Exception as e:
        logger.error(f"Voice synthesis service initialization error: {str(e)}")
        return False