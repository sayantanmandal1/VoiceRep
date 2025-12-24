"""
Celery tasks for audio processing operations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from celery import current_task
import ffmpeg
import librosa
import numpy as np
from scipy.io import wavfile

from app.core.celery_app import celery_app
from app.core.config import settings
from app.models.file import ProcessingStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioExtractionService:
    """Service for audio extraction and processing operations."""
    
    # Supported video formats
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # Audio quality presets
    QUALITY_PRESETS = {
        'low': {'sample_rate': 16000, 'bitrate': '64k', 'channels': 1},
        'medium': {'sample_rate': 22050, 'bitrate': '128k', 'channels': 1},
        'high': {'sample_rate': 44100, 'bitrate': '320k', 'channels': 2},
        'ultra': {'sample_rate': 48000, 'bitrate': '320k', 'channels': 2}
    }
    
    def __init__(self):
        """Initialize audio extraction service."""
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for directory in [settings.UPLOAD_DIR, settings.RESULTS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate_video_file(self, video_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate video file for audio extraction.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(video_path):
                return False, "Video file not found"
            
            # Check file extension
            file_ext = Path(video_path).suffix.lower()
            if file_ext not in self.SUPPORTED_VIDEO_FORMATS:
                return False, f"Unsupported video format: {file_ext}"
            
            # Probe video file to check for audio stream
            probe = ffmpeg.probe(video_path)
            audio_streams = [
                stream for stream in probe['streams'] 
                if stream['codec_type'] == 'audio'
            ]
            
            if not audio_streams:
                return False, "No audio stream found in video file"
            
            return True, None
            
        except ffmpeg.Error as e:
            return False, f"Invalid video file: {str(e)}"
        except Exception as e:
            return False, f"Error validating video file: {str(e)}"
    
    def extract_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive audio metadata.
        
        Args:
            file_path: Path to audio/video file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            probe = ffmpeg.probe(file_path)
            
            # Find audio stream
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                return {}
            
            metadata = {
                'duration': float(probe['format'].get('duration', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'bitrate': int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None,
                'format': probe['format'].get('format_name', 'unknown')
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}
    
    def extract_audio(
        self, 
        video_path: str, 
        output_path: str, 
        quality: str = 'high',
        progress_callback: Optional[callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Extract audio from video file with quality settings.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file
            quality: Quality preset (low, medium, high, ultra)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (success, error_message, metadata)
        """
        try:
            # Validate input
            is_valid, error_msg = self.validate_video_file(video_path)
            if not is_valid:
                return False, error_msg, {}
            
            # Get quality settings
            quality_settings = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS['high'])
            
            if progress_callback:
                progress_callback(10, "Analyzing video file")
            
            # Get input metadata
            input_metadata = self.extract_audio_metadata(video_path)
            
            if progress_callback:
                progress_callback(20, "Starting audio extraction")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Build ffmpeg command
            input_stream = ffmpeg.input(video_path)
            
            # Configure audio extraction parameters
            audio_params = {
                'acodec': 'pcm_s16le',  # Use PCM for high quality
                'ar': quality_settings['sample_rate'],
                'ac': quality_settings['channels'],
                'f': 'wav'  # Output as WAV for consistency
            }
            
            if progress_callback:
                progress_callback(30, "Extracting audio")
            
            # Extract audio
            output_stream = ffmpeg.output(input_stream, output_path, **audio_params)
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            
            if progress_callback:
                progress_callback(70, "Validating extracted audio")
            
            # Validate output
            if not os.path.exists(output_path):
                return False, "Audio extraction failed - output file not created", {}
            
            # Get output metadata
            output_metadata = self.extract_audio_metadata(output_path)
            
            if progress_callback:
                progress_callback(90, "Normalizing audio")
            
            # Normalize audio if needed
            normalized_path = self.normalize_audio(output_path, output_path)
            if not normalized_path:
                logger.warning("Audio normalization failed, using original extraction")
            
            if progress_callback:
                progress_callback(100, "Audio extraction complete")
            
            # Combine metadata
            result_metadata = {
                'input': input_metadata,
                'output': output_metadata,
                'quality_preset': quality,
                'extraction_settings': quality_settings
            }
            
            return True, None, result_metadata
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error during audio extraction: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"Unexpected error during audio extraction: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, {}
    
    def normalize_audio(self, input_path: str, output_path: str) -> Optional[str]:
        """
        Normalize audio levels and remove silence.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for normalized output
            
        Returns:
            Output path if successful, None otherwise
        """
        try:
            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            
            # Remove leading/trailing silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Normalize to prevent clipping
            if np.max(np.abs(y_trimmed)) > 0:
                y_normalized = y_trimmed / np.max(np.abs(y_trimmed)) * 0.95
            else:
                y_normalized = y_trimmed
            
            # Convert to 16-bit PCM
            y_int16 = (y_normalized * 32767).astype(np.int16)
            
            # Save normalized audio
            wavfile.write(output_path, sr, y_int16)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {str(e)}")
            return None
    
    def validate_extracted_audio(self, audio_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate extracted audio quality and characteristics.
        
        Args:
            audio_path: Path to extracted audio file
            
        Returns:
            Tuple of (is_valid, error_message, quality_metrics)
        """
        try:
            if not os.path.exists(audio_path):
                return False, "Audio file not found", {}
            
            # Load audio for analysis
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate quality metrics
            duration = librosa.get_duration(y=y, sr=sr)
            rms_energy = np.sqrt(np.mean(y**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            quality_metrics = {
                'duration': duration,
                'sample_rate': sr,
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(zero_crossing_rate),
                'spectral_centroid': float(spectral_centroid),
                'file_size': os.path.getsize(audio_path)
            }
            
            # Validation checks
            if duration < 0.5:
                return False, "Audio too short (minimum 0.5 seconds)", quality_metrics
            
            if rms_energy < 0.001:
                return False, "Audio signal too weak", quality_metrics
            
            if sr < 8000:
                return False, "Sample rate too low (minimum 8kHz)", quality_metrics
            
            return True, None, quality_metrics
            
        except Exception as e:
            return False, f"Error validating audio: {str(e)}", {}


# Global service instance
audio_extraction_service = AudioExtractionService()


@celery_app.task(bind=True)
def extract_audio_from_video(
    self, 
    video_file_path: str, 
    output_path: str, 
    quality: str = 'high',
    file_id: Optional[str] = None
):
    """
    Extract audio from video file using ffmpeg with progress tracking.
    
    Args:
        video_file_path: Path to input video file
        output_path: Path for extracted audio output
        quality: Audio quality preset (low, medium, high, ultra)
        file_id: Optional file ID for database updates
    
    Returns:
        dict: Task result with status, output path, and metadata
    """
    try:
        # Update task state
        def update_progress(progress: int, status: str):
            self.update_state(
                state='PROGRESS', 
                meta={
                    'progress': progress, 
                    'status': status,
                    'file_id': file_id
                }
            )
        
        update_progress(0, 'Starting audio extraction')
        
        # Extract audio using service
        success, error_msg, metadata = audio_extraction_service.extract_audio(
            video_file_path, 
            output_path, 
            quality,
            progress_callback=update_progress
        )
        
        if not success:
            self.update_state(
                state='FAILURE',
                meta={
                    'error': error_msg, 
                    'status': 'Audio extraction failed',
                    'file_id': file_id
                }
            )
            return {
                'status': 'FAILURE',
                'error': error_msg,
                'file_id': file_id
            }
        
        # Validate extracted audio
        update_progress(95, 'Validating extracted audio')
        is_valid, validation_error, quality_metrics = audio_extraction_service.validate_extracted_audio(output_path)
        
        if not is_valid:
            self.update_state(
                state='FAILURE',
                meta={
                    'error': f'Audio validation failed: {validation_error}', 
                    'status': 'Invalid audio output',
                    'file_id': file_id
                }
            )
            return {
                'status': 'FAILURE',
                'error': f'Audio validation failed: {validation_error}',
                'file_id': file_id
            }
        
        update_progress(100, 'Audio extraction complete')
        
        # Combine all metadata
        result_metadata = {**metadata, 'quality_metrics': quality_metrics}
        
        return {
            'status': 'SUCCESS',
            'output_path': output_path,
            'metadata': result_metadata,
            'message': 'Audio extracted and validated successfully',
            'file_id': file_id
        }
        
    except Exception as exc:
        error_msg = f"Unexpected error in audio extraction task: {str(exc)}"
        logger.error(error_msg)
        self.update_state(
            state='FAILURE',
            meta={
                'error': error_msg, 
                'status': 'Audio extraction failed',
                'file_id': file_id
            }
        )
        return {
            'status': 'FAILURE',
            'error': error_msg,
            'file_id': file_id
        }


@celery_app.task(bind=True)
def analyze_voice_characteristics(self, audio_file_path: str):
    """
    Analyze voice characteristics from audio file.
    
    Args:
        audio_file_path: Path to audio file for analysis
    
    Returns:
        dict: Voice analysis results
    """
    try:
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting voice analysis'})
        
        # Placeholder for actual voice analysis logic
        # This will be implemented in later tasks
        
        self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Voice analysis complete'})
        
        return {
            'status': 'SUCCESS',
            'voice_profile': {},
            'message': 'Voice analysis completed successfully'
        }
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'status': 'Voice analysis failed'}
        )
        raise exc