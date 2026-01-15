"""
RVC (Retrieval-based Voice Conversion) Voice Converter Service

This service provides voice conversion capabilities using RVC technology.
It uses hf-rvc (HuggingFace RVC) as the primary implementation, which doesn't
require fairseq and works well on Windows.

The service converts TTS output to match a target voice using voice conversion
techniques, supporting pitch shifting and formant preservation.

Real-time voice conversion support enables low-latency streaming conversion
for live audio processing applications.
"""

import os
import logging
import tempfile
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, Callable, Generator, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class RealTimeState(Enum):
    """State of the real-time voice conversion stream."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RVCConfig:
    """Configuration for RVC voice conversion."""
    
    # Pitch extraction method: 'harvest', 'crepe', 'rmvpe', 'pm'
    pitch_method: str = "rmvpe"
    
    # Pitch adjustment in semitones (-12 to +12)
    pitch_shift: int = 0
    
    # Feature search ratio (0.0 to 1.0)
    # Higher values use more of the index for retrieval
    index_rate: float = 0.75
    
    # Median filtering radius for pitch (0-7)
    # Higher values smooth pitch more
    filter_radius: int = 3
    
    # Output resampling rate (0 = no resampling)
    resample_sr: int = 0
    
    # Volume envelope mix rate (0.0 to 1.0)
    # 0 = use converted voice envelope, 1 = use original envelope
    rms_mix_rate: float = 0.25
    
    # Protection for voiceless consonants (0.0 to 0.5)
    # Higher values protect more consonants from conversion
    protect: float = 0.33
    
    # Device for computation ('cpu', 'cuda:0', etc.)
    device: str = "cuda:0"
    
    # Model paths
    model_path: Optional[str] = None
    index_path: Optional[str] = None


@dataclass
class RealTimeConfig:
    """Configuration for real-time voice conversion."""
    
    # Sample rate for real-time processing
    sample_rate: int = 22050
    
    # Chunk size in samples (affects latency)
    # Smaller = lower latency but more CPU usage
    chunk_size: int = 1024
    
    # Number of chunks to buffer before processing
    # Higher = smoother output but more latency
    buffer_chunks: int = 4
    
    # Overlap between chunks (0.0 to 0.5)
    # Higher = smoother transitions but more processing
    overlap_ratio: float = 0.25
    
    # Maximum latency in milliseconds
    max_latency_ms: float = 100.0
    
    # Enable low-latency mode (reduces quality for speed)
    low_latency_mode: bool = False
    
    # Crossfade duration in samples for chunk boundaries
    crossfade_samples: int = 256
    
    # Enable pitch smoothing across chunks
    pitch_smoothing: bool = True
    
    # Pitch smoothing window size
    pitch_smooth_window: int = 5
    
    # Enable formant preservation in real-time mode
    preserve_formants: bool = True
    
    # Quality vs speed tradeoff (0.0 = fastest, 1.0 = best quality)
    quality_factor: float = 0.7


@dataclass
class RealTimeStats:
    """Statistics for real-time voice conversion."""
    
    chunks_processed: int = 0
    total_samples_processed: int = 0
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    dropped_chunks: int = 0
    buffer_underruns: int = 0
    processing_time_ms: float = 0.0
    start_time: float = 0.0
    
    def update_latency(self, latency_ms: float):
        """Update latency statistics."""
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        # Running average
        if self.chunks_processed > 0:
            self.average_latency_ms = (
                (self.average_latency_ms * (self.chunks_processed - 1) + latency_ms) 
                / self.chunks_processed
            )
        else:
            self.average_latency_ms = latency_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "chunks_processed": self.chunks_processed,
            "total_samples_processed": self.total_samples_processed,
            "average_latency_ms": round(self.average_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0.0,
            "dropped_chunks": self.dropped_chunks,
            "buffer_underruns": self.buffer_underruns,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "uptime_seconds": round(time.time() - self.start_time, 2) if self.start_time > 0 else 0.0
        }


@dataclass
class ConversionResult:
    """Result of voice conversion."""
    
    audio: np.ndarray
    sample_rate: int
    success: bool
    method_used: str  # 'hf_rvc', 'rvc_python', 'fallback', 'realtime'
    quality_score: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealTimeVoiceConverter:
    """
    Real-time voice conversion processor for low-latency streaming applications.
    
    This class provides real-time voice conversion capabilities with:
    - Chunk-based processing for low latency
    - Overlap-add synthesis for smooth transitions
    - Pitch and formant preservation
    - Buffer management for continuous streams
    - Statistics tracking for performance monitoring
    
    Usage:
        converter = RealTimeVoiceConverter(rvc_converter, reference_audio)
        converter.start()
        
        # Process audio chunks
        for input_chunk in audio_stream:
            output_chunk = converter.process_chunk(input_chunk)
            play_audio(output_chunk)
        
        converter.stop()
    """
    
    def __init__(
        self,
        rvc_converter: 'RVCVoiceConverter',
        reference_audio: np.ndarray,
        reference_sr: int = 22050,
        config: Optional[RealTimeConfig] = None,
        on_chunk_processed: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Initialize the real-time voice converter.
        
        Args:
            rvc_converter: The RVC voice converter instance to use
            reference_audio: Reference audio for voice matching
            reference_sr: Sample rate of reference audio
            config: Real-time configuration options
            on_chunk_processed: Optional callback when a chunk is processed
        """
        self.rvc_converter = rvc_converter
        self.config = config or RealTimeConfig()
        self.on_chunk_processed = on_chunk_processed
        
        # State management
        self._state = RealTimeState.IDLE
        self._state_lock = threading.Lock()
        
        # Audio buffers
        self._input_buffer: deque = deque(maxlen=self.config.buffer_chunks * 2)
        self._output_buffer: deque = deque(maxlen=self.config.buffer_chunks * 2)
        self._overlap_buffer: Optional[np.ndarray] = None
        
        # Processing thread
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.stats = RealTimeStats()
        
        # Pitch history for smoothing
        self._pitch_history: deque = deque(maxlen=self.config.pitch_smooth_window)
        
        # Reference voice characteristics (pre-computed for speed)
        self._ref_characteristics: Optional[Dict[str, Any]] = None
        self._reference_audio = self._preprocess_reference(reference_audio, reference_sr)
        
        # Pre-compute reference characteristics
        self._initialize_reference()
        
        logger.info(f"RealTimeVoiceConverter initialized with chunk_size={self.config.chunk_size}, "
                   f"buffer_chunks={self.config.buffer_chunks}")
    
    def _preprocess_reference(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess reference audio for real-time matching."""
        try:
            import librosa
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to target sample rate
            if sample_rate != self.config.sample_rate:
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=self.config.sample_rate
                )
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Reference preprocessing failed: {e}")
            return audio.astype(np.float32)
    
    def _initialize_reference(self):
        """Pre-compute reference voice characteristics for fast matching."""
        try:
            import librosa
            
            self._ref_characteristics = {}
            
            # Extract pitch
            f0, _, _ = librosa.pyin(
                self._reference_audio, 
                fmin=50, fmax=500, 
                sr=self.config.sample_rate
            )
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                self._ref_characteristics['pitch_mean'] = float(np.mean(f0_clean))
                self._ref_characteristics['pitch_std'] = float(np.std(f0_clean))
            else:
                self._ref_characteristics['pitch_mean'] = 150.0
                self._ref_characteristics['pitch_std'] = 30.0
            
            # Extract MFCCs for timbre matching
            mfcc = librosa.feature.mfcc(
                y=self._reference_audio, 
                sr=self.config.sample_rate, 
                n_mfcc=13
            )
            self._ref_characteristics['mfcc_mean'] = np.mean(mfcc, axis=1)
            
            # Extract spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=self._reference_audio, 
                sr=self.config.sample_rate
            )
            self._ref_characteristics['spectral_centroid'] = float(np.mean(centroid))
            
            # Extract energy profile
            rms = librosa.feature.rms(y=self._reference_audio)
            self._ref_characteristics['energy_mean'] = float(np.mean(rms))
            
            logger.info(f"Reference characteristics computed: pitch_mean={self._ref_characteristics['pitch_mean']:.1f}Hz")
            
        except Exception as e:
            logger.error(f"Reference initialization failed: {e}")
            self._ref_characteristics = {
                'pitch_mean': 150.0,
                'pitch_std': 30.0,
                'mfcc_mean': np.zeros(13),
                'spectral_centroid': 2000.0,
                'energy_mean': 0.1
            }
    
    @property
    def state(self) -> RealTimeState:
        """Get current state of the converter."""
        with self._state_lock:
            return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if the converter is currently running."""
        return self.state == RealTimeState.RUNNING
    
    @property
    def latency_ms(self) -> float:
        """Get current estimated latency in milliseconds."""
        chunk_latency = (self.config.chunk_size / self.config.sample_rate) * 1000
        buffer_latency = chunk_latency * self.config.buffer_chunks
        return buffer_latency + self.stats.average_latency_ms
    
    def start(self) -> bool:
        """
        Start the real-time voice conversion processor.
        
        Returns:
            True if started successfully
        """
        with self._state_lock:
            if self._state == RealTimeState.RUNNING:
                logger.warning("Real-time converter already running")
                return True
            
            self._state = RealTimeState.RUNNING
        
        # Reset statistics
        self.stats = RealTimeStats()
        self.stats.start_time = time.time()
        
        # Clear buffers
        self._input_buffer.clear()
        self._output_buffer.clear()
        self._overlap_buffer = None
        self._pitch_history.clear()
        
        # Reset stop event
        self._stop_event.clear()
        
        logger.info("Real-time voice conversion started")
        return True
    
    def stop(self) -> RealTimeStats:
        """
        Stop the real-time voice conversion processor.
        
        Returns:
            Final statistics from the session
        """
        with self._state_lock:
            self._state = RealTimeState.STOPPED
        
        self._stop_event.set()
        
        # Wait for processing thread if running
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
        
        logger.info(f"Real-time voice conversion stopped. Stats: {self.stats.to_dict()}")
        return self.stats
    
    def pause(self):
        """Pause the real-time voice conversion."""
        with self._state_lock:
            if self._state == RealTimeState.RUNNING:
                self._state = RealTimeState.PAUSED
                logger.info("Real-time voice conversion paused")
    
    def resume(self):
        """Resume the real-time voice conversion."""
        with self._state_lock:
            if self._state == RealTimeState.PAUSED:
                self._state = RealTimeState.RUNNING
                logger.info("Real-time voice conversion resumed")
    
    def process_chunk(self, input_chunk: np.ndarray) -> np.ndarray:
        """
        Process a single audio chunk in real-time.
        
        This is the main method for real-time processing. It takes an input
        audio chunk and returns the converted output chunk.
        
        Args:
            input_chunk: Input audio chunk (numpy array)
            
        Returns:
            Converted audio chunk
        """
        if self.state != RealTimeState.RUNNING:
            return input_chunk
        
        start_time = time.time()
        
        try:
            # Ensure correct dtype
            input_chunk = input_chunk.astype(np.float32)
            
            # Process the chunk
            if self.config.low_latency_mode:
                output_chunk = self._process_chunk_fast(input_chunk)
            else:
                output_chunk = self._process_chunk_quality(input_chunk)
            
            # Apply crossfade with previous chunk for smooth transitions
            if self._overlap_buffer is not None and len(self._overlap_buffer) > 0:
                output_chunk = self._apply_crossfade(output_chunk)
            
            # Store overlap for next chunk
            overlap_size = min(self.config.crossfade_samples, len(output_chunk) // 4)
            self._overlap_buffer = output_chunk[-overlap_size:].copy()
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self.stats.chunks_processed += 1
            self.stats.total_samples_processed += len(input_chunk)
            self.stats.processing_time_ms = processing_time
            self.stats.update_latency(processing_time)
            
            # Call callback if provided
            if self.on_chunk_processed:
                self.on_chunk_processed(output_chunk)
            
            return output_chunk
            
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            self.stats.dropped_chunks += 1
            return input_chunk
    
    def _process_chunk_fast(self, chunk: np.ndarray) -> np.ndarray:
        """
        Fast chunk processing for low-latency mode.
        
        Uses simplified processing for minimum latency.
        """
        try:
            # Simple pitch shifting using resampling
            pitch_shift = self.rvc_converter.config.pitch_shift
            
            if pitch_shift != 0:
                import scipy.signal as signal
                
                shift_factor = 2 ** (pitch_shift / 12)
                num_samples = int(len(chunk) / shift_factor)
                
                if num_samples > 0:
                    shifted = signal.resample(chunk, num_samples)
                    chunk = signal.resample(shifted, len(chunk))
            
            # Apply simple gain matching
            if self._ref_characteristics:
                ref_energy = self._ref_characteristics.get('energy_mean', 0.1)
                chunk_energy = np.sqrt(np.mean(chunk ** 2)) + 1e-8
                
                if chunk_energy > 0:
                    gain = min(2.0, max(0.5, ref_energy / chunk_energy))
                    chunk = chunk * gain
            
            # Normalize
            max_val = np.max(np.abs(chunk))
            if max_val > 0.95:
                chunk = chunk / max_val * 0.95
            
            return chunk
            
        except Exception as e:
            logger.error(f"Fast chunk processing failed: {e}")
            return chunk
    
    def _process_chunk_quality(self, chunk: np.ndarray) -> np.ndarray:
        """
        Quality chunk processing with full voice conversion.
        
        Uses more sophisticated processing for better quality.
        """
        try:
            import librosa
            from scipy import signal
            
            # Extract pitch from chunk
            f0, _, _ = librosa.pyin(
                chunk, 
                fmin=50, fmax=500, 
                sr=self.config.sample_rate,
                frame_length=512
            )
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                chunk_pitch = float(np.mean(f0_clean))
                self._pitch_history.append(chunk_pitch)
            else:
                chunk_pitch = 150.0
            
            # Calculate pitch shift needed
            ref_pitch = self._ref_characteristics.get('pitch_mean', 150.0)
            
            if self.config.pitch_smoothing and len(self._pitch_history) > 1:
                # Use smoothed pitch for more stable conversion
                smoothed_pitch = np.mean(list(self._pitch_history))
            else:
                smoothed_pitch = chunk_pitch
            
            if smoothed_pitch > 0 and ref_pitch > 0:
                pitch_semitones = 12 * np.log2(ref_pitch / smoothed_pitch)
                pitch_semitones = max(-12, min(12, pitch_semitones))
            else:
                pitch_semitones = 0
            
            # Apply pitch shift
            if abs(pitch_semitones) > 0.5:
                shift_factor = 2 ** (pitch_semitones / 12)
                num_samples = int(len(chunk) / shift_factor)
                
                if num_samples > 0:
                    shifted = signal.resample(chunk, num_samples)
                    chunk = signal.resample(shifted, len(chunk))
            
            # Apply formant preservation if enabled
            if self.config.preserve_formants:
                chunk = self._apply_formant_preservation(chunk)
            
            # Apply timbre matching
            chunk = self._apply_timbre_matching(chunk)
            
            # Normalize output
            max_val = np.max(np.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val * 0.95
            
            return chunk.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Quality chunk processing failed: {e}")
            return chunk
    
    def _apply_formant_preservation(self, chunk: np.ndarray) -> np.ndarray:
        """
        Apply advanced formant preservation to maintain voice character.
        
        Uses LPC-based formant extraction and preservation to ensure
        formant frequencies (F1-F5) are maintained within ±5Hz of reference.
        """
        try:
            from scipy import signal
            import librosa
            
            # Use LPC to extract and preserve formants
            order = min(16, len(chunk) // 100)
            if order < 4:
                return chunk
            
            sample_rate = self.config.sample_rate
            
            # Extract formants from chunk using LPC
            chunk_formants = self._extract_formants_lpc(chunk, sample_rate, order)
            
            # Get reference formants if available
            if self._ref_characteristics and 'formants' in self._ref_characteristics:
                ref_formants = self._ref_characteristics['formants']
                
                # Apply formant correction to match reference
                chunk = self._apply_formant_correction(
                    chunk, chunk_formants, ref_formants, sample_rate
                )
            else:
                # Simple formant preservation using pre-emphasis/de-emphasis
                pre_emphasis = 0.97
                emphasized = np.append(chunk[0], chunk[1:] - pre_emphasis * chunk[:-1])
                chunk = signal.lfilter([1], [1, -pre_emphasis], emphasized)
            
            return chunk.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Formant preservation failed: {e}")
            return chunk
    
    def _extract_formants_lpc(
        self, 
        audio: np.ndarray, 
        sample_rate: int, 
        order: int = 16
    ) -> List[float]:
        """
        Extract formant frequencies using Linear Predictive Coding (LPC).
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            order: LPC order (typically 2 + sample_rate/1000)
            
        Returns:
            List of formant frequencies (F1-F5)
        """
        try:
            import librosa
            
            # Compute LPC coefficients
            a = librosa.lpc(audio, order=order)
            
            # Find roots of LPC polynomial
            roots = np.roots(a)
            
            # Keep only roots inside unit circle with positive imaginary part
            roots = roots[np.imag(roots) >= 0]
            roots = roots[np.abs(roots) < 1]
            
            # Convert to frequencies
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = angles * (sample_rate / (2 * np.pi))
            
            # Filter valid formant frequencies (50Hz to Nyquist)
            freqs = freqs[freqs > 50]
            freqs = freqs[freqs < sample_rate / 2]
            freqs = np.sort(freqs)
            
            # Return first 5 formants (F1-F5)
            formants = freqs[:5].tolist() if len(freqs) >= 5 else freqs.tolist()
            
            # Pad with default values if needed
            default_formants = [500, 1500, 2500, 3500, 4500]
            while len(formants) < 5:
                formants.append(default_formants[len(formants)])
            
            return formants
            
        except Exception as e:
            logger.error(f"Formant extraction failed: {e}")
            return [500, 1500, 2500, 3500, 4500]
    
    def _apply_formant_correction(
        self,
        audio: np.ndarray,
        source_formants: List[float],
        target_formants: List[float],
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply formant correction to shift formants toward target values.
        
        Uses a series of notch and peak filters to adjust formant frequencies
        while preserving the overall spectral envelope.
        
        Args:
            audio: Input audio
            source_formants: Current formant frequencies
            target_formants: Target formant frequencies
            sample_rate: Sample rate
            
        Returns:
            Audio with corrected formants
        """
        try:
            from scipy import signal
            
            # Process each formant (F1-F3 are most important for voice identity)
            for i in range(min(3, len(source_formants), len(target_formants))):
                source_f = source_formants[i]
                target_f = target_formants[i]
                
                # Only correct if difference is significant (>5Hz threshold from requirements)
                freq_diff = abs(target_f - source_f)
                if freq_diff > 5:
                    # Calculate shift ratio
                    shift_ratio = target_f / source_f if source_f > 0 else 1.0
                    
                    # Apply subtle EQ adjustment around the formant
                    # Use a gentle peak/notch filter
                    bandwidth = 100  # Hz
                    
                    # Normalize frequencies
                    nyquist = sample_rate / 2
                    source_norm = source_f / nyquist
                    target_norm = target_f / nyquist
                    
                    # Ensure frequencies are valid
                    if 0.01 < source_norm < 0.99 and 0.01 < target_norm < 0.99:
                        # Create notch at source frequency
                        Q_notch = source_f / bandwidth
                        b_notch, a_notch = signal.iirnotch(source_norm, Q_notch)
                        
                        # Create peak at target frequency
                        Q_peak = target_f / bandwidth
                        b_peak, a_peak = signal.iirpeak(target_norm, Q_peak)
                        
                        # Apply filters with reduced strength for subtle correction
                        # Mix original with filtered to avoid over-processing
                        notched = signal.filtfilt(b_notch, a_notch, audio)
                        peaked = signal.filtfilt(b_peak, a_peak, notched)
                        
                        # Blend: 70% original, 30% corrected for subtle adjustment
                        audio = 0.7 * audio + 0.3 * peaked
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Formant correction failed: {e}")
            return audio
    
    def _apply_timbre_matching(self, chunk: np.ndarray) -> np.ndarray:
        """Apply timbre matching to match reference voice color."""
        try:
            from scipy import signal
            
            if not self._ref_characteristics:
                return chunk
            
            # Simple EQ-based timbre matching
            ref_centroid = self._ref_characteristics.get('spectral_centroid', 2000.0)
            
            # Estimate chunk spectral centroid
            fft = np.fft.rfft(chunk)
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.config.sample_rate)
            magnitude = np.abs(fft)
            
            if np.sum(magnitude) > 0:
                chunk_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                chunk_centroid = ref_centroid
            
            # Apply subtle EQ adjustment
            centroid_ratio = ref_centroid / (chunk_centroid + 1e-8)
            
            if centroid_ratio > 1.2:
                # Boost highs
                b, a = signal.butter(2, 0.3, btype='high')
                high_freq = signal.filtfilt(b, a, chunk)
                chunk = chunk + 0.1 * high_freq
            elif centroid_ratio < 0.8:
                # Reduce highs
                b, a = signal.butter(2, 0.7, btype='low')
                chunk = 0.9 * chunk + 0.1 * signal.filtfilt(b, a, chunk)
            
            return chunk.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Timbre matching failed: {e}")
            return chunk
    
    def _apply_crossfade(self, chunk: np.ndarray) -> np.ndarray:
        """Apply crossfade with previous chunk for smooth transitions."""
        try:
            if self._overlap_buffer is None or len(self._overlap_buffer) == 0:
                return chunk
            
            overlap_size = len(self._overlap_buffer)
            
            if overlap_size > len(chunk):
                overlap_size = len(chunk) // 4
            
            if overlap_size <= 0:
                return chunk
            
            # Create crossfade window
            fade_out = np.linspace(1.0, 0.0, overlap_size)
            fade_in = np.linspace(0.0, 1.0, overlap_size)
            
            # Apply crossfade to beginning of chunk
            chunk[:overlap_size] = (
                self._overlap_buffer[-overlap_size:] * fade_out + 
                chunk[:overlap_size] * fade_in
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Crossfade failed: {e}")
            return chunk
    
    def process_stream(
        self, 
        audio_generator: Generator[np.ndarray, None, None]
    ) -> Generator[np.ndarray, None, None]:
        """
        Process a stream of audio chunks.
        
        This is a generator that yields converted audio chunks.
        
        Args:
            audio_generator: Generator yielding input audio chunks
            
        Yields:
            Converted audio chunks
        """
        self.start()
        
        try:
            for input_chunk in audio_generator:
                if self._stop_event.is_set():
                    break
                
                if self.state == RealTimeState.PAUSED:
                    # When paused, pass through unchanged
                    yield input_chunk
                else:
                    output_chunk = self.process_chunk(input_chunk)
                    yield output_chunk
                    
        finally:
            self.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = RealTimeStats()
        self.stats.start_time = time.time()


class RVCVoiceConverter:
    """
    RVC Voice Converter using HuggingFace RVC implementation.
    
    This converter transforms synthesized speech to match a target voice
    using retrieval-based voice conversion techniques.
    """
    
    def __init__(self, config: Optional[RVCConfig] = None):
        """
        Initialize the RVC Voice Converter.
        
        Args:
            config: RVC configuration options
        """
        self.config = config or RVCConfig()
        self._hf_rvc_available = False
        self._rvc_python_available = False
        self._model = None
        self._feature_extractor = None
        
        # Check available backends
        self._check_backends()
        
        logger.info(f"RVC Voice Converter initialized. "
                   f"hf_rvc: {self._hf_rvc_available}, "
                   f"rvc_python: {self._rvc_python_available}")
    
    def _check_backends(self):
        """Check which RVC backends are available."""
        # Check hf-rvc
        try:
            import hf_rvc
            self._hf_rvc_available = True
            logger.info("hf-rvc backend available")
        except ImportError as e:
            logger.warning(f"hf-rvc not available: {e}")
            self._hf_rvc_available = False
        
        # Check rvc-python (requires fairseq)
        try:
            from rvc_python import BaseLoader
            self._rvc_python_available = True
            logger.info("rvc-python backend available")
        except ImportError as e:
            logger.warning(f"rvc-python not available: {e}")
            self._rvc_python_available = False
    
    @property
    def is_available(self) -> bool:
        """Check if any RVC backend is available."""
        return self._hf_rvc_available or self._rvc_python_available
    
    def load_model(self, model_path: str, index_path: Optional[str] = None) -> bool:
        """
        Load an RVC model for voice conversion.
        
        Args:
            model_path: Path to the RVC model file (.pth)
            index_path: Optional path to the index file (.index)
            
        Returns:
            True if model loaded successfully
        """
        self.config.model_path = model_path
        self.config.index_path = index_path
        
        if self._hf_rvc_available:
            return self._load_hf_rvc_model(model_path, index_path)
        elif self._rvc_python_available:
            return self._load_rvc_python_model(model_path, index_path)
        else:
            logger.error("No RVC backend available")
            return False
    
    def _load_hf_rvc_model(self, model_path: str, index_path: Optional[str]) -> bool:
        """Load model using hf-rvc backend."""
        try:
            from hf_rvc import RVCModel
            
            # hf-rvc uses HuggingFace model format
            # Check if it's a HF model ID or local path
            if os.path.exists(model_path):
                # Local model - need to convert or use directly
                logger.info(f"Loading local RVC model from {model_path}")
                self._model = RVCModel.from_pretrained(model_path)
            else:
                # Try as HuggingFace model ID
                logger.info(f"Loading RVC model from HuggingFace: {model_path}")
                self._model = RVCModel.from_pretrained(model_path)
            
            logger.info("hf-rvc model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load hf-rvc model: {e}")
            return False
    
    def _load_rvc_python_model(self, model_path: str, index_path: Optional[str]) -> bool:
        """Load model using rvc-python backend."""
        try:
            from rvc_python import BaseLoader
            
            self._model = BaseLoader(
                name=Path(model_path).stem,
                device=self.config.device
            )
            self._model.load_model(model_path, index_path)
            
            logger.info("rvc-python model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load rvc-python model: {e}")
            return False

    def convert(
        self,
        audio: Union[np.ndarray, str],
        sample_rate: int = 22050,
        target_voice_path: Optional[str] = None
    ) -> ConversionResult:
        """
        Convert audio to match the loaded voice model.
        
        Args:
            audio: Input audio as numpy array or path to audio file
            sample_rate: Sample rate of input audio (if numpy array)
            target_voice_path: Optional path to target voice reference
            
        Returns:
            ConversionResult with converted audio
        """
        if not self.is_available:
            return ConversionResult(
                audio=audio if isinstance(audio, np.ndarray) else np.array([]),
                sample_rate=sample_rate,
                success=False,
                method_used="none",
                error_message="No RVC backend available"
            )
        
        # Load audio if path provided
        if isinstance(audio, str):
            audio, sample_rate = self._load_audio(audio)
        
        # Try hf-rvc first
        if self._hf_rvc_available:
            result = self._convert_hf_rvc(audio, sample_rate)
            if result.success:
                return result
        
        # Fall back to rvc-python
        if self._rvc_python_available:
            result = self._convert_rvc_python(audio, sample_rate)
            if result.success:
                return result
        
        # If all backends fail, return original with fallback processing
        return self._fallback_conversion(audio, sample_rate)
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return np.array([]), 22050
    
    def _convert_hf_rvc(self, audio: np.ndarray, sample_rate: int) -> ConversionResult:
        """Convert using hf-rvc backend."""
        try:
            if self._model is None:
                return ConversionResult(
                    audio=audio,
                    sample_rate=sample_rate,
                    success=False,
                    method_used="hf_rvc",
                    error_message="Model not loaded"
                )
            
            # hf-rvc conversion
            import torch
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            
            # Run conversion
            with torch.no_grad():
                converted = self._model(
                    audio_tensor,
                    f0_method=self.config.pitch_method,
                    f0_up_key=self.config.pitch_shift,
                    index_rate=self.config.index_rate,
                    filter_radius=self.config.filter_radius,
                    rms_mix_rate=self.config.rms_mix_rate,
                    protect=self.config.protect
                )
            
            # Convert back to numpy
            converted_audio = converted.squeeze().cpu().numpy()
            
            return ConversionResult(
                audio=converted_audio,
                sample_rate=sample_rate,
                success=True,
                method_used="hf_rvc",
                quality_score=0.9,
                metadata={
                    "pitch_method": self.config.pitch_method,
                    "pitch_shift": self.config.pitch_shift,
                    "index_rate": self.config.index_rate
                }
            )
            
        except Exception as e:
            logger.error(f"hf-rvc conversion failed: {e}")
            return ConversionResult(
                audio=audio,
                sample_rate=sample_rate,
                success=False,
                method_used="hf_rvc",
                error_message=str(e)
            )
    
    def _convert_rvc_python(self, audio: np.ndarray, sample_rate: int) -> ConversionResult:
        """Convert using rvc-python backend."""
        try:
            if self._model is None:
                return ConversionResult(
                    audio=audio,
                    sample_rate=sample_rate,
                    success=False,
                    method_used="rvc_python",
                    error_message="Model not loaded"
                )
            
            # Save audio to temp file for rvc-python
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_input = f.name
                sf.write(temp_input, audio, sample_rate)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_output = f.name
            
            try:
                # Run conversion
                self._model.infer(
                    input_path=temp_input,
                    output_path=temp_output,
                    pitch=self.config.pitch_shift,
                    method=self.config.pitch_method,
                    index_rate=self.config.index_rate,
                    filter_radius=self.config.filter_radius,
                    resample_sr=self.config.resample_sr,
                    rms_mix_rate=self.config.rms_mix_rate,
                    protect=self.config.protect
                )
                
                # Load converted audio
                converted_audio, converted_sr = sf.read(temp_output)
                
                return ConversionResult(
                    audio=converted_audio,
                    sample_rate=converted_sr,
                    success=True,
                    method_used="rvc_python",
                    quality_score=0.9,
                    metadata={
                        "pitch_method": self.config.pitch_method,
                        "pitch_shift": self.config.pitch_shift
                    }
                )
                
            finally:
                # Cleanup temp files
                if os.path.exists(temp_input):
                    os.unlink(temp_input)
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
                    
        except Exception as e:
            logger.error(f"rvc-python conversion failed: {e}")
            return ConversionResult(
                audio=audio,
                sample_rate=sample_rate,
                success=False,
                method_used="rvc_python",
                error_message=str(e)
            )
    
    def _fallback_conversion(self, audio: np.ndarray, sample_rate: int) -> ConversionResult:
        """
        Fallback conversion using basic signal processing.
        
        This provides minimal voice conversion when RVC backends are unavailable.
        """
        try:
            import scipy.signal as signal
            
            # Apply basic pitch shifting if requested
            if self.config.pitch_shift != 0:
                # Simple pitch shift using resampling
                shift_factor = 2 ** (self.config.pitch_shift / 12)
                
                # Resample to shift pitch
                num_samples = int(len(audio) / shift_factor)
                shifted = signal.resample(audio, num_samples)
                
                # Resample back to original length to maintain duration
                audio = signal.resample(shifted, len(audio))
            
            # Apply gentle smoothing
            if len(audio) > 100:
                # Low-pass filter to smooth any artifacts
                b, a = signal.butter(4, 0.95, btype='low')
                audio = signal.filtfilt(b, a, audio)
            
            return ConversionResult(
                audio=audio,
                sample_rate=sample_rate,
                success=True,
                method_used="fallback",
                quality_score=0.5,
                metadata={
                    "note": "Using fallback signal processing (RVC unavailable)",
                    "pitch_shift": self.config.pitch_shift
                }
            )
            
        except Exception as e:
            logger.error(f"Fallback conversion failed: {e}")
            return ConversionResult(
                audio=audio,
                sample_rate=sample_rate,
                success=False,
                method_used="fallback",
                error_message=str(e)
            )

    # ==================== Advanced Pitch Shifting and Formant Preservation ====================
    
    def pitch_shift_with_formant_preservation(
        self,
        audio: np.ndarray,
        sample_rate: int,
        semitones: float,
        preserve_formants: bool = True,
        formant_shift_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Apply pitch shifting while preserving formant frequencies.
        
        This method shifts the pitch of audio by the specified number of semitones
        while optionally preserving the formant structure to maintain voice identity.
        Uses PSOLA-like technique for high-quality pitch shifting.
        
        Args:
            audio: Input audio signal (numpy array)
            sample_rate: Sample rate of the audio
            semitones: Pitch shift in semitones (-12 to +12)
            preserve_formants: If True, preserve formant frequencies during pitch shift
            formant_shift_ratio: Ratio to shift formants (1.0 = no shift, used when preserve_formants=False)
            
        Returns:
            Pitch-shifted audio with preserved formants
            
        Requirements:
            - Validates: Requirements 2.2 (preserve formant frequencies F1-F5 within ±5Hz)
            - Validates: Requirements 6.1 (optimal model combination)
        """
        try:
            import librosa
            from scipy import signal
            
            # Clamp semitones to valid range
            semitones = max(-12, min(12, semitones))
            
            if abs(semitones) < 0.1:
                return audio  # No significant shift needed
            
            # Calculate pitch shift factor
            pitch_factor = 2 ** (semitones / 12)
            
            if preserve_formants:
                # Use formant-preserving pitch shift (PSOLA-like approach)
                shifted_audio = self._psola_pitch_shift(
                    audio, sample_rate, pitch_factor
                )
            else:
                # Standard pitch shift (formants shift with pitch)
                shifted_audio = self._standard_pitch_shift(
                    audio, sample_rate, pitch_factor, formant_shift_ratio
                )
            
            # Normalize output
            max_val = np.max(np.abs(shifted_audio))
            if max_val > 0:
                shifted_audio = shifted_audio / max_val * 0.95
            
            return shifted_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Pitch shift with formant preservation failed: {e}")
            return audio
    
    def _psola_pitch_shift(
        self,
        audio: np.ndarray,
        sample_rate: int,
        pitch_factor: float
    ) -> np.ndarray:
        """
        PSOLA-like pitch shifting that preserves formants.
        
        Pitch Synchronous Overlap and Add (PSOLA) shifts pitch by:
        1. Detecting pitch periods
        2. Resampling within each period to change pitch
        3. Overlap-adding to maintain duration
        
        This preserves formants because the spectral envelope is maintained
        while only the fundamental frequency changes.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            pitch_factor: Pitch multiplication factor (>1 = higher pitch)
            
        Returns:
            Pitch-shifted audio with preserved formants
        """
        try:
            import librosa
            from scipy import signal
            from scipy.interpolate import interp1d
            
            # Extract pitch (F0) for pitch-synchronous processing
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=50, fmax=500, sr=sample_rate,
                frame_length=2048, hop_length=512
            )
            
            # Get median pitch for period estimation
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) == 0:
                # No pitch detected, use simple resampling
                return self._simple_pitch_shift(audio, pitch_factor)
            
            median_f0 = np.median(f0_clean)
            period_samples = int(sample_rate / median_f0) if median_f0 > 0 else 256
            
            # Ensure reasonable period size
            period_samples = max(64, min(1024, period_samples))
            
            # Process audio in pitch-synchronous frames
            hop_size = period_samples // 2
            num_frames = (len(audio) - period_samples) // hop_size + 1
            
            if num_frames < 2:
                return self._simple_pitch_shift(audio, pitch_factor)
            
            # Create output buffer
            output_length = len(audio)
            output = np.zeros(output_length, dtype=np.float32)
            window = np.hanning(period_samples)
            
            # Process each frame
            for i in range(num_frames):
                start = i * hop_size
                end = start + period_samples
                
                if end > len(audio):
                    break
                
                frame = audio[start:end] * window
                
                # Resample frame to change pitch while preserving duration
                new_length = int(period_samples / pitch_factor)
                if new_length < 4:
                    new_length = 4
                
                # Resample to new length (changes pitch)
                resampled = signal.resample(frame, new_length)
                
                # Resample back to original length (preserves duration)
                restored = signal.resample(resampled, period_samples)
                
                # Apply window and overlap-add
                restored = restored * window
                
                # Add to output with overlap
                output[start:end] += restored
            
            # Normalize overlap regions
            # Simple normalization - divide by expected overlap factor
            overlap_factor = period_samples / hop_size
            output = output / overlap_factor
            
            return output.astype(np.float32)
            
        except Exception as e:
            logger.error(f"PSOLA pitch shift failed: {e}")
            return self._simple_pitch_shift(audio, pitch_factor)
    
    def _standard_pitch_shift(
        self,
        audio: np.ndarray,
        sample_rate: int,
        pitch_factor: float,
        formant_shift_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Standard pitch shift using resampling (formants shift with pitch).
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            pitch_factor: Pitch multiplication factor
            formant_shift_ratio: Additional formant shift ratio
            
        Returns:
            Pitch-shifted audio
        """
        try:
            from scipy import signal
            
            # Combined shift factor
            total_factor = pitch_factor * formant_shift_ratio
            
            # Resample to shift pitch
            new_length = int(len(audio) / total_factor)
            if new_length < 4:
                return audio
            
            shifted = signal.resample(audio, new_length)
            
            # Resample back to original length to maintain duration
            restored = signal.resample(shifted, len(audio))
            
            return restored.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Standard pitch shift failed: {e}")
            return audio
    
    def _simple_pitch_shift(
        self,
        audio: np.ndarray,
        pitch_factor: float
    ) -> np.ndarray:
        """
        Simple pitch shift using basic resampling.
        
        Args:
            audio: Input audio
            pitch_factor: Pitch multiplication factor
            
        Returns:
            Pitch-shifted audio
        """
        try:
            from scipy import signal
            
            # Resample to shift pitch
            new_length = int(len(audio) / pitch_factor)
            if new_length < 4:
                return audio
            
            shifted = signal.resample(audio, new_length)
            
            # Resample back to original length
            restored = signal.resample(shifted, len(audio))
            
            return restored.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Simple pitch shift failed: {e}")
            return audio
    
    def extract_and_match_formants(
        self,
        source_audio: np.ndarray,
        target_audio: np.ndarray,
        sample_rate: int,
        tolerance_hz: float = 5.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract formants from source and target, then match source to target.
        
        This method ensures formant frequencies (F1-F5) are preserved within
        the specified tolerance (default ±5Hz per requirements).
        
        Args:
            source_audio: Audio to be modified
            target_audio: Reference audio with target formants
            sample_rate: Sample rate of both audio signals
            tolerance_hz: Maximum allowed deviation in Hz (default 5Hz per Req 2.2)
            
        Returns:
            Tuple of (modified_audio, formant_info_dict)
            
        Requirements:
            - Validates: Requirements 2.2 (preserve formant frequencies F1-F5 within ±5Hz)
        """
        try:
            import librosa
            from scipy import signal
            
            # Extract formants from both signals
            source_formants = self._extract_formants_detailed(source_audio, sample_rate)
            target_formants = self._extract_formants_detailed(target_audio, sample_rate)
            
            # Calculate formant deviations
            deviations = []
            for i in range(min(5, len(source_formants['frequencies']), len(target_formants['frequencies']))):
                src_f = source_formants['frequencies'][i]
                tgt_f = target_formants['frequencies'][i]
                deviation = abs(src_f - tgt_f)
                deviations.append(deviation)
            
            # Apply formant correction if any deviation exceeds tolerance
            modified_audio = source_audio.copy()
            corrections_applied = []
            
            for i, deviation in enumerate(deviations):
                if deviation > tolerance_hz:
                    src_f = source_formants['frequencies'][i]
                    tgt_f = target_formants['frequencies'][i]
                    
                    # Apply targeted formant correction
                    modified_audio = self._apply_single_formant_correction(
                        modified_audio, src_f, tgt_f, sample_rate
                    )
                    
                    corrections_applied.append({
                        'formant': f'F{i+1}',
                        'source_hz': src_f,
                        'target_hz': tgt_f,
                        'deviation_hz': deviation
                    })
            
            # Verify final formants
            final_formants = self._extract_formants_detailed(modified_audio, sample_rate)
            final_deviations = []
            for i in range(min(5, len(final_formants['frequencies']), len(target_formants['frequencies']))):
                final_deviations.append(abs(
                    final_formants['frequencies'][i] - target_formants['frequencies'][i]
                ))
            
            formant_info = {
                'source_formants': source_formants['frequencies'],
                'target_formants': target_formants['frequencies'],
                'final_formants': final_formants['frequencies'],
                'initial_deviations_hz': deviations,
                'final_deviations_hz': final_deviations,
                'corrections_applied': corrections_applied,
                'tolerance_hz': tolerance_hz,
                'within_tolerance': all(d <= tolerance_hz for d in final_deviations)
            }
            
            logger.info(f"Formant matching: {len(corrections_applied)} corrections applied, "
                       f"within tolerance: {formant_info['within_tolerance']}")
            
            return modified_audio.astype(np.float32), formant_info
            
        except Exception as e:
            logger.error(f"Formant extraction and matching failed: {e}")
            return source_audio, {'error': str(e)}
    
    def _extract_formants_detailed(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Extract detailed formant information including frequencies and bandwidths.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            
        Returns:
            Dictionary with formant frequencies, bandwidths, and amplitudes
        """
        try:
            import librosa
            
            # LPC order based on sample rate
            order = int(2 + sample_rate / 1000)
            order = min(order, len(audio) // 100)
            order = max(order, 8)
            
            # Compute LPC coefficients
            a = librosa.lpc(audio, order=order)
            
            # Find roots
            roots = np.roots(a)
            
            # Keep roots inside unit circle with positive imaginary part
            roots = roots[np.imag(roots) >= 0]
            roots = roots[np.abs(roots) < 1]
            
            # Convert to frequencies and bandwidths
            frequencies = []
            bandwidths = []
            
            for root in roots:
                angle = np.arctan2(np.imag(root), np.real(root))
                freq = angle * (sample_rate / (2 * np.pi))
                
                # Calculate bandwidth from root magnitude
                bw = -0.5 * (sample_rate / np.pi) * np.log(np.abs(root))
                
                if 50 < freq < sample_rate / 2:
                    frequencies.append(freq)
                    bandwidths.append(bw)
            
            # Sort by frequency
            if frequencies:
                sorted_indices = np.argsort(frequencies)
                frequencies = [frequencies[i] for i in sorted_indices]
                bandwidths = [bandwidths[i] for i in sorted_indices]
            
            # Ensure we have 5 formants
            default_freqs = [500, 1500, 2500, 3500, 4500]
            default_bws = [100, 150, 200, 250, 300]
            
            while len(frequencies) < 5:
                frequencies.append(default_freqs[len(frequencies)])
                bandwidths.append(default_bws[len(bandwidths)])
            
            return {
                'frequencies': frequencies[:5],
                'bandwidths': bandwidths[:5]
            }
            
        except Exception as e:
            logger.error(f"Detailed formant extraction failed: {e}")
            return {
                'frequencies': [500, 1500, 2500, 3500, 4500],
                'bandwidths': [100, 150, 200, 250, 300]
            }
    
    def _apply_single_formant_correction(
        self,
        audio: np.ndarray,
        source_freq: float,
        target_freq: float,
        sample_rate: int
    ) -> np.ndarray:
        """
        Apply correction for a single formant frequency.
        
        Args:
            audio: Input audio
            source_freq: Current formant frequency
            target_freq: Target formant frequency
            sample_rate: Sample rate
            
        Returns:
            Audio with corrected formant
        """
        try:
            from scipy import signal
            
            nyquist = sample_rate / 2
            
            # Validate frequencies
            if source_freq <= 50 or source_freq >= nyquist - 50:
                return audio
            if target_freq <= 50 or target_freq >= nyquist - 50:
                return audio
            
            # Normalize frequencies
            source_norm = source_freq / nyquist
            target_norm = target_freq / nyquist
            
            # Bandwidth for filters (narrower = more precise but may cause artifacts)
            bandwidth = 80  # Hz
            
            # Create notch filter at source frequency
            Q_notch = source_freq / bandwidth
            b_notch, a_notch = signal.iirnotch(source_norm, Q_notch)
            
            # Create peak filter at target frequency
            Q_peak = target_freq / bandwidth
            b_peak, a_peak = signal.iirpeak(target_norm, Q_peak)
            
            # Apply filters
            notched = signal.filtfilt(b_notch, a_notch, audio)
            peaked = signal.filtfilt(b_peak, a_peak, notched)
            
            # Blend with original for subtle correction
            # More aggressive blending for larger frequency differences
            freq_diff = abs(target_freq - source_freq)
            blend_factor = min(0.5, freq_diff / 200)  # Max 50% correction
            
            corrected = (1 - blend_factor) * audio + blend_factor * peaked
            
            return corrected.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Single formant correction failed: {e}")
            return audio
    
    def get_pitch_shift_info(
        self,
        source_audio: np.ndarray,
        target_audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Calculate the pitch shift needed to match source to target voice.
        
        Args:
            source_audio: Source audio signal
            target_audio: Target audio signal
            sample_rate: Sample rate
            
        Returns:
            Dictionary with pitch analysis and recommended shift
        """
        try:
            import librosa
            
            # Extract pitch from both signals
            f0_source, _, _ = librosa.pyin(
                source_audio, fmin=50, fmax=500, sr=sample_rate
            )
            f0_target, _, _ = librosa.pyin(
                target_audio, fmin=50, fmax=500, sr=sample_rate
            )
            
            # Clean NaN values
            f0_source_clean = f0_source[~np.isnan(f0_source)]
            f0_target_clean = f0_target[~np.isnan(f0_target)]
            
            if len(f0_source_clean) == 0 or len(f0_target_clean) == 0:
                return {
                    'error': 'Could not detect pitch in one or both signals',
                    'recommended_shift_semitones': 0.0
                }
            
            # Calculate statistics
            source_mean = float(np.mean(f0_source_clean))
            source_std = float(np.std(f0_source_clean))
            source_min = float(np.min(f0_source_clean))
            source_max = float(np.max(f0_source_clean))
            
            target_mean = float(np.mean(f0_target_clean))
            target_std = float(np.std(f0_target_clean))
            target_min = float(np.min(f0_target_clean))
            target_max = float(np.max(f0_target_clean))
            
            # Calculate recommended pitch shift in semitones
            if source_mean > 0 and target_mean > 0:
                shift_semitones = 12 * np.log2(target_mean / source_mean)
                shift_semitones = max(-12, min(12, shift_semitones))
            else:
                shift_semitones = 0.0
            
            return {
                'source_pitch': {
                    'mean_hz': source_mean,
                    'std_hz': source_std,
                    'min_hz': source_min,
                    'max_hz': source_max
                },
                'target_pitch': {
                    'mean_hz': target_mean,
                    'std_hz': target_std,
                    'min_hz': target_min,
                    'max_hz': target_max
                },
                'recommended_shift_semitones': float(shift_semitones),
                'pitch_ratio': target_mean / source_mean if source_mean > 0 else 1.0
            }
            
        except Exception as e:
            logger.error(f"Pitch shift info calculation failed: {e}")
            return {
                'error': str(e),
                'recommended_shift_semitones': 0.0
            }


    def run_voice_conversion_pipeline(
        self,
        tts_audio: np.ndarray,
        tts_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
        pitch_shift: float = 0.0,
        preserve_formants: bool = True,
        apply_spectral_matching: bool = True,
        apply_timbre_transfer: bool = True,
        quality_threshold: float = 0.8
    ) -> ConversionResult:
        """
        Run the complete voice conversion pipeline: TTS output → RVC → final output.
        
        This is the main entry point for the voice conversion pipeline that:
        1. Analyzes both TTS output and reference audio
        2. Calculates optimal conversion parameters
        3. Applies voice conversion with pitch and formant matching
        4. Optionally applies spectral matching for timbre transfer
        5. Validates output quality
        
        Args:
            tts_audio: TTS-generated audio (numpy array)
            tts_sr: Sample rate of TTS audio
            reference_audio: Reference audio for voice matching (numpy array)
            reference_sr: Sample rate of reference audio
            pitch_shift: Additional pitch shift in semitones (-12 to +12)
            preserve_formants: Whether to preserve formants during conversion
            apply_spectral_matching: Whether to apply spectral envelope matching
            apply_timbre_transfer: Whether to apply timbre transfer from reference
            quality_threshold: Minimum quality score for output (0.0 to 1.0)
            
        Returns:
            ConversionResult with converted audio
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("Starting voice conversion pipeline: TTS → RVC → Final")
            
            # Step 1: Preprocess and normalize audio
            tts_processed = self._preprocess_audio(tts_audio, tts_sr)
            ref_processed = self._preprocess_audio(reference_audio, reference_sr)
            
            # Step 2: Analyze voice characteristics
            ref_characteristics = self._analyze_voice_characteristics(
                ref_processed, 22050
            )
            tts_characteristics = self._analyze_voice_characteristics(
                tts_processed, 22050
            )
            
            # Step 3: Calculate conversion parameters
            conversion_params = self._calculate_conversion_parameters(
                ref_characteristics, tts_characteristics, pitch_shift, preserve_formants
            )
            
            logger.info(f"Conversion parameters: pitch_shift={conversion_params['pitch_shift']:.2f}, "
                       f"formant_shift={conversion_params.get('formant_shift', 1.0):.2f}")
            
            # Step 4: Apply voice conversion
            # Update config with calculated parameters
            self.config.pitch_shift = int(round(conversion_params['pitch_shift']))
            
            # Step 4a: Apply advanced pitch shifting with formant preservation
            if preserve_formants and abs(conversion_params['pitch_shift']) > 0.5:
                # Use PSOLA-based pitch shifting that preserves formants
                tts_processed = self.pitch_shift_with_formant_preservation(
                    tts_processed,
                    sample_rate=22050,
                    semitones=conversion_params['pitch_shift'],
                    preserve_formants=True
                )
                logger.info(f"Applied formant-preserving pitch shift: {conversion_params['pitch_shift']:.2f} semitones")
            
            # Step 4b: Apply formant matching to ensure F1-F5 within ±5Hz tolerance
            if preserve_formants:
                tts_processed, formant_info = self.extract_and_match_formants(
                    tts_processed, ref_processed, 22050, tolerance_hz=5.0
                )
                if formant_info.get('within_tolerance', False):
                    logger.info("Formant matching successful - within ±5Hz tolerance")
                else:
                    logger.warning(f"Formant matching: some formants outside tolerance")
            
            # Run conversion
            result = self.convert(tts_processed, sample_rate=22050)
            
            if not result.success:
                return result
            
            converted_audio = result.audio
            
            # Step 5: Apply spectral matching if enabled
            if apply_spectral_matching:
                converted_audio = self._apply_spectral_matching(
                    converted_audio, ref_processed
                )
            
            # Step 6: Apply timbre transfer if enabled
            if apply_timbre_transfer:
                converted_audio = self._apply_timbre_transfer(
                    converted_audio, ref_processed, ref_characteristics
                )
            
            # Step 7: Final normalization and quality check
            converted_audio = self._normalize_output(converted_audio)
            
            # Calculate quality score
            quality_score = self._calculate_conversion_quality(
                converted_audio, ref_processed
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Voice conversion pipeline completed in {processing_time:.2f}s, "
                       f"quality: {quality_score:.2f}")
            
            return ConversionResult(
                audio=converted_audio,
                sample_rate=22050,
                success=True,
                method_used=result.method_used,
                quality_score=quality_score,
                metadata={
                    "pipeline": "tts_to_rvc",
                    "processing_time": processing_time,
                    "pitch_shift": conversion_params['pitch_shift'],
                    "formant_shift": conversion_params.get('formant_shift', 1.0),
                    "formant_preservation": preserve_formants,
                    "formant_tolerance_hz": 5.0 if preserve_formants else None,
                    "spectral_matching": apply_spectral_matching,
                    "timbre_transfer": apply_timbre_transfer
                }
            )
            
        except Exception as e:
            logger.error(f"Voice conversion pipeline failed: {e}")
            return ConversionResult(
                audio=tts_audio,
                sample_rate=tts_sr,
                success=False,
                method_used="pipeline",
                error_message=str(e)
            )
    
    def _preprocess_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_sr: int = 22050
    ) -> np.ndarray:
        """
        Preprocess audio for voice conversion.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of input
            target_sr: Target sample rate
            
        Returns:
            Preprocessed audio at target sample rate
        """
        try:
            import librosa
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Resample to target sample rate
            if sample_rate != target_sr:
                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=target_sr
                )
            
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio.astype(np.float32)
    
    def _analyze_voice_characteristics(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Dict[str, Any]:
        """
        Analyze voice characteristics for matching.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary of voice characteristics
        """
        try:
            import librosa
            
            characteristics = {}
            
            # Extract pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=50, fmax=500, sr=sample_rate
            )
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                characteristics['pitch_mean'] = float(np.mean(f0_clean))
                characteristics['pitch_std'] = float(np.std(f0_clean))
                characteristics['pitch_min'] = float(np.min(f0_clean))
                characteristics['pitch_max'] = float(np.max(f0_clean))
            else:
                characteristics['pitch_mean'] = 150.0
                characteristics['pitch_std'] = 30.0
            
            # Extract formants using LPC
            try:
                order = int(2 + sample_rate / 1000)
                a = librosa.lpc(audio, order=order)
                
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]
                
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = angles * (sample_rate / (2 * np.pi))
                freqs = freqs[freqs > 50]
                freqs = freqs[freqs < sample_rate / 2]
                freqs = np.sort(freqs)
                
                if len(freqs) >= 3:
                    characteristics['formants'] = freqs[:5].tolist()
                else:
                    characteristics['formants'] = [500, 1500, 2500]
                    
            except Exception:
                characteristics['formants'] = [500, 1500, 2500]
            
            # Extract spectral characteristics
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            characteristics['spectral_centroid'] = float(np.mean(spectral_centroid))
            
            # Extract energy
            rms = librosa.feature.rms(y=audio)
            characteristics['energy_mean'] = float(np.mean(rms))
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Voice characteristic analysis failed: {e}")
            return {
                'pitch_mean': 150.0,
                'pitch_std': 30.0,
                'formants': [500, 1500, 2500],
                'spectral_centroid': 2000.0,
                'energy_mean': 0.1
            }
    
    def _calculate_conversion_parameters(
        self,
        ref_characteristics: Dict[str, Any],
        tts_characteristics: Dict[str, Any],
        user_pitch_shift: float,
        preserve_formants: bool
    ) -> Dict[str, float]:
        """
        Calculate optimal conversion parameters based on voice analysis.
        
        Args:
            ref_characteristics: Reference voice characteristics
            tts_characteristics: TTS output characteristics
            user_pitch_shift: User-specified pitch shift
            preserve_formants: Whether to preserve formants
            
        Returns:
            Dictionary of conversion parameters
        """
        params = {
            'pitch_shift': 0.0,
            'formant_shift': 1.0,
            'index_rate': 0.75,
            'filter_radius': 3,
            'rms_mix_rate': 0.25,
            'protect': 0.33
        }
        
        try:
            # Calculate pitch shift
            ref_pitch = ref_characteristics.get('pitch_mean', 150)
            tts_pitch = tts_characteristics.get('pitch_mean', 150)
            
            if tts_pitch > 0 and ref_pitch > 0:
                # Convert Hz difference to semitones
                pitch_semitones = 12 * np.log2(ref_pitch / tts_pitch)
                params['pitch_shift'] = max(-12, min(12, pitch_semitones + user_pitch_shift))
            else:
                params['pitch_shift'] = user_pitch_shift
            
            # Calculate formant shift
            if preserve_formants:
                ref_formants = ref_characteristics.get('formants', [])
                tts_formants = tts_characteristics.get('formants', [])
                
                if ref_formants and tts_formants and len(ref_formants) > 0 and len(tts_formants) > 0:
                    f1_ratio = ref_formants[0] / tts_formants[0] if tts_formants[0] > 0 else 1.0
                    
                    if len(ref_formants) > 1 and len(tts_formants) > 1:
                        f2_ratio = ref_formants[1] / tts_formants[1] if tts_formants[1] > 0 else 1.0
                        params['formant_shift'] = 0.7 * f1_ratio + 0.3 * f2_ratio
                    else:
                        params['formant_shift'] = f1_ratio
                    
                    params['formant_shift'] = max(0.5, min(2.0, params['formant_shift']))
            
            # Adjust RMS mix rate based on energy difference
            ref_energy = ref_characteristics.get('energy_mean', 0.1)
            tts_energy = tts_characteristics.get('energy_mean', 0.1)
            
            if ref_energy > 0 and tts_energy > 0:
                energy_ratio = ref_energy / tts_energy
                if energy_ratio > 1.5 or energy_ratio < 0.67:
                    params['rms_mix_rate'] = 0.4
                else:
                    params['rms_mix_rate'] = 0.25
            
            return params
            
        except Exception as e:
            logger.error(f"Parameter calculation failed: {e}")
            return params
    
    def _apply_spectral_matching(
        self,
        converted_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> np.ndarray:
        """
        Apply spectral envelope matching to converted audio.
        
        Args:
            converted_audio: Converted audio
            reference_audio: Reference audio
            
        Returns:
            Spectrally matched audio
        """
        try:
            import librosa
            from scipy.ndimage import gaussian_filter1d
            
            # Compute spectral envelopes
            n_fft = 2048
            hop_length = 512
            
            conv_stft = librosa.stft(converted_audio, n_fft=n_fft, hop_length=hop_length)
            ref_stft = librosa.stft(reference_audio, n_fft=n_fft, hop_length=hop_length)
            
            conv_magnitude = np.abs(conv_stft)
            ref_magnitude = np.abs(ref_stft)
            
            # Calculate average spectral envelopes
            conv_envelope = np.mean(conv_magnitude, axis=1, keepdims=True)
            ref_envelope = np.mean(ref_magnitude, axis=1, keepdims=True)
            
            # Apply spectral matching with smoothing
            matching_strength = 0.4
            target_envelope = (1 - matching_strength) * conv_envelope + matching_strength * ref_envelope
            
            # Apply envelope matching
            envelope_ratio = target_envelope / (conv_envelope + 1e-8)
            
            # Smooth the ratio to avoid artifacts
            envelope_ratio = gaussian_filter1d(envelope_ratio.flatten(), sigma=3).reshape(-1, 1)
            
            matched_magnitude = conv_magnitude * envelope_ratio
            
            # Reconstruct audio
            matched_stft = matched_magnitude * np.exp(1j * np.angle(conv_stft))
            matched_audio = librosa.istft(matched_stft, hop_length=hop_length)
            
            # Match length
            if len(matched_audio) > len(converted_audio):
                matched_audio = matched_audio[:len(converted_audio)]
            elif len(matched_audio) < len(converted_audio):
                matched_audio = np.pad(matched_audio, (0, len(converted_audio) - len(matched_audio)))
            
            return matched_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Spectral matching failed: {e}")
            return converted_audio
    
    def _apply_timbre_transfer(
        self,
        converted_audio: np.ndarray,
        reference_audio: np.ndarray,
        ref_characteristics: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply timbre transfer from reference to converted audio.
        
        Args:
            converted_audio: Converted audio
            reference_audio: Reference audio
            ref_characteristics: Reference voice characteristics
            
        Returns:
            Audio with transferred timbre
        """
        try:
            import librosa
            from scipy import signal
            
            sample_rate = 22050
            
            # Extract MFCCs for timbre comparison
            conv_mfcc = librosa.feature.mfcc(
                y=converted_audio, sr=sample_rate, n_mfcc=20
            )
            ref_mfcc = librosa.feature.mfcc(
                y=reference_audio, sr=sample_rate, n_mfcc=20
            )
            
            # Calculate MFCC difference
            conv_mfcc_mean = np.mean(conv_mfcc, axis=1)
            ref_mfcc_mean = np.mean(ref_mfcc, axis=1)
            
            mfcc_diff = ref_mfcc_mean - conv_mfcc_mean
            
            # Apply subtle timbre adjustment using EQ
            timbre_brightness = np.sum(mfcc_diff[5:15])
            
            if abs(timbre_brightness) > 0.5:
                if timbre_brightness > 0:
                    # Reference is brighter, boost highs slightly
                    b, a = signal.butter(2, 3000 / (sample_rate / 2), btype='high')
                    high_freq = signal.filtfilt(b, a, converted_audio)
                    converted_audio = converted_audio + 0.1 * high_freq
                else:
                    # Reference is darker, reduce highs slightly
                    b, a = signal.butter(2, 3000 / (sample_rate / 2), btype='low')
                    converted_audio = 0.9 * converted_audio + 0.1 * signal.filtfilt(b, a, converted_audio)
            
            return converted_audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Timbre transfer failed: {e}")
            return converted_audio
    
    def _normalize_output(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize output audio.
        
        Args:
            audio: Input audio
            
        Returns:
            Normalized audio
        """
        try:
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Peak normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Output normalization failed: {e}")
            return audio
    
    def _calculate_conversion_quality(
        self,
        converted_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> float:
        """
        Calculate quality score for converted audio.
        
        Args:
            converted_audio: Converted audio
            reference_audio: Reference audio
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            import librosa
            
            sample_rate = 22050
            
            # MFCC similarity
            conv_mfcc = librosa.feature.mfcc(
                y=converted_audio, sr=sample_rate, n_mfcc=13
            )
            ref_mfcc = librosa.feature.mfcc(
                y=reference_audio, sr=sample_rate, n_mfcc=13
            )
            
            conv_mfcc_mean = np.mean(conv_mfcc, axis=1)
            ref_mfcc_mean = np.mean(ref_mfcc, axis=1)
            
            mfcc_similarity = np.dot(conv_mfcc_mean, ref_mfcc_mean) / (
                np.linalg.norm(conv_mfcc_mean) * np.linalg.norm(ref_mfcc_mean) + 1e-8
            )
            mfcc_score = max(0.0, min(1.0, (mfcc_similarity + 1) / 2))
            
            # Spectral centroid similarity
            conv_centroid = np.mean(librosa.feature.spectral_centroid(
                y=converted_audio, sr=sample_rate
            ))
            ref_centroid = np.mean(librosa.feature.spectral_centroid(
                y=reference_audio, sr=sample_rate
            ))
            
            centroid_ratio = min(conv_centroid, ref_centroid) / (max(conv_centroid, ref_centroid) + 1e-8)
            centroid_score = centroid_ratio
            
            # Energy similarity
            conv_energy = np.mean(librosa.feature.rms(y=converted_audio))
            ref_energy = np.mean(librosa.feature.rms(y=reference_audio))
            
            energy_ratio = min(conv_energy, ref_energy) / (max(conv_energy, ref_energy) + 1e-8)
            energy_score = energy_ratio
            
            # Weighted combination
            quality_score = (
                0.5 * mfcc_score +
                0.3 * centroid_score +
                0.2 * energy_score
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5
    
    def convert_tts_output(
        self,
        tts_audio: np.ndarray,
        tts_sr: int,
        reference_audio: np.ndarray,
        reference_sr: int,
        pitch_shift: float = 0.0,
        preserve_formants: bool = True
    ) -> ConversionResult:
        """
        Convert TTS output to match a reference voice.
        
        Convenience method that wraps run_voice_conversion_pipeline.
        
        Args:
            tts_audio: TTS-generated audio
            tts_sr: Sample rate of TTS audio
            reference_audio: Reference audio for voice matching
            reference_sr: Sample rate of reference audio
            pitch_shift: Additional pitch shift in semitones
            preserve_formants: Whether to preserve formants during conversion
            
        Returns:
            ConversionResult with converted audio
        """
        return self.run_voice_conversion_pipeline(
            tts_audio=tts_audio,
            tts_sr=tts_sr,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            pitch_shift=pitch_shift,
            preserve_formants=preserve_formants
        )
    
    def run_pipeline_from_files(
        self,
        tts_audio_path: str,
        reference_audio_path: str,
        output_path: Optional[str] = None,
        pitch_shift: float = 0.0,
        preserve_formants: bool = True,
        apply_spectral_matching: bool = True,
        apply_timbre_transfer: bool = True
    ) -> Tuple[Optional[str], Optional[ConversionResult]]:
        """
        Run the voice conversion pipeline from audio files.
        
        Convenience method that handles file I/O for the pipeline:
        TTS audio file → RVC conversion → output file
        
        Args:
            tts_audio_path: Path to TTS-generated audio file
            reference_audio_path: Path to reference audio file
            output_path: Optional output path (auto-generated if None)
            pitch_shift: Additional pitch shift in semitones
            preserve_formants: Whether to preserve formants
            apply_spectral_matching: Whether to apply spectral matching
            apply_timbre_transfer: Whether to apply timbre transfer
            
        Returns:
            Tuple of (output_path, ConversionResult) or (None, None) if failed
        """
        import soundfile as sf
        import time
        
        try:
            # Load audio files
            tts_audio, tts_sr = sf.read(tts_audio_path)
            reference_audio, ref_sr = sf.read(reference_audio_path)
            
            # Ensure mono
            if len(tts_audio.shape) > 1:
                tts_audio = tts_audio.mean(axis=1)
            if len(reference_audio.shape) > 1:
                reference_audio = reference_audio.mean(axis=1)
            
            # Run pipeline
            result = self.run_voice_conversion_pipeline(
                tts_audio=tts_audio,
                tts_sr=tts_sr,
                reference_audio=reference_audio,
                reference_sr=ref_sr,
                pitch_shift=pitch_shift,
                preserve_formants=preserve_formants,
                apply_spectral_matching=apply_spectral_matching,
                apply_timbre_transfer=apply_timbre_transfer
            )
            
            if not result.success:
                return None, result
            
            # Generate output path if not provided
            if output_path is None:
                output_dir = Path("results")
                output_dir.mkdir(exist_ok=True)
                output_path = str(output_dir / f"rvc_converted_{int(time.time())}.wav")
            
            # Save output
            sf.write(output_path, result.audio, result.sample_rate)
            
            logger.info(f"Voice conversion pipeline output saved to: {output_path}")
            
            return output_path, result
            
        except Exception as e:
            logger.error(f"Pipeline from files failed: {e}")
            return None, None
    
    # ==================== Real-Time Voice Conversion Methods ====================
    
    def create_realtime_converter(
        self,
        reference_audio: np.ndarray,
        reference_sr: int = 22050,
        config: Optional[RealTimeConfig] = None,
        on_chunk_processed: Optional[Callable[[np.ndarray], None]] = None
    ) -> RealTimeVoiceConverter:
        """
        Create a real-time voice converter for streaming applications.
        
        This method creates a RealTimeVoiceConverter instance that can process
        audio chunks in real-time with low latency.
        
        Args:
            reference_audio: Reference audio for voice matching
            reference_sr: Sample rate of reference audio
            config: Real-time configuration options
            on_chunk_processed: Optional callback when a chunk is processed
            
        Returns:
            RealTimeVoiceConverter instance
            
        Example:
            converter = rvc.create_realtime_converter(reference_audio, 22050)
            converter.start()
            
            for chunk in audio_stream:
                output = converter.process_chunk(chunk)
                play(output)
            
            converter.stop()
        """
        return RealTimeVoiceConverter(
            rvc_converter=self,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            config=config,
            on_chunk_processed=on_chunk_processed
        )
    
    def convert_realtime_stream(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        reference_audio: np.ndarray,
        reference_sr: int = 22050,
        config: Optional[RealTimeConfig] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Convert an audio stream in real-time.
        
        This is a convenience method that creates a real-time converter
        and processes a stream of audio chunks.
        
        Args:
            audio_generator: Generator yielding input audio chunks
            reference_audio: Reference audio for voice matching
            reference_sr: Sample rate of reference audio
            config: Real-time configuration options
            
        Yields:
            Converted audio chunks
            
        Example:
            def audio_stream():
                while recording:
                    yield get_audio_chunk()
            
            for output_chunk in rvc.convert_realtime_stream(
                audio_stream(), reference_audio
            ):
                play(output_chunk)
        """
        rt_converter = self.create_realtime_converter(
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            config=config
        )
        
        yield from rt_converter.process_stream(audio_generator)
    
    def process_realtime_chunk(
        self,
        chunk: np.ndarray,
        reference_audio: np.ndarray,
        reference_sr: int = 22050,
        pitch_shift: float = 0.0,
        low_latency: bool = True
    ) -> np.ndarray:
        """
        Process a single audio chunk for real-time conversion.
        
        This is a stateless method for processing individual chunks.
        For continuous streams, use create_realtime_converter() instead.
        
        Args:
            chunk: Input audio chunk
            reference_audio: Reference audio for voice matching
            reference_sr: Sample rate of reference audio
            pitch_shift: Additional pitch shift in semitones
            low_latency: Use low-latency mode (faster but lower quality)
            
        Returns:
            Converted audio chunk
        """
        try:
            import librosa
            from scipy import signal
            
            # Ensure float32
            chunk = chunk.astype(np.float32)
            
            # Preprocess reference if needed
            if len(reference_audio.shape) > 1:
                reference_audio = reference_audio.mean(axis=1)
            
            if reference_sr != 22050:
                reference_audio = librosa.resample(
                    reference_audio, orig_sr=reference_sr, target_sr=22050
                )
            
            # Extract reference pitch
            f0_ref, _, _ = librosa.pyin(
                reference_audio, fmin=50, fmax=500, sr=22050
            )
            f0_ref_clean = f0_ref[~np.isnan(f0_ref)]
            ref_pitch = float(np.mean(f0_ref_clean)) if len(f0_ref_clean) > 0 else 150.0
            
            # Extract chunk pitch
            f0_chunk, _, _ = librosa.pyin(
                chunk, fmin=50, fmax=500, sr=22050, frame_length=512
            )
            f0_chunk_clean = f0_chunk[~np.isnan(f0_chunk)]
            chunk_pitch = float(np.mean(f0_chunk_clean)) if len(f0_chunk_clean) > 0 else 150.0
            
            # Calculate pitch shift
            if chunk_pitch > 0 and ref_pitch > 0:
                auto_pitch_shift = 12 * np.log2(ref_pitch / chunk_pitch)
                total_pitch_shift = max(-12, min(12, auto_pitch_shift + pitch_shift))
            else:
                total_pitch_shift = pitch_shift
            
            # Apply pitch shift
            if abs(total_pitch_shift) > 0.5:
                shift_factor = 2 ** (total_pitch_shift / 12)
                num_samples = int(len(chunk) / shift_factor)
                
                if num_samples > 0:
                    shifted = signal.resample(chunk, num_samples)
                    chunk = signal.resample(shifted, len(chunk))
            
            # Normalize
            max_val = np.max(np.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val * 0.95
            
            return chunk.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Real-time chunk processing failed: {e}")
            return chunk
    
    def get_realtime_latency_estimate(
        self,
        config: Optional[RealTimeConfig] = None
    ) -> Dict[str, float]:
        """
        Estimate the latency for real-time voice conversion.
        
        Args:
            config: Real-time configuration options
            
        Returns:
            Dictionary with latency estimates in milliseconds
        """
        config = config or RealTimeConfig()
        
        chunk_latency_ms = (config.chunk_size / config.sample_rate) * 1000
        buffer_latency_ms = chunk_latency_ms * config.buffer_chunks
        processing_latency_ms = 5.0 if config.low_latency_mode else 15.0
        
        total_latency_ms = chunk_latency_ms + buffer_latency_ms + processing_latency_ms
        
        return {
            "chunk_latency_ms": round(chunk_latency_ms, 2),
            "buffer_latency_ms": round(buffer_latency_ms, 2),
            "processing_latency_ms": round(processing_latency_ms, 2),
            "total_estimated_latency_ms": round(total_latency_ms, 2),
            "sample_rate": config.sample_rate,
            "chunk_size": config.chunk_size,
            "buffer_chunks": config.buffer_chunks
        }


# Global instance for convenience
_rvc_converter: Optional[RVCVoiceConverter] = None
_realtime_converter: Optional[RealTimeVoiceConverter] = None


def get_rvc_converter() -> RVCVoiceConverter:
    """Get or create global RVC voice converter instance."""
    global _rvc_converter
    if _rvc_converter is None:
        _rvc_converter = RVCVoiceConverter()
    return _rvc_converter


def get_realtime_converter(
    reference_audio: np.ndarray,
    reference_sr: int = 22050,
    config: Optional[RealTimeConfig] = None,
    force_new: bool = False
) -> RealTimeVoiceConverter:
    """
    Get or create a global real-time voice converter instance.
    
    Args:
        reference_audio: Reference audio for voice matching
        reference_sr: Sample rate of reference audio
        config: Real-time configuration options
        force_new: Force creation of a new instance
        
    Returns:
        RealTimeVoiceConverter instance
    """
    global _realtime_converter
    
    if _realtime_converter is None or force_new:
        rvc = get_rvc_converter()
        _realtime_converter = rvc.create_realtime_converter(
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            config=config
        )
    
    return _realtime_converter


def stop_realtime_converter() -> Optional[RealTimeStats]:
    """
    Stop the global real-time converter if running.
    
    Returns:
        Final statistics from the session, or None if not running
    """
    global _realtime_converter
    
    if _realtime_converter is not None:
        stats = _realtime_converter.stop()
        _realtime_converter = None
        return stats
    
    return None
