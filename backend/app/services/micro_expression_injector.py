"""
Micro-Expression Injector for Perfect Voice Cloning.

This module implements advanced micro-expression analysis and injection to make
synthesized speech sound more natural and human-like. It includes:
1. Breathing pattern analysis from reference audio
2. Natural breathing injection at appropriate intervals
3. Hesitation and filler sound injection
4. Lip smack and mouth sound injection
5. Coarticulation smoothing

Goal: Add subtle human-like details that make speech indistinguishable from real human speech.
"""

import logging
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, butter, filtfilt
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class BreathType(Enum):
    """Types of breathing patterns detected in speech."""
    INHALATION = "inhalation"
    EXHALATION = "exhalation"
    PAUSE_BREATH = "pause_breath"
    PHRASE_BREATH = "phrase_breath"
    DEEP_BREATH = "deep_breath"


@dataclass
class BreathEvent:
    """Represents a single breathing event in audio."""
    breath_type: BreathType
    start_time: float
    end_time: float
    duration: float
    intensity: float
    frequency_profile: np.ndarray
    position_in_phrase: str  # 'start', 'middle', 'end'


@dataclass
class BreathingPattern:
    """Complete breathing pattern analysis from reference audio."""
    breath_events: List[BreathEvent]
    average_breath_interval: float
    average_breath_duration: float
    breath_intensity_mean: float
    breath_intensity_std: float
    breath_frequency_range: Tuple[float, float]
    breathing_rate: float  # breaths per minute
    phrase_breath_ratio: float  # ratio of breaths at phrase boundaries
    breath_spectral_profile: np.ndarray
    breath_envelope_template: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class BreathingAnalysisResult:
    """Result of breathing pattern analysis."""
    pattern: BreathingPattern
    quality_score: float
    confidence: float
    analysis_metadata: Dict[str, Any]


class BreathingPatternAnalyzer:
    """
    Analyzes breathing patterns from reference audio for natural speech synthesis.
    
    Breathing is a crucial micro-expression that makes speech sound natural.
    This analyzer extracts:
    - Breath timing and intervals
    - Breath intensity and duration
    - Spectral characteristics of breaths
    - Relationship between breathing and phrase structure
    """
    
    # Breathing frequency characteristics
    BREATH_FREQ_MIN = 100  # Hz - lower bound for breath noise
    BREATH_FREQ_MAX = 4000  # Hz - upper bound for breath noise
    BREATH_NOISE_BAND = (200, 2000)  # Primary breath noise band
    
    # Timing parameters
    MIN_BREATH_DURATION = 0.1  # seconds
    MAX_BREATH_DURATION = 1.5  # seconds
    MIN_BREATH_INTERVAL = 0.5  # seconds between breaths
    
    # Detection thresholds
    BREATH_ENERGY_THRESHOLD = 0.15  # relative to speech energy
    BREATH_SPECTRAL_FLATNESS_MIN = 0.3  # breaths have flatter spectrum
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize breathing pattern analyzer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        self.frame_length = 2048
        
        logger.info("Breathing Pattern Analyzer initialized")
    
    def analyze_breathing_patterns(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> BreathingAnalysisResult:
        """
        Analyze breathing patterns from reference audio.
        
        This method extracts comprehensive breathing characteristics including:
        - Individual breath events with timing and intensity
        - Average breathing rate and intervals
        - Spectral profile of breaths
        - Relationship to phrase boundaries
        
        Args:
            audio: Reference audio array
            sample_rate: Sample rate (uses default if not provided)
            
        Returns:
            BreathingAnalysisResult with complete pattern analysis
            
        Validates: Requirements 5.1 (natural breathing patterns)
        """
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        logger.info(f"Analyzing breathing patterns in {len(audio)/sr:.2f}s audio")
        
        # Step 1: Detect potential breath regions
        breath_candidates = self._detect_breath_candidates(audio, sr)
        
        # Step 2: Classify and validate breath events
        breath_events = self._classify_breath_events(audio, sr, breath_candidates)
        
        # Step 3: Extract breath spectral profile
        breath_spectral_profile = self._extract_breath_spectral_profile(
            audio, sr, breath_events
        )
        
        # Step 4: Create breath envelope template
        breath_envelope_template = self._create_breath_envelope_template(
            audio, sr, breath_events
        )
        
        # Step 5: Analyze phrase-breath relationships
        phrase_breath_ratio = self._analyze_phrase_breath_relationship(
            audio, sr, breath_events
        )
        
        # Step 6: Calculate breathing statistics
        stats = self._calculate_breathing_statistics(breath_events)
        
        # Create breathing pattern
        pattern = BreathingPattern(
            breath_events=breath_events,
            average_breath_interval=stats['average_interval'],
            average_breath_duration=stats['average_duration'],
            breath_intensity_mean=stats['intensity_mean'],
            breath_intensity_std=stats['intensity_std'],
            breath_frequency_range=(self.BREATH_FREQ_MIN, self.BREATH_FREQ_MAX),
            breathing_rate=stats['breathing_rate'],
            phrase_breath_ratio=phrase_breath_ratio,
            breath_spectral_profile=breath_spectral_profile,
            breath_envelope_template=breath_envelope_template,
            metadata={
                'audio_duration': len(audio) / sr,
                'num_breaths_detected': len(breath_events),
                'analysis_sample_rate': sr
            }
        )
        
        # Calculate quality and confidence scores
        quality_score = self._assess_pattern_quality(pattern)
        confidence = self._calculate_confidence(breath_events, len(audio) / sr)
        
        return BreathingAnalysisResult(
            pattern=pattern,
            quality_score=quality_score,
            confidence=confidence,
            analysis_metadata={
                'candidates_found': len(breath_candidates),
                'events_validated': len(breath_events),
                'detection_threshold': self.BREATH_ENERGY_THRESHOLD
            }
        )
    
    def _detect_breath_candidates(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Tuple[int, int]]:
        """
        Detect candidate breath regions using spectral and energy analysis.
        
        Breaths are characterized by:
        - Low energy compared to speech
        - Relatively flat spectrum (noise-like)
        - Specific frequency content (breath noise band)
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples for breath candidates
        """
        candidates = []
        
        # Compute features for breath detection
        # 1. RMS energy
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        )[0]
        
        # 2. Spectral flatness (breaths are more noise-like)
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # 3. Spectral centroid (breaths have lower centroid)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]
        
        # 4. Zero crossing rate (breaths have moderate ZCR)
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # 5. Breath band energy ratio
        breath_band_ratio = self._compute_breath_band_ratio(audio, sr)
        
        # Normalize features
        rms_norm = rms / (np.max(rms) + 1e-8)
        spectral_flatness_norm = spectral_flatness
        centroid_norm = spectral_centroid / (sr / 2)
        zcr_norm = zcr / (np.max(zcr) + 1e-8)
        
        # Breath detection criteria:
        # - Low to moderate energy (not silence, not loud speech)
        # - High spectral flatness (noise-like)
        # - Moderate spectral centroid
        # - High breath band energy ratio
        
        # Compute breath likelihood score
        breath_score = (
            (0.3 - np.abs(rms_norm - 0.15)) * 2 +  # Prefer moderate energy
            spectral_flatness_norm * 1.5 +  # Higher flatness = more breath-like
            (1 - centroid_norm) * 0.5 +  # Lower centroid
            breath_band_ratio * 1.0  # Breath band energy
        )
        
        # Smooth the score
        breath_score_smooth = gaussian_filter1d(breath_score, sigma=3)
        
        # Find peaks in breath score
        threshold = np.percentile(breath_score_smooth, 70)
        above_threshold = breath_score_smooth > threshold
        
        # Find contiguous regions
        in_breath = False
        start_frame = 0
        
        for i, is_breath in enumerate(above_threshold):
            if is_breath and not in_breath:
                start_frame = i
                in_breath = True
            elif not is_breath and in_breath:
                end_frame = i
                
                # Convert to samples
                start_sample = start_frame * self.hop_length
                end_sample = end_frame * self.hop_length
                
                # Check duration constraints
                duration = (end_sample - start_sample) / sr
                if self.MIN_BREATH_DURATION <= duration <= self.MAX_BREATH_DURATION:
                    candidates.append((start_sample, end_sample))
                
                in_breath = False
        
        # Handle case where audio ends during a breath
        if in_breath:
            end_sample = len(above_threshold) * self.hop_length
            duration = (end_sample - start_frame * self.hop_length) / sr
            if self.MIN_BREATH_DURATION <= duration <= self.MAX_BREATH_DURATION:
                candidates.append((start_frame * self.hop_length, end_sample))
        
        logger.debug(f"Found {len(candidates)} breath candidates")
        return candidates
    
    def _compute_breath_band_ratio(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Compute the ratio of energy in the breath frequency band.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Array of breath band energy ratios per frame
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        # Define breath band
        breath_band_mask = (freqs >= self.BREATH_NOISE_BAND[0]) & (freqs <= self.BREATH_NOISE_BAND[1])
        
        # Compute energy in breath band vs total
        breath_band_energy = np.sum(magnitude[breath_band_mask, :] ** 2, axis=0)
        total_energy = np.sum(magnitude ** 2, axis=0) + 1e-8
        
        return breath_band_energy / total_energy
    
    def _classify_breath_events(
        self,
        audio: np.ndarray,
        sr: int,
        candidates: List[Tuple[int, int]]
    ) -> List[BreathEvent]:
        """
        Classify and validate breath candidates into breath events.
        
        Args:
            audio: Audio array
            sr: Sample rate
            candidates: List of candidate regions
            
        Returns:
            List of validated BreathEvent objects
        """
        events = []
        
        for start_sample, end_sample in candidates:
            # Extract breath segment
            breath_segment = audio[start_sample:min(end_sample, len(audio))]
            
            if len(breath_segment) < int(self.MIN_BREATH_DURATION * sr):
                continue
            
            # Validate this is actually a breath
            if not self._validate_breath(breath_segment, sr):
                continue
            
            # Classify breath type
            breath_type = self._classify_breath_type(breath_segment, sr)
            
            # Calculate intensity
            intensity = self._calculate_breath_intensity(breath_segment, audio)
            
            # Extract frequency profile
            freq_profile = self._extract_breath_frequency_profile(breath_segment, sr)
            
            # Determine position in phrase
            position = self._determine_phrase_position(audio, sr, start_sample, end_sample)
            
            event = BreathEvent(
                breath_type=breath_type,
                start_time=start_sample / sr,
                end_time=end_sample / sr,
                duration=(end_sample - start_sample) / sr,
                intensity=intensity,
                frequency_profile=freq_profile,
                position_in_phrase=position
            )
            
            events.append(event)
        
        logger.debug(f"Validated {len(events)} breath events from {len(candidates)} candidates")
        return events
    
    def _validate_breath(self, segment: np.ndarray, sr: int) -> bool:
        """
        Validate that a segment is actually a breath.
        
        Args:
            segment: Audio segment
            sr: Sample rate
            
        Returns:
            True if segment is a valid breath
        """
        # Check spectral flatness
        flatness = librosa.feature.spectral_flatness(y=segment)[0]
        avg_flatness = np.mean(flatness)
        
        if avg_flatness < self.BREATH_SPECTRAL_FLATNESS_MIN:
            return False
        
        # Check that it's not silence
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 0.001:
            return False
        
        # Check that it's not too loud (would be speech)
        if rms > 0.3:
            return False
        
        # Check for pitch (breaths should have minimal pitch)
        f0, voiced_flag, _ = librosa.pyin(
            segment, 
            fmin=80, 
            fmax=400, 
            sr=sr
        )
        voiced_ratio = np.mean(voiced_flag) if voiced_flag is not None else 0
        
        # Breaths should have very little voicing
        if voiced_ratio > 0.3:
            return False
        
        return True
    
    def _classify_breath_type(self, segment: np.ndarray, sr: int) -> BreathType:
        """
        Classify the type of breath based on spectral and temporal characteristics.
        
        Args:
            segment: Breath audio segment
            sr: Sample rate
            
        Returns:
            BreathType classification
        """
        duration = len(segment) / sr
        
        # Compute envelope
        envelope = np.abs(segment)
        envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.01 * sr))
        
        # Find envelope peak position
        peak_idx = np.argmax(envelope_smooth)
        peak_position = peak_idx / len(envelope_smooth)
        
        # Classify based on envelope shape and duration
        if duration > 0.8:
            return BreathType.DEEP_BREATH
        elif peak_position < 0.4:
            # Peak early = inhalation (quick intake)
            return BreathType.INHALATION
        elif peak_position > 0.6:
            # Peak late = exhalation (gradual release)
            return BreathType.EXHALATION
        elif duration < 0.3:
            return BreathType.PAUSE_BREATH
        else:
            return BreathType.PHRASE_BREATH
    
    def _calculate_breath_intensity(
        self,
        breath_segment: np.ndarray,
        full_audio: np.ndarray
    ) -> float:
        """
        Calculate breath intensity relative to speech.
        
        Args:
            breath_segment: Breath audio segment
            full_audio: Full audio for reference
            
        Returns:
            Intensity value (0-1)
        """
        breath_rms = np.sqrt(np.mean(breath_segment ** 2))
        audio_rms = np.sqrt(np.mean(full_audio ** 2))
        
        if audio_rms > 0:
            relative_intensity = breath_rms / audio_rms
            return float(np.clip(relative_intensity, 0, 1))
        return 0.0
    
    def _extract_breath_frequency_profile(
        self,
        segment: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Extract the frequency profile of a breath.
        
        Args:
            segment: Breath audio segment
            sr: Sample rate
            
        Returns:
            Frequency profile array
        """
        # Compute magnitude spectrum
        stft = librosa.stft(segment, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        
        # Average across time
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Normalize
        avg_spectrum = avg_spectrum / (np.max(avg_spectrum) + 1e-8)
        
        return avg_spectrum
    
    def _determine_phrase_position(
        self,
        audio: np.ndarray,
        sr: int,
        start_sample: int,
        end_sample: int
    ) -> str:
        """
        Determine where the breath occurs relative to phrase boundaries.
        
        Args:
            audio: Full audio
            sr: Sample rate
            start_sample: Breath start
            end_sample: Breath end
            
        Returns:
            Position string: 'start', 'middle', or 'end'
        """
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Find speech regions (high energy)
        threshold = np.percentile(rms, 50)
        speech_mask = rms > threshold
        
        # Convert breath position to frames
        breath_frame = start_sample // self.hop_length
        
        # Look at surrounding context
        context_frames = int(0.5 * sr / self.hop_length)  # 0.5 second context
        
        before_start = max(0, breath_frame - context_frames)
        after_end = min(len(speech_mask), breath_frame + context_frames)
        
        speech_before = np.mean(speech_mask[before_start:breath_frame]) if breath_frame > before_start else 0
        speech_after = np.mean(speech_mask[breath_frame:after_end]) if after_end > breath_frame else 0
        
        if speech_before < 0.3 and speech_after > 0.5:
            return 'start'
        elif speech_before > 0.5 and speech_after < 0.3:
            return 'end'
        else:
            return 'middle'
    
    def _extract_breath_spectral_profile(
        self,
        audio: np.ndarray,
        sr: int,
        breath_events: List[BreathEvent]
    ) -> np.ndarray:
        """
        Extract average spectral profile from all detected breaths.
        
        Args:
            audio: Full audio
            sr: Sample rate
            breath_events: List of breath events
            
        Returns:
            Average breath spectral profile
        """
        if not breath_events:
            # Return default breath-like spectrum
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            # Create a noise-like spectrum with breath band emphasis
            profile = np.exp(-((freqs - 800) ** 2) / (2 * 500 ** 2))
            return profile / (np.max(profile) + 1e-8)
        
        profiles = []
        for event in breath_events:
            start_sample = int(event.start_time * sr)
            end_sample = int(event.end_time * sr)
            
            if end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
                profile = self._extract_breath_frequency_profile(segment, sr)
                profiles.append(profile)
        
        if profiles:
            avg_profile = np.mean(profiles, axis=0)
            return avg_profile / (np.max(avg_profile) + 1e-8)
        
        return np.ones(self.n_fft // 2 + 1)
    
    def _create_breath_envelope_template(
        self,
        audio: np.ndarray,
        sr: int,
        breath_events: List[BreathEvent]
    ) -> np.ndarray:
        """
        Create a template envelope for breath injection.
        
        Args:
            audio: Full audio
            sr: Sample rate
            breath_events: List of breath events
            
        Returns:
            Normalized envelope template
        """
        if not breath_events:
            # Create default breath envelope (attack-sustain-release)
            template_length = int(0.3 * sr)  # 300ms default
            attack = int(0.1 * template_length)
            release = int(0.3 * template_length)
            
            envelope = np.ones(template_length)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            
            return envelope
        
        # Average envelope from detected breaths
        envelopes = []
        target_length = int(np.mean([e.duration for e in breath_events]) * sr)
        target_length = max(int(0.1 * sr), min(target_length, int(1.0 * sr)))
        
        for event in breath_events:
            start_sample = int(event.start_time * sr)
            end_sample = int(event.end_time * sr)
            
            if end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
                envelope = np.abs(segment)
                envelope = gaussian_filter1d(envelope, sigma=int(0.01 * sr))
                
                # Resample to target length
                if len(envelope) != target_length:
                    envelope = np.interp(
                        np.linspace(0, 1, target_length),
                        np.linspace(0, 1, len(envelope)),
                        envelope
                    )
                
                envelopes.append(envelope)
        
        if envelopes:
            avg_envelope = np.mean(envelopes, axis=0)
            return avg_envelope / (np.max(avg_envelope) + 1e-8)
        
        return np.ones(target_length)
    
    def _analyze_phrase_breath_relationship(
        self,
        audio: np.ndarray,
        sr: int,
        breath_events: List[BreathEvent]
    ) -> float:
        """
        Analyze the relationship between breaths and phrase boundaries.
        
        Args:
            audio: Full audio
            sr: Sample rate
            breath_events: List of breath events
            
        Returns:
            Ratio of breaths at phrase boundaries (0-1)
        """
        if not breath_events:
            return 0.5  # Default assumption
        
        phrase_breaths = sum(
            1 for e in breath_events 
            if e.position_in_phrase in ['start', 'end']
        )
        
        return phrase_breaths / len(breath_events)
    
    def _calculate_breathing_statistics(
        self,
        breath_events: List[BreathEvent]
    ) -> Dict[str, float]:
        """
        Calculate statistical measures of breathing patterns.
        
        Args:
            breath_events: List of breath events
            
        Returns:
            Dictionary of breathing statistics
        """
        if not breath_events:
            return {
                'average_interval': 3.0,  # Default 3 seconds
                'average_duration': 0.3,
                'intensity_mean': 0.1,
                'intensity_std': 0.05,
                'breathing_rate': 20.0  # breaths per minute
            }
        
        # Calculate intervals between breaths
        if len(breath_events) > 1:
            intervals = [
                breath_events[i+1].start_time - breath_events[i].end_time
                for i in range(len(breath_events) - 1)
            ]
            avg_interval = np.mean(intervals) if intervals else 3.0
        else:
            avg_interval = 3.0
        
        # Calculate duration statistics
        durations = [e.duration for e in breath_events]
        avg_duration = np.mean(durations)
        
        # Calculate intensity statistics
        intensities = [e.intensity for e in breath_events]
        intensity_mean = np.mean(intensities)
        intensity_std = np.std(intensities)
        
        # Calculate breathing rate
        if len(breath_events) > 1:
            total_time = breath_events[-1].end_time - breath_events[0].start_time
            breathing_rate = (len(breath_events) / total_time) * 60 if total_time > 0 else 20.0
        else:
            breathing_rate = 20.0
        
        return {
            'average_interval': float(avg_interval),
            'average_duration': float(avg_duration),
            'intensity_mean': float(intensity_mean),
            'intensity_std': float(intensity_std),
            'breathing_rate': float(breathing_rate)
        }
    
    def _assess_pattern_quality(self, pattern: BreathingPattern) -> float:
        """
        Assess the quality of the extracted breathing pattern.
        
        Args:
            pattern: Extracted breathing pattern
            
        Returns:
            Quality score (0-1)
        """
        score = 0.0
        
        # Check if we have enough breath events
        num_events = len(pattern.breath_events)
        if num_events >= 3:
            score += 0.3
        elif num_events >= 1:
            score += 0.15
        
        # Check breathing rate is reasonable (12-20 breaths/min is normal)
        if 10 <= pattern.breathing_rate <= 25:
            score += 0.2
        elif 5 <= pattern.breathing_rate <= 35:
            score += 0.1
        
        # Check average duration is reasonable
        if 0.15 <= pattern.average_breath_duration <= 0.8:
            score += 0.2
        elif 0.1 <= pattern.average_breath_duration <= 1.2:
            score += 0.1
        
        # Check intensity consistency
        if pattern.breath_intensity_std < pattern.breath_intensity_mean * 0.5:
            score += 0.15
        
        # Check phrase-breath relationship
        if 0.3 <= pattern.phrase_breath_ratio <= 0.8:
            score += 0.15
        
        return float(np.clip(score, 0, 1))
    
    def _calculate_confidence(
        self,
        breath_events: List[BreathEvent],
        audio_duration: float
    ) -> float:
        """
        Calculate confidence in the breathing pattern analysis.
        
        Args:
            breath_events: Detected breath events
            audio_duration: Duration of analyzed audio
            
        Returns:
            Confidence score (0-1)
        """
        if audio_duration < 3.0:
            # Very short audio - low confidence
            return 0.3
        
        # Expected number of breaths based on duration
        expected_breaths = audio_duration / 3.0  # Assume ~1 breath per 3 seconds
        actual_breaths = len(breath_events)
        
        # Confidence based on detection rate
        if expected_breaths > 0:
            detection_ratio = actual_breaths / expected_breaths
            # Optimal is around 0.5-1.5 of expected
            if 0.3 <= detection_ratio <= 2.0:
                confidence = 0.8
            elif 0.1 <= detection_ratio <= 3.0:
                confidence = 0.5
            else:
                confidence = 0.3
        else:
            confidence = 0.5
        
        # Boost confidence for longer audio
        if audio_duration > 10:
            confidence = min(1.0, confidence + 0.1)
        
        return float(confidence)


class NaturalBreathingInjector:
    """
    Injects natural breathing sounds at appropriate intervals in synthesized speech.
    
    This class uses analyzed breathing patterns from reference audio to inject
    realistic breath sounds that match the speaker's natural breathing style.
    
    Key features:
    - Detects appropriate breath insertion points (phrase boundaries, pauses)
    - Generates breath sounds matching reference spectral profile
    - Applies proper envelope and intensity matching
    - Ensures natural timing and spacing of breaths
    
    Validates: Requirements 5.1 (natural breathing patterns at appropriate intervals)
    """
    
    # Timing parameters for breath insertion
    MIN_PHRASE_GAP_FOR_BREATH = 0.3  # Minimum pause duration to insert breath (seconds)
    MAX_PHRASE_GAP_FOR_BREATH = 2.0  # Maximum pause where breath makes sense
    MIN_INTERVAL_BETWEEN_BREATHS = 2.0  # Minimum time between injected breaths
    IDEAL_BREATH_INTERVAL = 4.0  # Ideal interval between breaths (seconds)
    
    # Energy thresholds
    SILENCE_THRESHOLD = 0.02  # RMS threshold for silence detection
    SPEECH_THRESHOLD = 0.1  # RMS threshold for speech detection
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the natural breathing injector.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        self.analyzer = BreathingPatternAnalyzer(sample_rate)
        
        logger.info("Natural Breathing Injector initialized")
    
    def inject_breathing(
        self,
        synthesized_audio: np.ndarray,
        breathing_pattern: BreathingPattern,
        sample_rate: Optional[int] = None,
        intensity_scale: float = 1.0,
        min_breaths: int = 0,
        max_breaths: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Inject natural breathing sounds into synthesized audio.
        
        This method analyzes the synthesized audio to find appropriate insertion
        points (phrase boundaries, pauses) and injects breath sounds that match
        the reference speaker's breathing pattern.
        
        Args:
            synthesized_audio: The synthesized speech audio array
            breathing_pattern: Analyzed breathing pattern from reference audio
            sample_rate: Sample rate (uses default if not provided)
            intensity_scale: Scale factor for breath intensity (0.5-1.5 recommended)
            min_breaths: Minimum number of breaths to inject
            max_breaths: Maximum number of breaths to inject (None for automatic)
            
        Returns:
            Tuple of (audio with breaths injected, list of injection metadata)
            
        Validates: Requirements 5.1 (natural breathing patterns at appropriate intervals)
        """
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            synthesized_audio = librosa.resample(
                synthesized_audio, orig_sr=sr, target_sr=self.sample_rate
            )
            sr = self.sample_rate
        
        audio_duration = len(synthesized_audio) / sr
        logger.info(f"Injecting breathing into {audio_duration:.2f}s synthesized audio")
        
        # Step 1: Find appropriate breath insertion points
        insertion_points = self._find_breath_insertion_points(synthesized_audio, sr)
        
        if not insertion_points:
            logger.warning("No suitable breath insertion points found")
            return synthesized_audio, []
        
        # Step 2: Select which points to use based on timing constraints
        selected_points = self._select_breath_points(
            insertion_points,
            breathing_pattern,
            audio_duration,
            min_breaths,
            max_breaths
        )
        
        if not selected_points:
            logger.warning("No breath points selected after filtering")
            return synthesized_audio, []
        
        # Step 3: Generate breath sounds for each selected point
        breath_sounds = self._generate_breath_sounds(
            selected_points,
            breathing_pattern,
            sr,
            intensity_scale
        )
        
        # Step 4: Inject breaths into audio with crossfading
        result_audio, injection_metadata = self._inject_breaths_into_audio(
            synthesized_audio,
            breath_sounds,
            selected_points,
            sr
        )
        
        logger.info(f"Injected {len(injection_metadata)} breaths into audio")
        
        return result_audio, injection_metadata
    
    def _find_breath_insertion_points(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Dict[str, Any]]:
        """
        Find appropriate points in the audio to insert breaths.
        
        Breaths should be inserted at:
        - Phrase boundaries (pauses between sentences/clauses)
        - Natural pause points in speech
        - Before long utterances
        
        Args:
            audio: Synthesized audio array
            sr: Sample rate
            
        Returns:
            List of insertion point dictionaries with timing and context info
        """
        insertion_points = []
        
        # Compute RMS energy for pause detection
        rms = librosa.feature.rms(
            y=audio,
            frame_length=2048,
            hop_length=self.hop_length
        )[0]
        
        # Smooth RMS for more stable detection
        rms_smooth = gaussian_filter1d(rms, sigma=3)
        
        # Detect silence/pause regions
        silence_mask = rms_smooth < self.SILENCE_THRESHOLD
        speech_mask = rms_smooth > self.SPEECH_THRESHOLD
        
        # Find transitions from speech to silence (potential breath points)
        in_silence = False
        silence_start_frame = 0
        
        for i in range(len(silence_mask)):
            if silence_mask[i] and not in_silence:
                # Entering silence
                silence_start_frame = i
                in_silence = True
            elif not silence_mask[i] and in_silence:
                # Exiting silence
                silence_end_frame = i
                in_silence = False
                
                # Calculate silence duration
                silence_start_time = silence_start_frame * self.hop_length / sr
                silence_end_time = silence_end_frame * self.hop_length / sr
                silence_duration = silence_end_time - silence_start_time
                
                # Check if this is a suitable breath insertion point
                if self.MIN_PHRASE_GAP_FOR_BREATH <= silence_duration <= self.MAX_PHRASE_GAP_FOR_BREATH:
                    # Calculate insertion position (middle of silence)
                    insertion_time = (silence_start_time + silence_end_time) / 2
                    
                    # Determine context (what comes before and after)
                    context = self._analyze_insertion_context(
                        audio, sr, silence_start_frame, silence_end_frame, rms_smooth
                    )
                    
                    insertion_points.append({
                        'time': insertion_time,
                        'silence_start': silence_start_time,
                        'silence_end': silence_end_time,
                        'silence_duration': silence_duration,
                        'context': context,
                        'suitability_score': self._calculate_suitability_score(
                            silence_duration, context
                        )
                    })
        
        # Sort by suitability score
        insertion_points.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        logger.debug(f"Found {len(insertion_points)} potential breath insertion points")
        return insertion_points
    
    def _analyze_insertion_context(
        self,
        audio: np.ndarray,
        sr: int,
        silence_start_frame: int,
        silence_end_frame: int,
        rms: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze the context around a potential breath insertion point.
        
        Args:
            audio: Audio array
            sr: Sample rate
            silence_start_frame: Frame where silence starts
            silence_end_frame: Frame where silence ends
            rms: RMS energy array
            
        Returns:
            Context dictionary with speech characteristics before/after
        """
        context_frames = int(0.5 * sr / self.hop_length)  # 0.5 second context
        
        # Analyze speech before silence
        before_start = max(0, silence_start_frame - context_frames)
        before_rms = rms[before_start:silence_start_frame]
        speech_before_energy = np.mean(before_rms) if len(before_rms) > 0 else 0
        
        # Analyze speech after silence
        after_end = min(len(rms), silence_end_frame + context_frames)
        after_rms = rms[silence_end_frame:after_end]
        speech_after_energy = np.mean(after_rms) if len(after_rms) > 0 else 0
        
        # Determine if this is a phrase boundary
        is_phrase_boundary = (
            speech_before_energy > self.SPEECH_THRESHOLD and
            speech_after_energy > self.SPEECH_THRESHOLD
        )
        
        # Check if this is at the start or end of audio
        is_start = silence_start_frame < context_frames
        is_end = silence_end_frame > len(rms) - context_frames
        
        return {
            'speech_before_energy': float(speech_before_energy),
            'speech_after_energy': float(speech_after_energy),
            'is_phrase_boundary': is_phrase_boundary,
            'is_start': is_start,
            'is_end': is_end,
            'position_type': 'start' if is_start else ('end' if is_end else 'middle')
        }
    
    def _calculate_suitability_score(
        self,
        silence_duration: float,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate how suitable a point is for breath insertion.
        
        Args:
            silence_duration: Duration of the silence
            context: Context information
            
        Returns:
            Suitability score (0-1)
        """
        score = 0.0
        
        # Prefer phrase boundaries
        if context['is_phrase_boundary']:
            score += 0.4
        
        # Prefer middle positions over start/end
        if context['position_type'] == 'middle':
            score += 0.2
        elif context['position_type'] == 'start':
            score += 0.3  # Breaths at start are natural
        
        # Prefer optimal silence duration (0.4-0.8 seconds)
        if 0.4 <= silence_duration <= 0.8:
            score += 0.3
        elif 0.3 <= silence_duration <= 1.0:
            score += 0.2
        else:
            score += 0.1
        
        # Prefer points with balanced speech energy before/after
        energy_ratio = min(
            context['speech_before_energy'],
            context['speech_after_energy']
        ) / (max(
            context['speech_before_energy'],
            context['speech_after_energy']
        ) + 1e-8)
        score += energy_ratio * 0.1
        
        return float(np.clip(score, 0, 1))
    
    def _select_breath_points(
        self,
        insertion_points: List[Dict[str, Any]],
        breathing_pattern: BreathingPattern,
        audio_duration: float,
        min_breaths: int,
        max_breaths: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Select which insertion points to use based on timing constraints.
        
        Args:
            insertion_points: All potential insertion points
            breathing_pattern: Reference breathing pattern
            audio_duration: Total audio duration
            min_breaths: Minimum breaths to inject
            max_breaths: Maximum breaths to inject
            
        Returns:
            Selected insertion points
        """
        if not insertion_points:
            return []
        
        # Calculate target number of breaths based on audio duration and pattern
        target_interval = breathing_pattern.average_breath_interval
        if target_interval < self.MIN_INTERVAL_BETWEEN_BREATHS:
            target_interval = self.IDEAL_BREATH_INTERVAL
        
        estimated_breaths = int(audio_duration / target_interval)
        
        # Apply constraints
        if max_breaths is not None:
            estimated_breaths = min(estimated_breaths, max_breaths)
        estimated_breaths = max(estimated_breaths, min_breaths)
        
        # Don't inject more breaths than we have points
        estimated_breaths = min(estimated_breaths, len(insertion_points))
        
        if estimated_breaths == 0:
            return []
        
        # Select points ensuring minimum spacing
        selected = []
        last_breath_time = -self.MIN_INTERVAL_BETWEEN_BREATHS
        
        for point in insertion_points:
            if len(selected) >= estimated_breaths:
                break
            
            # Check minimum interval constraint
            if point['time'] - last_breath_time >= self.MIN_INTERVAL_BETWEEN_BREATHS:
                selected.append(point)
                last_breath_time = point['time']
        
        # If we haven't met minimum, try to add more from remaining points
        if len(selected) < min_breaths:
            remaining = [p for p in insertion_points if p not in selected]
            for point in remaining:
                if len(selected) >= min_breaths:
                    break
                # Relax spacing constraint slightly
                times = [s['time'] for s in selected]
                min_dist = min([abs(point['time'] - t) for t in times]) if times else float('inf')
                if min_dist >= self.MIN_INTERVAL_BETWEEN_BREATHS * 0.7:
                    selected.append(point)
        
        # Sort by time
        selected.sort(key=lambda x: x['time'])
        
        logger.debug(f"Selected {len(selected)} breath points from {len(insertion_points)} candidates")
        return selected
    
    def _generate_breath_sounds(
        self,
        insertion_points: List[Dict[str, Any]],
        breathing_pattern: BreathingPattern,
        sr: int,
        intensity_scale: float
    ) -> List[np.ndarray]:
        """
        Generate breath sounds for each insertion point.
        
        Args:
            insertion_points: Selected insertion points
            breathing_pattern: Reference breathing pattern
            sr: Sample rate
            intensity_scale: Intensity scaling factor
            
        Returns:
            List of breath sound arrays
        """
        breath_sounds = []
        
        for i, point in enumerate(insertion_points):
            # Determine breath type based on context
            if point['context']['position_type'] == 'start':
                breath_type = BreathType.INHALATION
            elif point['context']['position_type'] == 'end':
                breath_type = BreathType.EXHALATION
            else:
                # Alternate between inhalation and phrase breath
                breath_type = BreathType.INHALATION if i % 2 == 0 else BreathType.PHRASE_BREATH
            
            # Calculate breath duration based on available silence
            max_duration = point['silence_duration'] * 0.7  # Leave some padding
            target_duration = min(
                breathing_pattern.average_breath_duration,
                max_duration
            )
            target_duration = max(0.15, target_duration)  # Minimum 150ms
            
            # Calculate intensity based on context and pattern
            base_intensity = breathing_pattern.breath_intensity_mean
            # Adjust based on surrounding speech energy
            context_energy = (
                point['context']['speech_before_energy'] +
                point['context']['speech_after_energy']
            ) / 2
            intensity = base_intensity * (0.5 + context_energy) * intensity_scale
            intensity = np.clip(intensity, 0.02, 0.15)  # Keep breaths subtle
            
            # Generate the breath sound
            breath = self._synthesize_breath(
                breathing_pattern,
                breath_type,
                target_duration,
                intensity,
                sr
            )
            
            breath_sounds.append(breath)
        
        return breath_sounds
    
    def _synthesize_breath(
        self,
        breathing_pattern: BreathingPattern,
        breath_type: BreathType,
        duration: float,
        intensity: float,
        sr: int
    ) -> np.ndarray:
        """
        Synthesize a breath sound matching the reference pattern.
        
        Args:
            breathing_pattern: Reference breathing pattern
            breath_type: Type of breath to generate
            duration: Target duration in seconds
            intensity: Target intensity
            sr: Sample rate
            
        Returns:
            Synthesized breath audio array
        """
        num_samples = int(duration * sr)
        
        # Generate noise base
        noise = np.random.randn(num_samples)
        
        # Apply spectral shaping based on reference breath profile
        # Use STFT to shape the noise
        stft = librosa.stft(noise, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Get the spectral profile from the breathing pattern
        spectral_profile = breathing_pattern.breath_spectral_profile
        
        # Ensure profile matches STFT frequency bins
        if len(spectral_profile) != stft.shape[0]:
            spectral_profile = np.interp(
                np.linspace(0, 1, stft.shape[0]),
                np.linspace(0, 1, len(spectral_profile)),
                spectral_profile
            )
        
        # Apply spectral shaping
        shaped_stft = stft * spectral_profile[:, np.newaxis]
        
        # Convert back to time domain
        shaped_noise = librosa.istft(shaped_stft, hop_length=self.hop_length, length=num_samples)
        
        # Apply envelope based on breath type
        envelope = self._create_breath_envelope(breath_type, num_samples, sr)
        
        # Apply envelope and intensity
        breath = shaped_noise * envelope * intensity
        
        # Apply gentle lowpass filter to smooth
        nyquist = sr / 2
        cutoff = 4000 / nyquist
        b, a = butter(4, cutoff, btype='low')
        breath = filtfilt(b, a, breath)
        
        # Normalize to target intensity
        current_rms = np.sqrt(np.mean(breath ** 2))
        if current_rms > 0:
            breath = breath * (intensity / current_rms)
        
        return breath
    
    def _create_breath_envelope(
        self,
        breath_type: BreathType,
        num_samples: int,
        sr: int
    ) -> np.ndarray:
        """
        Create an envelope for the breath based on its type.
        
        Args:
            breath_type: Type of breath
            num_samples: Number of samples
            sr: Sample rate
            
        Returns:
            Envelope array
        """
        envelope = np.ones(num_samples)
        
        # Define attack and release times based on breath type
        if breath_type == BreathType.INHALATION:
            # Quick attack, gradual release
            attack_samples = int(0.05 * num_samples)
            release_samples = int(0.4 * num_samples)
        elif breath_type == BreathType.EXHALATION:
            # Gradual attack, quick release
            attack_samples = int(0.3 * num_samples)
            release_samples = int(0.1 * num_samples)
        elif breath_type == BreathType.PHRASE_BREATH:
            # Balanced attack and release
            attack_samples = int(0.15 * num_samples)
            release_samples = int(0.25 * num_samples)
        elif breath_type == BreathType.DEEP_BREATH:
            # Longer attack and release
            attack_samples = int(0.2 * num_samples)
            release_samples = int(0.3 * num_samples)
        else:
            # Default: pause breath
            attack_samples = int(0.1 * num_samples)
            release_samples = int(0.2 * num_samples)
        
        # Ensure we don't exceed array bounds
        attack_samples = min(attack_samples, num_samples // 3)
        release_samples = min(release_samples, num_samples // 3)
        
        # Apply attack (fade in)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 0.5
        
        # Apply release (fade out)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples) ** 0.5
        
        return envelope
    
    def _inject_breaths_into_audio(
        self,
        audio: np.ndarray,
        breath_sounds: List[np.ndarray],
        insertion_points: List[Dict[str, Any]],
        sr: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Inject breath sounds into the audio with proper crossfading.
        
        Args:
            audio: Original synthesized audio
            breath_sounds: List of breath sounds to inject
            insertion_points: Insertion point metadata
            sr: Sample rate
            
        Returns:
            Tuple of (audio with breaths, injection metadata)
        """
        result = audio.copy()
        injection_metadata = []
        
        # Crossfade duration in samples
        crossfade_samples = int(0.02 * sr)  # 20ms crossfade
        
        for breath, point in zip(breath_sounds, insertion_points):
            insertion_sample = int(point['time'] * sr)
            breath_length = len(breath)
            
            # Calculate start and end positions
            start_pos = insertion_sample - breath_length // 2
            end_pos = start_pos + breath_length
            
            # Ensure we're within bounds
            if start_pos < 0:
                breath = breath[-start_pos:]
                start_pos = 0
            if end_pos > len(result):
                breath = breath[:len(result) - start_pos]
                end_pos = len(result)
            
            if len(breath) < crossfade_samples * 2:
                continue  # Breath too short to inject properly
            
            # Create crossfade envelope for smooth blending
            blend_envelope = np.ones(len(breath))
            
            # Fade in at start
            if crossfade_samples > 0 and crossfade_samples < len(blend_envelope):
                blend_envelope[:crossfade_samples] = np.linspace(0, 1, crossfade_samples)
            
            # Fade out at end
            if crossfade_samples > 0 and crossfade_samples < len(blend_envelope):
                blend_envelope[-crossfade_samples:] = np.linspace(1, 0, crossfade_samples)
            
            # Blend breath with existing audio
            existing_segment = result[start_pos:start_pos + len(breath)]
            blended = existing_segment * (1 - blend_envelope * 0.5) + breath * blend_envelope
            
            # Insert blended audio
            result[start_pos:start_pos + len(breath)] = blended
            
            # Record metadata
            injection_metadata.append({
                'time': point['time'],
                'duration': len(breath) / sr,
                'position_type': point['context']['position_type'],
                'start_sample': start_pos,
                'end_sample': start_pos + len(breath)
            })
        
        return result, injection_metadata
    
    def inject_breathing_from_reference(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: np.ndarray,
        synthesized_sr: Optional[int] = None,
        reference_sr: Optional[int] = None,
        intensity_scale: float = 1.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Convenience method to analyze reference and inject breathing in one call.
        
        Args:
            synthesized_audio: Synthesized speech to enhance
            reference_audio: Reference audio to analyze for breathing patterns
            synthesized_sr: Sample rate of synthesized audio
            reference_sr: Sample rate of reference audio
            intensity_scale: Breath intensity scaling factor
            
        Returns:
            Tuple of (enhanced audio, injection metadata)
            
        Validates: Requirements 5.1 (natural breathing patterns at appropriate intervals)
        """
        synth_sr = synthesized_sr or self.sample_rate
        ref_sr = reference_sr or self.sample_rate
        
        # Analyze breathing patterns from reference
        analysis_result = self.analyzer.analyze_breathing_patterns(
            reference_audio, ref_sr
        )
        
        logger.info(
            f"Analyzed reference: {len(analysis_result.pattern.breath_events)} breaths, "
            f"quality={analysis_result.quality_score:.2f}, "
            f"confidence={analysis_result.confidence:.2f}"
        )
        
        # Inject breathing into synthesized audio
        return self.inject_breathing(
            synthesized_audio,
            analysis_result.pattern,
            synth_sr,
            intensity_scale
        )


# Global instances
_breathing_analyzer: Optional[BreathingPatternAnalyzer] = None
_breathing_injector: Optional[NaturalBreathingInjector] = None


def get_breathing_analyzer() -> BreathingPatternAnalyzer:
    """Get or create global breathing pattern analyzer instance."""
    global _breathing_analyzer
    if _breathing_analyzer is None:
        _breathing_analyzer = BreathingPatternAnalyzer()
    return _breathing_analyzer


def get_breathing_injector() -> NaturalBreathingInjector:
    """Get or create global natural breathing injector instance."""
    global _breathing_injector
    if _breathing_injector is None:
        _breathing_injector = NaturalBreathingInjector()
    return _breathing_injector


class FillerType(Enum):
    """Types of filler sounds and hesitations in speech."""
    UH = "uh"  # Short hesitation sound
    UM = "um"  # Longer hesitation with nasal
    ER = "er"  # British-style hesitation
    AH = "ah"  # Open vowel hesitation
    HMM = "hmm"  # Thinking sound
    ELONGATION = "elongation"  # Word stretching
    REPETITION = "repetition"  # Word/syllable repetition
    FALSE_START = "false_start"  # Incomplete word


@dataclass
class HesitationEvent:
    """Represents a single hesitation or filler event in audio."""
    filler_type: FillerType
    start_time: float
    end_time: float
    duration: float
    pitch_mean: float
    pitch_std: float
    intensity: float
    spectral_profile: np.ndarray
    context: str  # 'sentence_start', 'mid_sentence', 'before_complex'


@dataclass
class HesitationPattern:
    """Complete hesitation pattern analysis from reference audio."""
    hesitation_events: List[HesitationEvent]
    filler_frequency: Dict[FillerType, float]  # Frequency of each filler type
    average_hesitation_rate: float  # Hesitations per minute
    average_duration: float
    pitch_characteristics: Dict[str, float]
    preferred_fillers: List[FillerType]  # Most common fillers for this speaker
    hesitation_spectral_templates: Dict[FillerType, np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class HesitationAnalysisResult:
    """Result of hesitation pattern analysis."""
    pattern: HesitationPattern
    quality_score: float
    confidence: float
    analysis_metadata: Dict[str, Any]


class HesitationPatternAnalyzer:
    """
    Analyzes hesitation and filler patterns from reference audio.
    
    Hesitations and fillers are crucial micro-expressions that make speech
    sound natural. This analyzer extracts:
    - Types of fillers used (uh, um, er, etc.)
    - Frequency and distribution of hesitations
    - Spectral characteristics of each filler type
    - Context where hesitations typically occur
    
    Validates: Requirements 5.2 (hesitations and natural speech artifacts)
    """
    
    # Filler detection parameters
    FILLER_DURATION_MIN = 0.1  # seconds
    FILLER_DURATION_MAX = 1.0  # seconds
    FILLER_PITCH_RANGE = (80, 300)  # Hz - typical filler pitch range
    
    # Spectral characteristics for filler classification
    FILLER_FORMANT_RANGES = {
        FillerType.UH: {'F1': (500, 700), 'F2': (1000, 1400)},
        FillerType.UM: {'F1': (400, 600), 'F2': (900, 1200)},
        FillerType.ER: {'F1': (450, 650), 'F2': (1200, 1600)},
        FillerType.AH: {'F1': (600, 900), 'F2': (1000, 1400)},
        FillerType.HMM: {'F1': (200, 400), 'F2': (800, 1200)},
    }
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize hesitation pattern analyzer."""
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        logger.info("Hesitation Pattern Analyzer initialized")

    
    def analyze_hesitation_patterns(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> HesitationAnalysisResult:
        """
        Analyze hesitation and filler patterns from reference audio.
        
        Args:
            audio: Reference audio array
            sample_rate: Sample rate (uses default if not provided)
            
        Returns:
            HesitationAnalysisResult with complete pattern analysis
            
        Validates: Requirements 5.2 (hesitations and natural speech artifacts)
        """
        sr = sample_rate or self.sample_rate
        
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        logger.info(f"Analyzing hesitation patterns in {len(audio)/sr:.2f}s audio")
        
        # Step 1: Detect potential hesitation regions
        candidates = self._detect_hesitation_candidates(audio, sr)
        
        # Step 2: Classify and validate hesitation events
        events = self._classify_hesitation_events(audio, sr, candidates)
        
        # Step 3: Extract spectral templates for each filler type
        templates = self._extract_filler_templates(audio, sr, events)
        
        # Step 4: Calculate hesitation statistics
        stats = self._calculate_hesitation_statistics(events, len(audio) / sr)
        
        # Create hesitation pattern
        pattern = HesitationPattern(
            hesitation_events=events,
            filler_frequency=stats['filler_frequency'],
            average_hesitation_rate=stats['hesitation_rate'],
            average_duration=stats['average_duration'],
            pitch_characteristics=stats['pitch_characteristics'],
            preferred_fillers=stats['preferred_fillers'],
            hesitation_spectral_templates=templates,
            metadata={'audio_duration': len(audio) / sr, 'num_hesitations': len(events)}
        )
        
        quality = self._assess_pattern_quality(pattern)
        confidence = self._calculate_confidence(events, len(audio) / sr)
        
        return HesitationAnalysisResult(
            pattern=pattern, quality_score=quality, confidence=confidence,
            analysis_metadata={'candidates_found': len(candidates), 'events_validated': len(events)}
        )

    
    def _detect_hesitation_candidates(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Tuple[int, int]]:
        """
        Detect candidate hesitation regions using acoustic features.
        
        Hesitations are characterized by:
        - Relatively stable pitch (sustained vowel-like sounds)
        - Lower energy than surrounding speech
        - Specific formant patterns
        """
        candidates = []
        
        # Compute pitch using pyin
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=self.FILLER_PITCH_RANGE[0], 
            fmax=self.FILLER_PITCH_RANGE[1], sr=sr,
            hop_length=self.hop_length
        )
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        rms_norm = rms / (np.max(rms) + 1e-8)
        
        # Compute spectral flatness
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        
        # Hesitation detection: voiced, moderate energy, low pitch variation
        if voiced_flag is None:
            return candidates
            
        # Find regions with stable pitch (low pitch variation)
        pitch_valid = ~np.isnan(f0)
        
        # Compute local pitch stability
        pitch_stability = np.zeros_like(f0)
        window = 5
        for i in range(window, len(f0) - window):
            if np.all(pitch_valid[i-window:i+window]):
                local_std = np.std(f0[i-window:i+window])
                pitch_stability[i] = 1.0 / (1.0 + local_std / 10)
        
        # Hesitation score: voiced + stable pitch + moderate energy
        hesitation_score = (
            voiced_flag.astype(float) * 0.4 +
            pitch_stability * 0.3 +
            (0.3 - np.abs(rms_norm - 0.2)) * 0.3
        )
        
        hesitation_score = gaussian_filter1d(hesitation_score, sigma=3)
        threshold = np.percentile(hesitation_score[hesitation_score > 0], 75) if np.any(hesitation_score > 0) else 0.5
        
        # Find contiguous regions
        in_hesitation = False
        start_frame = 0
        
        for i, score in enumerate(hesitation_score):
            if score > threshold and not in_hesitation:
                start_frame = i
                in_hesitation = True
            elif score <= threshold and in_hesitation:
                end_frame = i
                start_sample = start_frame * self.hop_length
                end_sample = end_frame * self.hop_length
                duration = (end_sample - start_sample) / sr
                
                if self.FILLER_DURATION_MIN <= duration <= self.FILLER_DURATION_MAX:
                    candidates.append((start_sample, end_sample))
                in_hesitation = False
        
        logger.debug(f"Found {len(candidates)} hesitation candidates")
        return candidates

    
    def _classify_hesitation_events(
        self,
        audio: np.ndarray,
        sr: int,
        candidates: List[Tuple[int, int]]
    ) -> List[HesitationEvent]:
        """Classify and validate hesitation candidates into events."""
        events = []
        
        for start_sample, end_sample in candidates:
            segment = audio[start_sample:min(end_sample, len(audio))]
            if len(segment) < int(self.FILLER_DURATION_MIN * sr):
                continue
            
            # Classify filler type based on formants
            filler_type = self._classify_filler_type(segment, sr)
            
            # Extract pitch characteristics
            f0, voiced, _ = librosa.pyin(
                segment, fmin=self.FILLER_PITCH_RANGE[0],
                fmax=self.FILLER_PITCH_RANGE[1], sr=sr
            )
            valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([150])
            pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 150.0
            pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 10.0
            
            # Calculate intensity
            intensity = float(np.sqrt(np.mean(segment ** 2)))
            
            # Extract spectral profile
            stft = librosa.stft(segment, n_fft=self.n_fft)
            spectral_profile = np.mean(np.abs(stft), axis=1)
            spectral_profile = spectral_profile / (np.max(spectral_profile) + 1e-8)
            
            # Determine context
            context = self._determine_context(audio, sr, start_sample, end_sample)
            
            events.append(HesitationEvent(
                filler_type=filler_type,
                start_time=start_sample / sr,
                end_time=end_sample / sr,
                duration=(end_sample - start_sample) / sr,
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                intensity=intensity,
                spectral_profile=spectral_profile,
                context=context
            ))
        
        logger.debug(f"Validated {len(events)} hesitation events")
        return events

    
    def _classify_filler_type(self, segment: np.ndarray, sr: int) -> FillerType:
        """Classify filler type based on formant analysis."""
        # Extract formants using LPC
        try:
            # Compute LPC coefficients
            lpc_order = int(sr / 1000) + 2
            lpc_coeffs = librosa.lpc(segment, order=lpc_order)
            
            # Find formant frequencies from LPC roots
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            angles = np.angle(roots)
            freqs = angles * sr / (2 * np.pi)
            freqs = freqs[(freqs > 100) & (freqs < 4000)]
            freqs = np.sort(freqs)
            
            if len(freqs) >= 2:
                f1, f2 = freqs[0], freqs[1]
                
                # Match against known filler formant ranges
                best_match = FillerType.UH
                best_score = 0
                
                for filler_type, ranges in self.FILLER_FORMANT_RANGES.items():
                    f1_range, f2_range = ranges['F1'], ranges['F2']
                    f1_score = 1.0 if f1_range[0] <= f1 <= f1_range[1] else 0.5
                    f2_score = 1.0 if f2_range[0] <= f2 <= f2_range[1] else 0.5
                    score = f1_score * f2_score
                    
                    if score > best_score:
                        best_score = score
                        best_match = filler_type
                
                return best_match
        except Exception:
            pass
        
        return FillerType.UH  # Default
    
    def _determine_context(
        self, audio: np.ndarray, sr: int, start: int, end: int
    ) -> str:
        """Determine the context where hesitation occurs."""
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        threshold = np.percentile(rms, 50)
        
        frame = start // self.hop_length
        context_frames = int(0.5 * sr / self.hop_length)
        
        before_start = max(0, frame - context_frames)
        speech_before = np.mean(rms[before_start:frame] > threshold) if frame > before_start else 0
        
        if speech_before < 0.3:
            return 'sentence_start'
        return 'mid_sentence'

    
    def _extract_filler_templates(
        self,
        audio: np.ndarray,
        sr: int,
        events: List[HesitationEvent]
    ) -> Dict[FillerType, np.ndarray]:
        """Extract spectral templates for each filler type."""
        templates = {}
        filler_profiles = {ft: [] for ft in FillerType}
        
        for event in events:
            start = int(event.start_time * sr)
            end = int(event.end_time * sr)
            if end <= len(audio):
                filler_profiles[event.filler_type].append(event.spectral_profile)
        
        for filler_type, profiles in filler_profiles.items():
            if profiles:
                templates[filler_type] = np.mean(profiles, axis=0)
            else:
                # Create default template
                templates[filler_type] = self._create_default_template(filler_type)
        
        return templates
    
    def _create_default_template(self, filler_type: FillerType) -> np.ndarray:
        """Create a default spectral template for a filler type."""
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        template = np.zeros_like(freqs)
        
        if filler_type in self.FILLER_FORMANT_RANGES:
            ranges = self.FILLER_FORMANT_RANGES[filler_type]
            f1_center = np.mean(ranges['F1'])
            f2_center = np.mean(ranges['F2'])
            
            # Create formant peaks
            template += np.exp(-((freqs - f1_center) ** 2) / (2 * 100 ** 2))
            template += 0.7 * np.exp(-((freqs - f2_center) ** 2) / (2 * 150 ** 2))
        else:
            # Generic vowel-like spectrum
            template = np.exp(-((freqs - 600) ** 2) / (2 * 200 ** 2))
        
        return template / (np.max(template) + 1e-8)

    
    def _calculate_hesitation_statistics(
        self,
        events: List[HesitationEvent],
        audio_duration: float
    ) -> Dict[str, Any]:
        """Calculate statistical measures of hesitation patterns."""
        if not events:
            return {
                'filler_frequency': {ft: 0.0 for ft in FillerType},
                'hesitation_rate': 0.0,
                'average_duration': 0.3,
                'pitch_characteristics': {'mean': 150.0, 'std': 20.0},
                'preferred_fillers': [FillerType.UH, FillerType.UM]
            }
        
        # Count filler types
        filler_counts = {ft: 0 for ft in FillerType}
        for event in events:
            filler_counts[event.filler_type] += 1
        
        total = len(events)
        filler_frequency = {ft: count / total for ft, count in filler_counts.items()}
        
        # Calculate hesitation rate (per minute)
        hesitation_rate = (total / audio_duration) * 60 if audio_duration > 0 else 0
        
        # Average duration
        avg_duration = np.mean([e.duration for e in events])
        
        # Pitch characteristics
        pitch_means = [e.pitch_mean for e in events]
        pitch_characteristics = {
            'mean': float(np.mean(pitch_means)),
            'std': float(np.std(pitch_means))
        }
        
        # Preferred fillers (top 2)
        sorted_fillers = sorted(filler_counts.items(), key=lambda x: x[1], reverse=True)
        preferred = [ft for ft, _ in sorted_fillers[:2] if filler_counts[ft] > 0]
        if not preferred:
            preferred = [FillerType.UH, FillerType.UM]
        
        return {
            'filler_frequency': filler_frequency,
            'hesitation_rate': float(hesitation_rate),
            'average_duration': float(avg_duration),
            'pitch_characteristics': pitch_characteristics,
            'preferred_fillers': preferred
        }

    
    def _assess_pattern_quality(self, pattern: HesitationPattern) -> float:
        """Assess the quality of the extracted hesitation pattern."""
        score = 0.0
        
        num_events = len(pattern.hesitation_events)
        if num_events >= 3:
            score += 0.3
        elif num_events >= 1:
            score += 0.15
        
        # Check hesitation rate is reasonable (1-10 per minute is typical)
        if 1 <= pattern.average_hesitation_rate <= 15:
            score += 0.25
        elif 0.5 <= pattern.average_hesitation_rate <= 20:
            score += 0.1
        
        # Check duration is reasonable
        if 0.15 <= pattern.average_duration <= 0.6:
            score += 0.25
        
        # Check pitch characteristics
        if 80 <= pattern.pitch_characteristics['mean'] <= 300:
            score += 0.2
        
        return float(np.clip(score, 0, 1))
    
    def _calculate_confidence(
        self,
        events: List[HesitationEvent],
        audio_duration: float
    ) -> float:
        """Calculate confidence in the hesitation pattern analysis."""
        if audio_duration < 5.0:
            return 0.3
        
        # Expected hesitations based on duration (roughly 2-5 per minute)
        expected = (audio_duration / 60) * 3
        actual = len(events)
        
        if expected > 0:
            ratio = actual / expected
            if 0.3 <= ratio <= 3.0:
                confidence = 0.7
            elif 0.1 <= ratio <= 5.0:
                confidence = 0.5
            else:
                confidence = 0.3
        else:
            confidence = 0.5
        
        if audio_duration > 30:
            confidence = min(1.0, confidence + 0.15)
        
        return float(confidence)



class HesitationInjector:
    """
    Injects hesitation and filler sounds into synthesized speech.
    
    This class uses analyzed hesitation patterns from reference audio to inject
    realistic filler sounds (uh, um, er, etc.) that match the speaker's natural
    hesitation style.
    
    Validates: Requirements 5.2 (hesitations and natural speech artifacts)
    """
    
    MIN_PAUSE_FOR_HESITATION = 0.4  # Minimum pause to insert hesitation
    MAX_PAUSE_FOR_HESITATION = 2.0  # Maximum pause where hesitation makes sense
    MIN_INTERVAL_BETWEEN_HESITATIONS = 5.0  # Minimum time between hesitations
    DEFAULT_HESITATION_RATE = 2.0  # Hesitations per minute if no pattern
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize the hesitation injector."""
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
        self.analyzer = HesitationPatternAnalyzer(sample_rate)
        logger.info("Hesitation Injector initialized")

    
    def inject_hesitations(
        self,
        synthesized_audio: np.ndarray,
        hesitation_pattern: HesitationPattern,
        sample_rate: Optional[int] = None,
        intensity_scale: float = 1.0,
        hesitation_probability: float = 0.3
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Inject hesitation sounds into synthesized audio.
        
        Args:
            synthesized_audio: The synthesized speech audio array
            hesitation_pattern: Analyzed hesitation pattern from reference
            sample_rate: Sample rate (uses default if not provided)
            intensity_scale: Scale factor for hesitation intensity
            hesitation_probability: Probability of inserting at valid points
            
        Returns:
            Tuple of (audio with hesitations, list of injection metadata)
            
        Validates: Requirements 5.2 (hesitations and natural speech artifacts)
        """
        sr = sample_rate or self.sample_rate
        
        if sr != self.sample_rate:
            synthesized_audio = librosa.resample(
                synthesized_audio, orig_sr=sr, target_sr=self.sample_rate
            )
            sr = self.sample_rate
        
        audio_duration = len(synthesized_audio) / sr
        logger.info(f"Injecting hesitations into {audio_duration:.2f}s audio")
        
        # Find appropriate insertion points
        insertion_points = self._find_hesitation_points(synthesized_audio, sr)
        
        if not insertion_points:
            logger.warning("No suitable hesitation insertion points found")
            return synthesized_audio, []
        
        # Select points based on pattern and probability
        selected = self._select_hesitation_points(
            insertion_points, hesitation_pattern, 
            audio_duration, hesitation_probability
        )
        
        if not selected:
            return synthesized_audio, []
        
        # Generate hesitation sounds
        hesitations = self._generate_hesitations(
            selected, hesitation_pattern, sr, intensity_scale
        )
        
        # Inject into audio
        result, metadata = self._inject_hesitations_into_audio(
            synthesized_audio, hesitations, selected, sr
        )
        
        logger.info(f"Injected {len(metadata)} hesitations into audio")
        return result, metadata

    
    def _find_hesitation_points(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Dict[str, Any]]:
        """Find appropriate points to insert hesitations."""
        points = []
        
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        rms_smooth = gaussian_filter1d(rms, sigma=3)
        
        silence_threshold = 0.02
        speech_threshold = 0.08
        
        silence_mask = rms_smooth < silence_threshold
        
        in_silence = False
        silence_start = 0
        
        for i in range(len(silence_mask)):
            if silence_mask[i] and not in_silence:
                silence_start = i
                in_silence = True
            elif not silence_mask[i] and in_silence:
                silence_end = i
                in_silence = False
                
                start_time = silence_start * self.hop_length / sr
                end_time = silence_end * self.hop_length / sr
                duration = end_time - start_time
                
                if self.MIN_PAUSE_FOR_HESITATION <= duration <= self.MAX_PAUSE_FOR_HESITATION:
                    # Check context
                    context_frames = int(0.3 * sr / self.hop_length)
                    before_start = max(0, silence_start - context_frames)
                    after_end = min(len(rms_smooth), silence_end + context_frames)
                    
                    speech_before = np.mean(rms_smooth[before_start:silence_start] > speech_threshold)
                    speech_after = np.mean(rms_smooth[silence_end:after_end] > speech_threshold)
                    
                    # Prefer mid-sentence pauses
                    if speech_before > 0.3 and speech_after > 0.3:
                        suitability = 0.8
                        context = 'mid_sentence'
                    elif speech_after > 0.3:
                        suitability = 0.6
                        context = 'sentence_start'
                    else:
                        suitability = 0.3
                        context = 'sentence_end'
                    
                    points.append({
                        'time': (start_time + end_time) / 2,
                        'silence_start': start_time,
                        'silence_end': end_time,
                        'duration': duration,
                        'context': context,
                        'suitability': suitability
                    })
        
        points.sort(key=lambda x: x['suitability'], reverse=True)
        return points

    
    def _select_hesitation_points(
        self,
        points: List[Dict[str, Any]],
        pattern: HesitationPattern,
        audio_duration: float,
        probability: float
    ) -> List[Dict[str, Any]]:
        """Select which points to use for hesitation injection."""
        if not points:
            return []
        
        # Calculate target number based on pattern rate
        rate = pattern.average_hesitation_rate if pattern.average_hesitation_rate > 0 else self.DEFAULT_HESITATION_RATE
        target_count = int((audio_duration / 60) * rate * probability)
        target_count = max(0, min(target_count, len(points)))
        
        if target_count == 0:
            return []
        
        # Select points with minimum spacing
        selected = []
        last_time = -self.MIN_INTERVAL_BETWEEN_HESITATIONS
        
        for point in points:
            if len(selected) >= target_count:
                break
            if point['time'] - last_time >= self.MIN_INTERVAL_BETWEEN_HESITATIONS:
                selected.append(point)
                last_time = point['time']
        
        selected.sort(key=lambda x: x['time'])
        return selected
    
    def _generate_hesitations(
        self,
        points: List[Dict[str, Any]],
        pattern: HesitationPattern,
        sr: int,
        intensity_scale: float
    ) -> List[np.ndarray]:
        """Generate hesitation sounds for each point."""
        hesitations = []
        preferred = pattern.preferred_fillers if pattern.preferred_fillers else [FillerType.UH, FillerType.UM]
        
        for i, point in enumerate(points):
            # Select filler type
            filler_type = preferred[i % len(preferred)]
            
            # Calculate duration
            max_dur = point['duration'] * 0.6
            target_dur = min(pattern.average_duration, max_dur)
            target_dur = max(0.15, min(target_dur, 0.5))
            
            # Get spectral template
            template = pattern.hesitation_spectral_templates.get(
                filler_type, 
                self.analyzer._create_default_template(filler_type)
            )
            
            # Calculate intensity
            intensity = 0.08 * intensity_scale
            
            # Synthesize
            hesitation = self._synthesize_hesitation(
                filler_type, template, target_dur,
                pattern.pitch_characteristics['mean'],
                intensity, sr
            )
            hesitations.append(hesitation)
        
        return hesitations

    
    def _synthesize_hesitation(
        self,
        filler_type: FillerType,
        spectral_template: np.ndarray,
        duration: float,
        pitch: float,
        intensity: float,
        sr: int
    ) -> np.ndarray:
        """Synthesize a hesitation sound."""
        num_samples = int(duration * sr)
        
        # Generate voiced source (glottal pulse train)
        t = np.arange(num_samples) / sr
        period = 1.0 / pitch
        
        # Create glottal-like source
        source = np.zeros(num_samples)
        pulse_positions = np.arange(0, duration, period)
        
        for pos in pulse_positions:
            idx = int(pos * sr)
            if idx < num_samples:
                # Simple glottal pulse approximation
                pulse_len = min(int(period * sr * 0.4), num_samples - idx)
                pulse = np.sin(np.linspace(0, np.pi, pulse_len)) ** 2
                source[idx:idx + pulse_len] += pulse
        
        # Add slight noise for naturalness
        noise = np.random.randn(num_samples) * 0.1
        source = source + noise
        
        # Apply spectral shaping
        stft = librosa.stft(source, n_fft=self.n_fft, hop_length=self.hop_length)
        
        if len(spectral_template) != stft.shape[0]:
            spectral_template = np.interp(
                np.linspace(0, 1, stft.shape[0]),
                np.linspace(0, 1, len(spectral_template)),
                spectral_template
            )
        
        shaped_stft = stft * spectral_template[:, np.newaxis]
        hesitation = librosa.istft(shaped_stft, hop_length=self.hop_length, length=num_samples)
        
        # Apply envelope
        envelope = self._create_hesitation_envelope(filler_type, num_samples)
        hesitation = hesitation * envelope
        
        # Normalize to target intensity
        current_rms = np.sqrt(np.mean(hesitation ** 2))
        if current_rms > 0:
            hesitation = hesitation * (intensity / current_rms)
        
        return hesitation

    
    def _create_hesitation_envelope(
        self,
        filler_type: FillerType,
        num_samples: int
    ) -> np.ndarray:
        """Create envelope for hesitation sound."""
        envelope = np.ones(num_samples)
        
        # Different envelope shapes for different fillers
        if filler_type in [FillerType.UH, FillerType.AH]:
            attack = int(0.1 * num_samples)
            release = int(0.2 * num_samples)
        elif filler_type == FillerType.UM:
            attack = int(0.1 * num_samples)
            release = int(0.3 * num_samples)  # Longer release for nasal
        elif filler_type == FillerType.HMM:
            attack = int(0.15 * num_samples)
            release = int(0.15 * num_samples)
        else:
            attack = int(0.1 * num_samples)
            release = int(0.2 * num_samples)
        
        attack = min(attack, num_samples // 3)
        release = min(release, num_samples // 3)
        
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack) ** 0.7
        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release) ** 0.7
        
        return envelope
    
    def _inject_hesitations_into_audio(
        self,
        audio: np.ndarray,
        hesitations: List[np.ndarray],
        points: List[Dict[str, Any]],
        sr: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Inject hesitation sounds into audio with crossfading."""
        result = audio.copy()
        metadata = []
        crossfade = int(0.015 * sr)
        
        for hesitation, point in zip(hesitations, points):
            insert_sample = int(point['time'] * sr)
            hes_len = len(hesitation)
            
            start = insert_sample - hes_len // 2
            end = start + hes_len
            
            if start < 0:
                hesitation = hesitation[-start:]
                start = 0
            if end > len(result):
                hesitation = hesitation[:len(result) - start]
                end = len(result)
            
            if len(hesitation) < crossfade * 2:
                continue
            
            # Crossfade blend
            blend = np.ones(len(hesitation))
            if crossfade > 0 and crossfade < len(blend):
                blend[:crossfade] = np.linspace(0, 1, crossfade)
                blend[-crossfade:] = np.linspace(1, 0, crossfade)
            
            existing = result[start:start + len(hesitation)]
            blended = existing * (1 - blend * 0.6) + hesitation * blend
            result[start:start + len(hesitation)] = blended
            
            metadata.append({
                'time': point['time'],
                'duration': len(hesitation) / sr,
                'context': point['context'],
                'start_sample': start,
                'end_sample': start + len(hesitation)
            })
        
        return result, metadata

    
    def inject_hesitations_from_reference(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: np.ndarray,
        synthesized_sr: Optional[int] = None,
        reference_sr: Optional[int] = None,
        intensity_scale: float = 1.0,
        hesitation_probability: float = 0.3
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Convenience method to analyze reference and inject hesitations.
        
        Args:
            synthesized_audio: Synthesized speech to enhance
            reference_audio: Reference audio to analyze
            synthesized_sr: Sample rate of synthesized audio
            reference_sr: Sample rate of reference audio
            intensity_scale: Hesitation intensity scaling
            hesitation_probability: Probability of inserting at valid points
            
        Returns:
            Tuple of (enhanced audio, injection metadata)
            
        Validates: Requirements 5.2 (hesitations and natural speech artifacts)
        """
        synth_sr = synthesized_sr or self.sample_rate
        ref_sr = reference_sr or self.sample_rate
        
        # Analyze hesitation patterns
        analysis = self.analyzer.analyze_hesitation_patterns(reference_audio, ref_sr)
        
        logger.info(
            f"Analyzed reference: {len(analysis.pattern.hesitation_events)} hesitations, "
            f"rate={analysis.pattern.average_hesitation_rate:.1f}/min, "
            f"quality={analysis.quality_score:.2f}"
        )
        
        return self.inject_hesitations(
            synthesized_audio, analysis.pattern, synth_sr,
            intensity_scale, hesitation_probability
        )


# Global instances
_breathing_analyzer: Optional[BreathingPatternAnalyzer] = None
_breathing_injector: Optional[NaturalBreathingInjector] = None
_hesitation_analyzer: Optional[HesitationPatternAnalyzer] = None
_hesitation_injector: Optional[HesitationInjector] = None


def get_breathing_analyzer() -> BreathingPatternAnalyzer:
    """Get or create global breathing pattern analyzer instance."""
    global _breathing_analyzer
    if _breathing_analyzer is None:
        _breathing_analyzer = BreathingPatternAnalyzer()
    return _breathing_analyzer


def get_breathing_injector() -> NaturalBreathingInjector:
    """Get or create global natural breathing injector instance."""
    global _breathing_injector
    if _breathing_injector is None:
        _breathing_injector = NaturalBreathingInjector()
    return _breathing_injector


def get_hesitation_analyzer() -> HesitationPatternAnalyzer:
    """Get or create global hesitation pattern analyzer instance."""
    global _hesitation_analyzer
    if _hesitation_analyzer is None:
        _hesitation_analyzer = HesitationPatternAnalyzer()
    return _hesitation_analyzer


def get_hesitation_injector() -> HesitationInjector:
    """Get or create global hesitation injector instance."""
    global _hesitation_injector
    if _hesitation_injector is None:
        _hesitation_injector = HesitationInjector()
    return _hesitation_injector


# ============================================================================
# LIP SMACK AND MOUTH SOUND INJECTION
# ============================================================================
# This section implements lip smack and mouth sound analysis and injection
# to make synthesized speech sound more natural and human-like.
# Validates: Requirements 5.2 (preserve lip smacks, hesitations, and natural speech artifacts)
# ============================================================================


class LipSmackType(Enum):
    """Types of lip smacks and mouth sounds detected in speech."""
    LIP_SMACK = "lip_smack"  # Quick lip separation sound
    TONGUE_CLICK = "tongue_click"  # Tongue clicking against palate
    SALIVA_SOUND = "saliva_sound"  # Wet mouth sounds
    LIP_POP = "lip_pop"  # Popping sound from lips
    MOUTH_OPEN = "mouth_open"  # Sound of mouth opening before speech
    SWALLOW = "swallow"  # Swallowing sound
    TEETH_CLICK = "teeth_click"  # Teeth touching sound


@dataclass
class LipSmackEvent:
    """Represents a single lip smack or mouth sound event in audio."""
    sound_type: LipSmackType
    start_time: float
    end_time: float
    duration: float
    intensity: float
    frequency_peak: float  # Dominant frequency of the sound
    spectral_profile: np.ndarray
    position_context: str  # 'before_speech', 'mid_speech', 'after_pause'
    transient_sharpness: float  # How sharp/sudden the transient is (0-1)


@dataclass
class LipSmackPattern:
    """Complete lip smack and mouth sound pattern analysis from reference audio."""
    lip_smack_events: List[LipSmackEvent]
    sound_type_frequency: Dict[LipSmackType, float]  # Frequency of each sound type
    average_occurrence_rate: float  # Occurrences per minute
    average_duration: float
    average_intensity: float
    intensity_std: float
    preferred_sounds: List[LipSmackType]  # Most common sounds for this speaker
    spectral_templates: Dict[LipSmackType, np.ndarray]  # Templates for each type
    transient_characteristics: Dict[str, float]  # Attack/decay characteristics
    metadata: Dict[str, Any]


@dataclass
class LipSmackAnalysisResult:
    """Result of lip smack and mouth sound pattern analysis."""
    pattern: LipSmackPattern
    quality_score: float
    confidence: float
    analysis_metadata: Dict[str, Any]


class LipSmackPatternAnalyzer:
    """
    Analyzes lip smack and mouth sound patterns from reference audio.
    
    Lip smacks and mouth sounds are subtle but important micro-expressions
    that make speech sound natural. This analyzer extracts:
    - Types of mouth sounds (lip smacks, tongue clicks, etc.)
    - Frequency and distribution of these sounds
    - Spectral characteristics of each sound type
    - Context where these sounds typically occur
    
    Validates: Requirements 5.2 (preserve lip smacks and natural speech artifacts)
    """
    
    # Detection parameters
    LIP_SMACK_DURATION_MIN = 0.01  # seconds - very short transients
    LIP_SMACK_DURATION_MAX = 0.15  # seconds
    TRANSIENT_THRESHOLD = 0.3  # Relative threshold for transient detection
    
    # Frequency characteristics for different mouth sounds
    SOUND_FREQ_RANGES = {
        LipSmackType.LIP_SMACK: (1000, 4000),  # Mid-high frequency click
        LipSmackType.TONGUE_CLICK: (2000, 6000),  # Higher frequency
        LipSmackType.SALIVA_SOUND: (500, 3000),  # Broader, wetter sound
        LipSmackType.LIP_POP: (800, 3000),  # Lower pop sound
        LipSmackType.MOUTH_OPEN: (200, 1500),  # Low frequency opening
        LipSmackType.SWALLOW: (300, 2000),  # Mid-range
        LipSmackType.TEETH_CLICK: (3000, 8000),  # High frequency click
    }
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize lip smack pattern analyzer."""
        self.sample_rate = sample_rate
        self.hop_length = 256  # Smaller hop for transient detection
        self.n_fft = 1024  # Smaller FFT for better time resolution
        logger.info("Lip Smack Pattern Analyzer initialized")

    
    def analyze_lip_smack_patterns(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> LipSmackAnalysisResult:
        """
        Analyze lip smack and mouth sound patterns from reference audio.
        
        Args:
            audio: Reference audio array
            sample_rate: Sample rate (uses default if not provided)
            
        Returns:
            LipSmackAnalysisResult with complete pattern analysis
            
        Validates: Requirements 5.2 (preserve lip smacks and natural speech artifacts)
        """
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        logger.info(f"Analyzing lip smack patterns in {len(audio)/sr:.2f}s audio")
        
        # Step 1: Detect transient candidates (potential mouth sounds)
        transient_candidates = self._detect_transient_candidates(audio, sr)
        
        # Step 2: Classify and validate mouth sound events
        lip_smack_events = self._classify_mouth_sounds(audio, sr, transient_candidates)
        
        # Step 3: Extract spectral templates for each sound type
        spectral_templates = self._extract_spectral_templates(audio, sr, lip_smack_events)
        
        # Step 4: Calculate statistics
        stats = self._calculate_lip_smack_statistics(lip_smack_events, len(audio) / sr)
        
        # Step 5: Determine preferred sounds
        preferred_sounds = self._determine_preferred_sounds(lip_smack_events)
        
        # Create pattern
        pattern = LipSmackPattern(
            lip_smack_events=lip_smack_events,
            sound_type_frequency=stats['type_frequency'],
            average_occurrence_rate=stats['occurrence_rate'],
            average_duration=stats['average_duration'],
            average_intensity=stats['average_intensity'],
            intensity_std=stats['intensity_std'],
            preferred_sounds=preferred_sounds,
            spectral_templates=spectral_templates,
            transient_characteristics=stats['transient_characteristics'],
            metadata={
                'audio_duration': len(audio) / sr,
                'num_events_detected': len(lip_smack_events),
                'analysis_sample_rate': sr
            }
        )
        
        quality_score = self._assess_pattern_quality(pattern)
        confidence = self._calculate_confidence(lip_smack_events, len(audio) / sr)
        
        return LipSmackAnalysisResult(
            pattern=pattern,
            quality_score=quality_score,
            confidence=confidence,
            analysis_metadata={
                'candidates_found': len(transient_candidates),
                'events_validated': len(lip_smack_events)
            }
        )

    
    def _detect_transient_candidates(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Tuple[int, int]]:
        """
        Detect candidate transient regions that might be mouth sounds.
        
        Mouth sounds are characterized by:
        - Very short duration (10-150ms)
        - Sharp attack (sudden onset)
        - Specific frequency content
        - Often occur near speech boundaries
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples for candidates
        """
        candidates = []
        
        # Compute onset strength for transient detection
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=sr, hop_length=self.hop_length
        )
        
        # Compute spectral flux for additional transient detection
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
        spectral_flux = np.concatenate([[0], spectral_flux])
        
        # Normalize
        onset_env_norm = onset_env / (np.max(onset_env) + 1e-8)
        spectral_flux_norm = spectral_flux / (np.max(spectral_flux) + 1e-8)
        
        # Combined transient score
        transient_score = onset_env_norm * 0.6 + spectral_flux_norm * 0.4
        
        # Find peaks in transient score
        peaks, properties = find_peaks(
            transient_score,
            height=self.TRANSIENT_THRESHOLD,
            distance=int(0.05 * sr / self.hop_length)  # Min 50ms between peaks
        )
        
        # For each peak, define a region around it
        for peak_frame in peaks:
            # Convert to samples
            peak_sample = peak_frame * self.hop_length
            
            # Define region (typically 10-100ms around peak)
            half_window = int(0.05 * sr)  # 50ms half-window
            start_sample = max(0, peak_sample - half_window)
            end_sample = min(len(audio), peak_sample + half_window)
            
            # Refine boundaries based on energy
            segment = audio[start_sample:end_sample]
            if len(segment) > 0:
                envelope = np.abs(segment)
                envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.002 * sr))
                
                # Find actual start (where energy rises)
                threshold = np.max(envelope_smooth) * 0.1
                above_threshold = envelope_smooth > threshold
                
                if np.any(above_threshold):
                    actual_start = np.argmax(above_threshold)
                    actual_end = len(above_threshold) - np.argmax(above_threshold[::-1])
                    
                    refined_start = start_sample + actual_start
                    refined_end = start_sample + actual_end
                    
                    duration = (refined_end - refined_start) / sr
                    if self.LIP_SMACK_DURATION_MIN <= duration <= self.LIP_SMACK_DURATION_MAX:
                        candidates.append((refined_start, refined_end))
        
        logger.debug(f"Found {len(candidates)} transient candidates")
        return candidates

    
    def _classify_mouth_sounds(
        self,
        audio: np.ndarray,
        sr: int,
        candidates: List[Tuple[int, int]]
    ) -> List[LipSmackEvent]:
        """
        Classify and validate transient candidates as mouth sounds.
        
        Args:
            audio: Audio array
            sr: Sample rate
            candidates: List of candidate regions
            
        Returns:
            List of validated LipSmackEvent objects
        """
        events = []
        
        for start_sample, end_sample in candidates:
            segment = audio[start_sample:min(end_sample, len(audio))]
            
            if len(segment) < int(self.LIP_SMACK_DURATION_MIN * sr):
                continue
            
            # Validate this is a mouth sound (not speech or noise)
            if not self._validate_mouth_sound(segment, sr):
                continue
            
            # Classify the sound type
            sound_type = self._classify_sound_type(segment, sr)
            
            # Calculate characteristics
            intensity = self._calculate_intensity(segment, audio)
            freq_peak = self._find_frequency_peak(segment, sr)
            spectral_profile = self._extract_spectral_profile(segment, sr)
            transient_sharpness = self._calculate_transient_sharpness(segment, sr)
            position_context = self._determine_position_context(audio, sr, start_sample)
            
            event = LipSmackEvent(
                sound_type=sound_type,
                start_time=start_sample / sr,
                end_time=end_sample / sr,
                duration=(end_sample - start_sample) / sr,
                intensity=intensity,
                frequency_peak=freq_peak,
                spectral_profile=spectral_profile,
                position_context=position_context,
                transient_sharpness=transient_sharpness
            )
            
            events.append(event)
        
        logger.debug(f"Validated {len(events)} mouth sound events")
        return events
    
    def _validate_mouth_sound(self, segment: np.ndarray, sr: int) -> bool:
        """
        Validate that a segment is actually a mouth sound.
        
        Args:
            segment: Audio segment
            sr: Sample rate
            
        Returns:
            True if segment is a valid mouth sound
        """
        # Check it's not silence
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 0.005:
            return False
        
        # Check it's not too loud (would be speech)
        if rms > 0.4:
            return False
        
        # Check for transient characteristics (sharp attack)
        envelope = np.abs(segment)
        envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.001 * sr))
        
        if len(envelope_smooth) < 10:
            return False
        
        # Find peak position
        peak_idx = np.argmax(envelope_smooth)
        peak_position = peak_idx / len(envelope_smooth)
        
        # Mouth sounds typically have peak in first half (sharp attack)
        if peak_position > 0.7:
            return False
        
        # Check spectral characteristics - mouth sounds have specific frequency content
        stft = librosa.stft(segment, n_fft=min(self.n_fft, len(segment)))
        magnitude = np.abs(stft)
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Mouth sounds should have energy in mid-high frequencies
        freqs = librosa.fft_frequencies(sr=sr, n_fft=min(self.n_fft, len(segment)))
        mid_high_mask = (freqs >= 500) & (freqs <= 6000)
        
        if len(avg_spectrum) > 0 and np.sum(mid_high_mask) > 0:
            mid_high_energy = np.sum(avg_spectrum[mid_high_mask])
            total_energy = np.sum(avg_spectrum) + 1e-8
            mid_high_ratio = mid_high_energy / total_energy
            
            # Mouth sounds should have significant mid-high frequency content
            if mid_high_ratio < 0.2:
                return False
        
        return True

    
    def _classify_sound_type(self, segment: np.ndarray, sr: int) -> LipSmackType:
        """
        Classify the type of mouth sound based on spectral characteristics.
        
        Args:
            segment: Audio segment
            sr: Sample rate
            
        Returns:
            LipSmackType classification
        """
        # Compute spectrum
        stft = librosa.stft(segment, n_fft=min(self.n_fft, len(segment)))
        magnitude = np.abs(stft)
        avg_spectrum = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=min(self.n_fft, len(segment)))
        
        # Find dominant frequency
        if len(avg_spectrum) > 0:
            dominant_idx = np.argmax(avg_spectrum)
            dominant_freq = freqs[dominant_idx] if dominant_idx < len(freqs) else 1000
        else:
            dominant_freq = 1000
        
        # Calculate spectral centroid
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        avg_centroid = np.mean(centroid) if centroid.size > 0 else 2000
        
        # Calculate spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        avg_bandwidth = np.mean(bandwidth) if bandwidth.size > 0 else 1000
        
        # Classify based on frequency characteristics
        scores = {}
        
        for sound_type, freq_range in self.SOUND_FREQ_RANGES.items():
            freq_min, freq_max = freq_range
            
            # Score based on dominant frequency
            if freq_min <= dominant_freq <= freq_max:
                freq_score = 1.0
            else:
                dist = min(abs(dominant_freq - freq_min), abs(dominant_freq - freq_max))
                freq_score = max(0, 1 - dist / 2000)
            
            # Score based on centroid
            center = (freq_min + freq_max) / 2
            centroid_score = max(0, 1 - abs(avg_centroid - center) / 2000)
            
            scores[sound_type] = freq_score * 0.6 + centroid_score * 0.4
        
        # Additional heuristics
        duration = len(segment) / sr
        
        # Very short sounds are likely clicks
        if duration < 0.03:
            scores[LipSmackType.TONGUE_CLICK] *= 1.3
            scores[LipSmackType.TEETH_CLICK] *= 1.2
        
        # Slightly longer sounds are likely lip smacks
        if 0.03 <= duration <= 0.08:
            scores[LipSmackType.LIP_SMACK] *= 1.2
            scores[LipSmackType.LIP_POP] *= 1.2
        
        # Longer sounds might be saliva or swallow
        if duration > 0.08:
            scores[LipSmackType.SALIVA_SOUND] *= 1.2
            scores[LipSmackType.SWALLOW] *= 1.1
        
        # High bandwidth suggests clicks
        if avg_bandwidth > 2000:
            scores[LipSmackType.TONGUE_CLICK] *= 1.1
            scores[LipSmackType.TEETH_CLICK] *= 1.1
        
        # Return highest scoring type
        return max(scores, key=scores.get)

    
    def _calculate_intensity(self, segment: np.ndarray, full_audio: np.ndarray) -> float:
        """Calculate intensity relative to full audio."""
        segment_rms = np.sqrt(np.mean(segment ** 2))
        audio_rms = np.sqrt(np.mean(full_audio ** 2))
        
        if audio_rms > 0:
            return float(np.clip(segment_rms / audio_rms, 0, 1))
        return 0.0
    
    def _find_frequency_peak(self, segment: np.ndarray, sr: int) -> float:
        """Find the dominant frequency of the sound."""
        stft = librosa.stft(segment, n_fft=min(self.n_fft, len(segment)))
        magnitude = np.abs(stft)
        avg_spectrum = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=min(self.n_fft, len(segment)))
        
        if len(avg_spectrum) > 0 and len(freqs) > 0:
            peak_idx = np.argmax(avg_spectrum)
            return float(freqs[min(peak_idx, len(freqs) - 1)])
        return 1000.0
    
    def _extract_spectral_profile(self, segment: np.ndarray, sr: int) -> np.ndarray:
        """Extract normalized spectral profile."""
        stft = librosa.stft(segment, n_fft=min(self.n_fft, len(segment)))
        magnitude = np.abs(stft)
        avg_spectrum = np.mean(magnitude, axis=1)
        return avg_spectrum / (np.max(avg_spectrum) + 1e-8)
    
    def _calculate_transient_sharpness(self, segment: np.ndarray, sr: int) -> float:
        """Calculate how sharp/sudden the transient attack is."""
        envelope = np.abs(segment)
        envelope_smooth = gaussian_filter1d(envelope, sigma=int(0.001 * sr))
        
        if len(envelope_smooth) < 5:
            return 0.5
        
        # Find peak
        peak_idx = np.argmax(envelope_smooth)
        
        if peak_idx == 0:
            return 1.0  # Instant attack
        
        # Calculate attack slope
        attack_portion = envelope_smooth[:peak_idx + 1]
        if len(attack_portion) > 1:
            attack_slope = (attack_portion[-1] - attack_portion[0]) / len(attack_portion)
            # Normalize to 0-1 range
            sharpness = np.clip(attack_slope * sr / 10, 0, 1)
            return float(sharpness)
        
        return 0.5
    
    def _determine_position_context(
        self,
        audio: np.ndarray,
        sr: int,
        start_sample: int
    ) -> str:
        """Determine the context where the mouth sound occurs."""
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Convert to frame
        frame = start_sample // self.hop_length
        context_frames = int(0.2 * sr / self.hop_length)  # 200ms context
        
        # Check speech before
        before_start = max(0, frame - context_frames)
        before_rms = rms[before_start:frame]
        speech_before = np.mean(before_rms) > 0.05 if len(before_rms) > 0 else False
        
        # Check speech after
        after_end = min(len(rms), frame + context_frames)
        after_rms = rms[frame:after_end]
        speech_after = np.mean(after_rms) > 0.05 if len(after_rms) > 0 else False
        
        if not speech_before and speech_after:
            return 'before_speech'
        elif speech_before and not speech_after:
            return 'after_pause'
        else:
            return 'mid_speech'

    
    def _extract_spectral_templates(
        self,
        audio: np.ndarray,
        sr: int,
        events: List[LipSmackEvent]
    ) -> Dict[LipSmackType, np.ndarray]:
        """Extract average spectral templates for each sound type."""
        templates = {}
        
        for sound_type in LipSmackType:
            type_events = [e for e in events if e.sound_type == sound_type]
            
            if type_events:
                profiles = [e.spectral_profile for e in type_events]
                # Ensure all profiles have same length
                max_len = max(len(p) for p in profiles)
                padded_profiles = [
                    np.pad(p, (0, max_len - len(p))) if len(p) < max_len else p[:max_len]
                    for p in profiles
                ]
                avg_profile = np.mean(padded_profiles, axis=0)
                templates[sound_type] = avg_profile / (np.max(avg_profile) + 1e-8)
            else:
                # Create default template based on frequency range
                freq_range = self.SOUND_FREQ_RANGES.get(sound_type, (1000, 3000))
                freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
                center = (freq_range[0] + freq_range[1]) / 2
                width = (freq_range[1] - freq_range[0]) / 2
                template = np.exp(-((freqs - center) ** 2) / (2 * width ** 2))
                templates[sound_type] = template / (np.max(template) + 1e-8)
        
        return templates
    
    def _calculate_lip_smack_statistics(
        self,
        events: List[LipSmackEvent],
        audio_duration: float
    ) -> Dict[str, Any]:
        """Calculate statistics about lip smack patterns."""
        if not events:
            return {
                'type_frequency': {t: 0.0 for t in LipSmackType},
                'occurrence_rate': 0.0,
                'average_duration': 0.05,
                'average_intensity': 0.1,
                'intensity_std': 0.05,
                'transient_characteristics': {
                    'average_sharpness': 0.5,
                    'sharpness_std': 0.2
                }
            }
        
        # Type frequency
        type_counts = {t: 0 for t in LipSmackType}
        for event in events:
            type_counts[event.sound_type] += 1
        
        total = len(events)
        type_frequency = {t: count / total for t, count in type_counts.items()}
        
        # Occurrence rate (per minute)
        occurrence_rate = (len(events) / audio_duration) * 60 if audio_duration > 0 else 0
        
        # Duration statistics
        durations = [e.duration for e in events]
        average_duration = np.mean(durations)
        
        # Intensity statistics
        intensities = [e.intensity for e in events]
        average_intensity = np.mean(intensities)
        intensity_std = np.std(intensities)
        
        # Transient characteristics
        sharpnesses = [e.transient_sharpness for e in events]
        
        return {
            'type_frequency': type_frequency,
            'occurrence_rate': float(occurrence_rate),
            'average_duration': float(average_duration),
            'average_intensity': float(average_intensity),
            'intensity_std': float(intensity_std),
            'transient_characteristics': {
                'average_sharpness': float(np.mean(sharpnesses)),
                'sharpness_std': float(np.std(sharpnesses))
            }
        }
    
    def _determine_preferred_sounds(
        self,
        events: List[LipSmackEvent]
    ) -> List[LipSmackType]:
        """Determine the most common sound types for this speaker."""
        if not events:
            return [LipSmackType.LIP_SMACK, LipSmackType.MOUTH_OPEN]
        
        type_counts = {}
        for event in events:
            type_counts[event.sound_type] = type_counts.get(event.sound_type, 0) + 1
        
        # Sort by frequency and return top types
        sorted_types = sorted(type_counts.keys(), key=lambda t: type_counts[t], reverse=True)
        return sorted_types[:3] if len(sorted_types) >= 3 else sorted_types

    
    def _assess_pattern_quality(self, pattern: LipSmackPattern) -> float:
        """Assess the quality of the extracted pattern."""
        score = 0.0
        
        # Check if we have events
        num_events = len(pattern.lip_smack_events)
        if num_events >= 3:
            score += 0.3
        elif num_events >= 1:
            score += 0.15
        
        # Check occurrence rate is reasonable (0-30 per minute is normal)
        if 0 <= pattern.average_occurrence_rate <= 30:
            score += 0.2
        elif pattern.average_occurrence_rate <= 50:
            score += 0.1
        
        # Check duration is reasonable
        if 0.01 <= pattern.average_duration <= 0.15:
            score += 0.2
        
        # Check intensity consistency
        if pattern.intensity_std < pattern.average_intensity * 0.5:
            score += 0.15
        
        # Check we have variety in sound types
        active_types = sum(1 for f in pattern.sound_type_frequency.values() if f > 0)
        if active_types >= 2:
            score += 0.15
        
        return float(np.clip(score, 0, 1))
    
    def _calculate_confidence(
        self,
        events: List[LipSmackEvent],
        audio_duration: float
    ) -> float:
        """Calculate confidence in the analysis."""
        if audio_duration < 3.0:
            return 0.3
        
        # Confidence based on detection rate
        expected_events = audio_duration / 5.0  # Expect ~1 event per 5 seconds
        actual_events = len(events)
        
        if expected_events > 0:
            detection_ratio = actual_events / expected_events
            if 0.2 <= detection_ratio <= 3.0:
                confidence = 0.7
            elif 0.1 <= detection_ratio <= 5.0:
                confidence = 0.5
            else:
                confidence = 0.3
        else:
            confidence = 0.5
        
        # Boost for longer audio
        if audio_duration > 10:
            confidence = min(1.0, confidence + 0.1)
        
        return float(confidence)



class LipSmackInjector:
    """
    Injects lip smack and mouth sounds into synthesized speech.
    
    This class uses analyzed mouth sound patterns from reference audio to inject
    realistic lip smacks and mouth sounds that match the speaker's natural style.
    
    Key features:
    - Detects appropriate insertion points (before speech, at pauses)
    - Generates mouth sounds matching reference spectral profile
    - Applies proper timing and intensity
    - Ensures natural spacing of sounds
    
    Validates: Requirements 5.2 (preserve lip smacks and natural speech artifacts)
    """
    
    # Timing parameters
    MIN_INTERVAL_BETWEEN_SOUNDS = 1.0  # Minimum seconds between injected sounds
    IDEAL_INTERVAL = 5.0  # Ideal interval between sounds
    
    # Insertion preferences
    BEFORE_SPEECH_PROBABILITY = 0.4  # Probability of inserting before speech
    AFTER_PAUSE_PROBABILITY = 0.3  # Probability of inserting after pause
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize lip smack injector."""
        self.sample_rate = sample_rate
        self.hop_length = 256
        self.n_fft = 1024
        self.analyzer = LipSmackPatternAnalyzer(sample_rate)
        logger.info("Lip Smack Injector initialized")
    
    def inject_lip_smacks(
        self,
        synthesized_audio: np.ndarray,
        lip_smack_pattern: LipSmackPattern,
        sample_rate: Optional[int] = None,
        intensity_scale: float = 1.0,
        injection_probability: float = 0.3,
        min_sounds: int = 0,
        max_sounds: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Inject lip smack and mouth sounds into synthesized audio.
        
        Args:
            synthesized_audio: The synthesized speech audio array
            lip_smack_pattern: Analyzed pattern from reference audio
            sample_rate: Sample rate (uses default if not provided)
            intensity_scale: Scale factor for sound intensity (0.5-1.5 recommended)
            injection_probability: Probability of injecting at valid points
            min_sounds: Minimum number of sounds to inject
            max_sounds: Maximum number of sounds to inject
            
        Returns:
            Tuple of (audio with sounds injected, list of injection metadata)
            
        Validates: Requirements 5.2 (preserve lip smacks and natural speech artifacts)
        """
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            synthesized_audio = librosa.resample(
                synthesized_audio, orig_sr=sr, target_sr=self.sample_rate
            )
            sr = self.sample_rate
        
        audio_duration = len(synthesized_audio) / sr
        logger.info(f"Injecting lip smacks into {audio_duration:.2f}s synthesized audio")
        
        # Step 1: Find appropriate insertion points
        insertion_points = self._find_insertion_points(synthesized_audio, sr)
        
        if not insertion_points:
            logger.warning("No suitable lip smack insertion points found")
            return synthesized_audio, []
        
        # Step 2: Select which points to use
        selected_points = self._select_injection_points(
            insertion_points,
            lip_smack_pattern,
            audio_duration,
            injection_probability,
            min_sounds,
            max_sounds
        )
        
        if not selected_points:
            logger.warning("No lip smack points selected after filtering")
            return synthesized_audio, []
        
        # Step 3: Generate mouth sounds for each point
        mouth_sounds = self._generate_mouth_sounds(
            selected_points,
            lip_smack_pattern,
            sr,
            intensity_scale
        )
        
        # Step 4: Inject sounds into audio
        result_audio, injection_metadata = self._inject_sounds_into_audio(
            synthesized_audio,
            mouth_sounds,
            selected_points,
            sr
        )
        
        logger.info(f"Injected {len(injection_metadata)} lip smacks/mouth sounds")
        
        return result_audio, injection_metadata

    
    def _find_insertion_points(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[Dict[str, Any]]:
        """
        Find appropriate points to insert lip smacks and mouth sounds.
        
        Mouth sounds typically occur:
        - Just before speech starts
        - At phrase boundaries
        - After short pauses
        
        Args:
            audio: Synthesized audio array
            sr: Sample rate
            
        Returns:
            List of insertion point dictionaries
        """
        insertion_points = []
        
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=1024,
            hop_length=self.hop_length
        )[0]
        
        # Smooth RMS
        rms_smooth = gaussian_filter1d(rms, sigma=3)
        
        # Detect speech/silence transitions
        silence_threshold = np.percentile(rms_smooth, 30)
        speech_threshold = np.percentile(rms_smooth, 60)
        
        silence_mask = rms_smooth < silence_threshold
        speech_mask = rms_smooth > speech_threshold
        
        # Find transitions from silence to speech (good for mouth opening sounds)
        for i in range(1, len(silence_mask)):
            if silence_mask[i-1] and speech_mask[i]:
                # Transition from silence to speech
                time = i * self.hop_length / sr
                
                # Check there's enough silence before
                lookback = min(i, int(0.1 * sr / self.hop_length))
                if np.mean(silence_mask[i-lookback:i]) > 0.7:
                    insertion_points.append({
                        'time': time,
                        'context': 'before_speech',
                        'suitability_score': 0.8
                    })
        
        # Find short pauses within speech (good for lip smacks)
        in_pause = False
        pause_start = 0
        
        for i in range(len(silence_mask)):
            if silence_mask[i] and not in_pause:
                pause_start = i
                in_pause = True
            elif not silence_mask[i] and in_pause:
                pause_end = i
                in_pause = False
                
                pause_duration = (pause_end - pause_start) * self.hop_length / sr
                
                # Short pauses (100-500ms) are good for lip smacks
                if 0.1 <= pause_duration <= 0.5:
                    time = (pause_start + pause_end) / 2 * self.hop_length / sr
                    insertion_points.append({
                        'time': time,
                        'context': 'mid_pause',
                        'suitability_score': 0.6
                    })
        
        # Add points at the very beginning if there's speech soon after
        if len(speech_mask) > 0 and np.any(speech_mask[:int(0.5 * sr / self.hop_length)]):
            first_speech = np.argmax(speech_mask)
            if first_speech > int(0.05 * sr / self.hop_length):
                insertion_points.append({
                    'time': 0.02,  # Very beginning
                    'context': 'audio_start',
                    'suitability_score': 0.7
                })
        
        # Sort by time
        insertion_points.sort(key=lambda x: x['time'])
        
        logger.debug(f"Found {len(insertion_points)} potential lip smack insertion points")
        return insertion_points

    
    def _select_injection_points(
        self,
        insertion_points: List[Dict[str, Any]],
        pattern: LipSmackPattern,
        audio_duration: float,
        injection_probability: float,
        min_sounds: int,
        max_sounds: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Select which insertion points to use based on pattern and constraints.
        
        Args:
            insertion_points: All potential insertion points
            pattern: Reference lip smack pattern
            audio_duration: Total audio duration
            injection_probability: Probability of using each point
            min_sounds: Minimum sounds to inject
            max_sounds: Maximum sounds to inject
            
        Returns:
            Selected insertion points
        """
        if not insertion_points:
            return []
        
        # Calculate target number based on reference pattern
        if pattern.average_occurrence_rate > 0:
            target_count = int((pattern.average_occurrence_rate / 60) * audio_duration)
        else:
            target_count = int(audio_duration / self.IDEAL_INTERVAL)
        
        # Apply constraints
        if max_sounds is not None:
            target_count = min(target_count, max_sounds)
        target_count = max(target_count, min_sounds)
        target_count = min(target_count, len(insertion_points))
        
        if target_count == 0:
            return []
        
        # Select points with probability weighting
        selected = []
        last_time = -self.MIN_INTERVAL_BETWEEN_SOUNDS
        
        # Sort by suitability score
        sorted_points = sorted(
            insertion_points,
            key=lambda x: x['suitability_score'],
            reverse=True
        )
        
        for point in sorted_points:
            if len(selected) >= target_count:
                break
            
            # Check minimum interval
            if point['time'] - last_time < self.MIN_INTERVAL_BETWEEN_SOUNDS:
                continue
            
            # Apply probability
            if np.random.random() < injection_probability:
                selected.append(point)
                last_time = point['time']
        
        # If we haven't met minimum, try to add more
        if len(selected) < min_sounds:
            remaining = [p for p in sorted_points if p not in selected]
            for point in remaining:
                if len(selected) >= min_sounds:
                    break
                
                # Relax interval constraint
                times = [s['time'] for s in selected]
                min_dist = min([abs(point['time'] - t) for t in times]) if times else float('inf')
                
                if min_dist >= self.MIN_INTERVAL_BETWEEN_SOUNDS * 0.5:
                    selected.append(point)
        
        # Sort by time
        selected.sort(key=lambda x: x['time'])
        
        logger.debug(f"Selected {len(selected)} lip smack points")
        return selected

    
    def _generate_mouth_sounds(
        self,
        insertion_points: List[Dict[str, Any]],
        pattern: LipSmackPattern,
        sr: int,
        intensity_scale: float
    ) -> List[np.ndarray]:
        """
        Generate mouth sounds for each insertion point.
        
        Args:
            insertion_points: Selected insertion points
            pattern: Reference lip smack pattern
            sr: Sample rate
            intensity_scale: Intensity scaling factor
            
        Returns:
            List of mouth sound arrays
        """
        mouth_sounds = []
        
        for point in insertion_points:
            # Select sound type based on context and pattern preferences
            sound_type = self._select_sound_type(point['context'], pattern)
            
            # Calculate duration (typically 20-80ms)
            base_duration = pattern.average_duration if pattern.average_duration > 0 else 0.05
            duration = base_duration * (0.8 + np.random.random() * 0.4)  # ±20% variation
            duration = np.clip(duration, 0.02, 0.1)
            
            # Calculate intensity
            base_intensity = pattern.average_intensity if pattern.average_intensity > 0 else 0.1
            intensity = base_intensity * intensity_scale
            intensity *= (0.8 + np.random.random() * 0.4)  # ±20% variation
            intensity = np.clip(intensity, 0.02, 0.2)
            
            # Get spectral template
            spectral_template = pattern.spectral_templates.get(
                sound_type,
                self._create_default_template(sound_type, sr)
            )
            
            # Generate the sound
            sound = self._synthesize_mouth_sound(
                sound_type,
                duration,
                intensity,
                spectral_template,
                pattern.transient_characteristics,
                sr
            )
            
            mouth_sounds.append(sound)
        
        return mouth_sounds
    
    def _select_sound_type(
        self,
        context: str,
        pattern: LipSmackPattern
    ) -> LipSmackType:
        """Select appropriate sound type based on context."""
        # Context-based preferences
        if context == 'before_speech' or context == 'audio_start':
            # Mouth opening sounds before speech
            preferred = [LipSmackType.MOUTH_OPEN, LipSmackType.LIP_SMACK]
        elif context == 'mid_pause':
            # Lip smacks and clicks during pauses
            preferred = [LipSmackType.LIP_SMACK, LipSmackType.TONGUE_CLICK, LipSmackType.SALIVA_SOUND]
        else:
            preferred = [LipSmackType.LIP_SMACK]
        
        # Combine with pattern preferences
        if pattern.preferred_sounds:
            # Weight towards pattern preferences
            combined = []
            for sound in pattern.preferred_sounds:
                if sound in preferred:
                    combined.extend([sound] * 3)  # Higher weight
                else:
                    combined.append(sound)
            for sound in preferred:
                if sound not in combined:
                    combined.append(sound)
            
            return np.random.choice(combined)
        
        return np.random.choice(preferred)
    
    def _create_default_template(self, sound_type: LipSmackType, sr: int) -> np.ndarray:
        """Create a default spectral template for a sound type."""
        freq_range = LipSmackPatternAnalyzer.SOUND_FREQ_RANGES.get(
            sound_type, (1000, 3000)
        )
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        center = (freq_range[0] + freq_range[1]) / 2
        width = (freq_range[1] - freq_range[0]) / 2
        template = np.exp(-((freqs - center) ** 2) / (2 * width ** 2))
        return template / (np.max(template) + 1e-8)

    
    def _synthesize_mouth_sound(
        self,
        sound_type: LipSmackType,
        duration: float,
        intensity: float,
        spectral_template: np.ndarray,
        transient_characteristics: Dict[str, float],
        sr: int
    ) -> np.ndarray:
        """
        Synthesize a mouth sound matching the reference characteristics.
        
        Args:
            sound_type: Type of sound to generate
            duration: Target duration in seconds
            intensity: Target intensity
            spectral_template: Spectral profile to match
            transient_characteristics: Attack/decay characteristics
            sr: Sample rate
            
        Returns:
            Synthesized mouth sound array
        """
        num_samples = int(duration * sr)
        
        # Generate noise base
        noise = np.random.randn(num_samples)
        
        # Apply spectral shaping
        stft = librosa.stft(noise, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Ensure template matches STFT frequency bins
        if len(spectral_template) != stft.shape[0]:
            spectral_template = np.interp(
                np.linspace(0, 1, stft.shape[0]),
                np.linspace(0, 1, len(spectral_template)),
                spectral_template
            )
        
        # Apply spectral shaping
        shaped_stft = stft * spectral_template[:, np.newaxis]
        
        # Convert back to time domain
        shaped_noise = librosa.istft(shaped_stft, hop_length=self.hop_length, length=num_samples)
        
        # Create envelope based on sound type and transient characteristics
        envelope = self._create_mouth_sound_envelope(
            sound_type,
            num_samples,
            transient_characteristics,
            sr
        )
        
        # Apply envelope and intensity
        sound = shaped_noise * envelope * intensity
        
        # Apply highpass filter to remove low frequency rumble
        nyquist = sr / 2
        cutoff = 200 / nyquist
        if cutoff < 1:
            b, a = butter(2, cutoff, btype='high')
            sound = filtfilt(b, a, sound)
        
        # Normalize to target intensity
        current_rms = np.sqrt(np.mean(sound ** 2))
        if current_rms > 0:
            sound = sound * (intensity / current_rms)
        
        return sound
    
    def _create_mouth_sound_envelope(
        self,
        sound_type: LipSmackType,
        num_samples: int,
        transient_characteristics: Dict[str, float],
        sr: int
    ) -> np.ndarray:
        """
        Create an envelope for the mouth sound based on its type.
        
        Args:
            sound_type: Type of mouth sound
            num_samples: Number of samples
            transient_characteristics: Attack/decay characteristics
            sr: Sample rate
            
        Returns:
            Envelope array
        """
        envelope = np.ones(num_samples)
        
        # Get sharpness from characteristics
        sharpness = transient_characteristics.get('average_sharpness', 0.7)
        
        # Define attack and decay based on sound type
        if sound_type in [LipSmackType.LIP_SMACK, LipSmackType.LIP_POP]:
            # Sharp attack, moderate decay
            attack_ratio = 0.1 * (1 - sharpness * 0.5)
            decay_ratio = 0.6
        elif sound_type in [LipSmackType.TONGUE_CLICK, LipSmackType.TEETH_CLICK]:
            # Very sharp attack, quick decay
            attack_ratio = 0.05
            decay_ratio = 0.7
        elif sound_type == LipSmackType.MOUTH_OPEN:
            # Gradual attack, gradual decay
            attack_ratio = 0.3
            decay_ratio = 0.4
        elif sound_type == LipSmackType.SALIVA_SOUND:
            # Moderate attack, longer sustain
            attack_ratio = 0.2
            decay_ratio = 0.5
        else:
            # Default
            attack_ratio = 0.15
            decay_ratio = 0.5
        
        attack_samples = int(attack_ratio * num_samples)
        decay_samples = int(decay_ratio * num_samples)
        
        # Ensure we don't exceed array bounds
        attack_samples = min(attack_samples, num_samples // 3)
        decay_samples = min(decay_samples, num_samples - attack_samples)
        
        # Apply attack (exponential rise for sharpness)
        if attack_samples > 0:
            attack_curve = np.linspace(0, 1, attack_samples) ** (1 / (sharpness + 0.1))
            envelope[:attack_samples] = attack_curve
        
        # Apply decay (exponential fall)
        if decay_samples > 0:
            decay_curve = np.linspace(1, 0, decay_samples) ** 0.5
            envelope[-decay_samples:] = decay_curve
        
        return envelope

    
    def _inject_sounds_into_audio(
        self,
        audio: np.ndarray,
        mouth_sounds: List[np.ndarray],
        insertion_points: List[Dict[str, Any]],
        sr: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Inject mouth sounds into the audio with proper crossfading.
        
        Args:
            audio: Original synthesized audio
            mouth_sounds: List of mouth sounds to inject
            insertion_points: Insertion point metadata
            sr: Sample rate
            
        Returns:
            Tuple of (audio with sounds, injection metadata)
        """
        result = audio.copy()
        injection_metadata = []
        
        # Very short crossfade for transient sounds
        crossfade_samples = int(0.005 * sr)  # 5ms crossfade
        
        for sound, point in zip(mouth_sounds, insertion_points):
            insertion_sample = int(point['time'] * sr)
            sound_length = len(sound)
            
            # Calculate position (insert just before the time point)
            start_pos = max(0, insertion_sample - sound_length // 2)
            end_pos = start_pos + sound_length
            
            # Ensure within bounds
            if end_pos > len(result):
                sound = sound[:len(result) - start_pos]
                end_pos = len(result)
            
            if len(sound) < crossfade_samples * 2:
                continue
            
            # Create blend envelope
            blend_envelope = np.ones(len(sound))
            
            # Fade in
            if crossfade_samples > 0 and crossfade_samples < len(blend_envelope):
                blend_envelope[:crossfade_samples] = np.linspace(0, 1, crossfade_samples)
            
            # Fade out
            if crossfade_samples > 0 and crossfade_samples < len(blend_envelope):
                blend_envelope[-crossfade_samples:] = np.linspace(1, 0, crossfade_samples)
            
            # Add sound to existing audio (additive mixing for transients)
            existing_segment = result[start_pos:start_pos + len(sound)]
            blended = existing_segment + sound * blend_envelope * 0.8
            
            # Soft clip to prevent distortion
            blended = np.tanh(blended)
            
            result[start_pos:start_pos + len(sound)] = blended
            
            injection_metadata.append({
                'time': point['time'],
                'duration': len(sound) / sr,
                'context': point['context'],
                'start_sample': start_pos,
                'end_sample': start_pos + len(sound)
            })
        
        return result, injection_metadata
    
    def inject_lip_smacks_from_reference(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: np.ndarray,
        synthesized_sr: Optional[int] = None,
        reference_sr: Optional[int] = None,
        intensity_scale: float = 1.0,
        injection_probability: float = 0.3
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Convenience method to analyze reference and inject lip smacks in one call.
        
        Args:
            synthesized_audio: Synthesized speech to enhance
            reference_audio: Reference audio to analyze for mouth sound patterns
            synthesized_sr: Sample rate of synthesized audio
            reference_sr: Sample rate of reference audio
            intensity_scale: Sound intensity scaling factor
            injection_probability: Probability of inserting at valid points
            
        Returns:
            Tuple of (enhanced audio, injection metadata)
            
        Validates: Requirements 5.2 (preserve lip smacks and natural speech artifacts)
        """
        synth_sr = synthesized_sr or self.sample_rate
        ref_sr = reference_sr or self.sample_rate
        
        # Analyze lip smack patterns from reference
        analysis_result = self.analyzer.analyze_lip_smack_patterns(
            reference_audio, ref_sr
        )
        
        logger.info(
            f"Analyzed reference: {len(analysis_result.pattern.lip_smack_events)} mouth sounds, "
            f"rate={analysis_result.pattern.average_occurrence_rate:.1f}/min, "
            f"quality={analysis_result.quality_score:.2f}"
        )
        
        # Inject lip smacks into synthesized audio
        return self.inject_lip_smacks(
            synthesized_audio,
            analysis_result.pattern,
            synth_sr,
            intensity_scale,
            injection_probability
        )



# Global instances for lip smack analysis and injection
_lip_smack_analyzer: Optional[LipSmackPatternAnalyzer] = None
_lip_smack_injector: Optional[LipSmackInjector] = None


def get_lip_smack_analyzer() -> LipSmackPatternAnalyzer:
    """Get or create global lip smack pattern analyzer instance."""
    global _lip_smack_analyzer
    if _lip_smack_analyzer is None:
        _lip_smack_analyzer = LipSmackPatternAnalyzer()
    return _lip_smack_analyzer


def get_lip_smack_injector() -> LipSmackInjector:
    """Get or create global lip smack injector instance."""
    global _lip_smack_injector
    if _lip_smack_injector is None:
        _lip_smack_injector = LipSmackInjector()
    return _lip_smack_injector


# Convenience function for complete micro-expression injection
def inject_all_micro_expressions(
    synthesized_audio: np.ndarray,
    reference_audio: np.ndarray,
    synthesized_sr: int = 22050,
    reference_sr: int = 22050,
    include_breathing: bool = True,
    include_hesitations: bool = True,
    include_lip_smacks: bool = True,
    breathing_intensity: float = 1.0,
    hesitation_probability: float = 0.2,
    lip_smack_probability: float = 0.3
) -> Tuple[np.ndarray, Dict[str, List[Dict[str, Any]]]]:
    """
    Inject all types of micro-expressions into synthesized audio.
    
    This is a convenience function that applies breathing, hesitations,
    and lip smacks in sequence to create maximally natural-sounding speech.
    
    Args:
        synthesized_audio: Synthesized speech to enhance
        reference_audio: Reference audio to analyze for patterns
        synthesized_sr: Sample rate of synthesized audio
        reference_sr: Sample rate of reference audio
        include_breathing: Whether to inject breathing sounds
        include_hesitations: Whether to inject hesitation sounds
        include_lip_smacks: Whether to inject lip smacks and mouth sounds
        breathing_intensity: Intensity scale for breathing (0.5-1.5)
        hesitation_probability: Probability of hesitation injection
        lip_smack_probability: Probability of lip smack injection
        
    Returns:
        Tuple of (enhanced audio, dict of injection metadata by type)
        
    Validates: Requirements 5.1, 5.2 (natural micro-expressions)
    """
    result_audio = synthesized_audio.copy()
    all_metadata = {}
    
    # Inject breathing first (most subtle)
    if include_breathing:
        breathing_injector = get_breathing_injector()
        result_audio, breathing_metadata = breathing_injector.inject_breathing_from_reference(
            result_audio,
            reference_audio,
            synthesized_sr,
            reference_sr,
            breathing_intensity
        )
        all_metadata['breathing'] = breathing_metadata
        logger.info(f"Injected {len(breathing_metadata)} breathing sounds")
    
    # Inject lip smacks (subtle transients)
    if include_lip_smacks:
        lip_smack_injector = get_lip_smack_injector()
        result_audio, lip_smack_metadata = lip_smack_injector.inject_lip_smacks_from_reference(
            result_audio,
            reference_audio,
            synthesized_sr,
            reference_sr,
            intensity_scale=1.0,
            injection_probability=lip_smack_probability
        )
        all_metadata['lip_smacks'] = lip_smack_metadata
        logger.info(f"Injected {len(lip_smack_metadata)} lip smacks/mouth sounds")
    
    # Inject hesitations last (most noticeable)
    if include_hesitations:
        hesitation_injector = get_hesitation_injector()
        result_audio, hesitation_metadata = hesitation_injector.inject_hesitations_from_reference(
            result_audio,
            reference_audio,
            synthesized_sr,
            reference_sr,
            intensity_scale=1.0,
            hesitation_probability=hesitation_probability
        )
        all_metadata['hesitations'] = hesitation_metadata
        logger.info(f"Injected {len(hesitation_metadata)} hesitations")
    
    return result_audio, all_metadata


# =============================================================================
# COARTICULATION SMOOTHING
# =============================================================================

@dataclass
class CoarticulationRegion:
    """Represents a region where coarticulation smoothing should be applied."""
    start_time: float
    end_time: float
    start_sample: int
    end_sample: int
    transition_type: str  # 'phoneme_boundary', 'word_boundary', 'syllable_boundary'
    smoothing_strength: float  # 0-1, how much smoothing to apply
    spectral_discontinuity: float  # Measure of spectral change at boundary
    formant_shift: Dict[str, float]  # F1, F2, F3 shifts across boundary


@dataclass
class CoarticulationAnalysis:
    """Result of coarticulation analysis on audio."""
    regions: List[CoarticulationRegion]
    average_discontinuity: float
    smoothing_applied: bool
    quality_improvement: float
    metadata: Dict[str, Any]


class CoarticulationSmoother:
    """
    Implements coarticulation smoothing for natural phoneme transitions.
    
    Coarticulation is the phenomenon where adjacent speech sounds influence
    each other, creating smooth transitions between phonemes. This class:
    1. Detects phoneme/syllable boundaries in synthesized speech
    2. Analyzes spectral discontinuities at boundaries
    3. Applies smoothing to create natural transitions
    4. Preserves voice characteristics while improving naturalness
    
    Validates: Requirements 5.4 (smooth coarticulation between phonemes)
    """
    
    # Analysis parameters
    FRAME_LENGTH = 2048
    HOP_LENGTH = 256  # Smaller hop for finer resolution
    N_FFT = 2048
    
    # Boundary detection thresholds
    SPECTRAL_FLUX_THRESHOLD = 0.3  # Threshold for detecting spectral changes
    ENERGY_CHANGE_THRESHOLD = 0.2  # Threshold for energy-based boundary detection
    MIN_BOUNDARY_INTERVAL = 0.03  # Minimum 30ms between boundaries
    
    # Smoothing parameters
    DEFAULT_SMOOTHING_WINDOW = 0.015  # 15ms smoothing window
    MAX_SMOOTHING_WINDOW = 0.030  # Maximum 30ms smoothing
    FORMANT_SMOOTHING_FACTOR = 0.7  # How much to smooth formant transitions
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the coarticulation smoother.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        logger.info("Coarticulation Smoother initialized")

    
    def smooth_coarticulation(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        smoothing_strength: float = 0.5,
        preserve_formants: bool = True
    ) -> Tuple[np.ndarray, CoarticulationAnalysis]:
        """
        Apply coarticulation smoothing to synthesized audio.
        
        This method detects phoneme boundaries and applies spectral smoothing
        to create natural transitions between sounds, mimicking how humans
        naturally blend phonemes together.
        
        Args:
            audio: Synthesized audio array
            sample_rate: Sample rate (uses default if not provided)
            smoothing_strength: How aggressively to smooth (0-1)
            preserve_formants: Whether to preserve formant structure
            
        Returns:
            Tuple of (smoothed audio, analysis results)
            
        Validates: Requirements 5.4 (smooth coarticulation between phonemes)
        """
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        
        logger.info(f"Applying coarticulation smoothing to {len(audio)/sr:.2f}s audio")
        
        # Step 1: Detect phoneme/syllable boundaries
        boundaries = self._detect_boundaries(audio, sr)
        
        if not boundaries:
            logger.info("No significant boundaries detected, returning original audio")
            return audio, CoarticulationAnalysis(
                regions=[],
                average_discontinuity=0.0,
                smoothing_applied=False,
                quality_improvement=0.0,
                metadata={'reason': 'no_boundaries_detected'}
            )
        
        # Step 2: Analyze each boundary for coarticulation regions
        regions = self._analyze_coarticulation_regions(audio, sr, boundaries)

        
        # Step 3: Apply smoothing to each region
        smoothed_audio = self._apply_smoothing(
            audio, sr, regions, smoothing_strength, preserve_formants
        )
        
        # Step 4: Calculate quality improvement
        quality_improvement = self._calculate_quality_improvement(
            audio, smoothed_audio, sr
        )
        
        # Calculate average discontinuity
        avg_discontinuity = np.mean([r.spectral_discontinuity for r in regions]) if regions else 0.0
        
        analysis = CoarticulationAnalysis(
            regions=regions,
            average_discontinuity=float(avg_discontinuity),
            smoothing_applied=True,
            quality_improvement=float(quality_improvement),
            metadata={
                'num_boundaries': len(boundaries),
                'num_regions_smoothed': len(regions),
                'smoothing_strength': smoothing_strength,
                'preserve_formants': preserve_formants
            }
        )
        
        logger.info(
            f"Coarticulation smoothing complete: {len(regions)} regions smoothed, "
            f"quality improvement: {quality_improvement:.2%}"
        )
        
        return smoothed_audio, analysis
    
    def _detect_boundaries(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[int]:
        """
        Detect phoneme/syllable boundaries using spectral flux and energy changes.
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            List of boundary sample positions
        """
        boundaries = []

        
        # Compute STFT for spectral analysis
        stft = librosa.stft(audio, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH)
        magnitude = np.abs(stft)
        
        # Compute spectral flux (rate of spectral change)
        spectral_flux = np.zeros(magnitude.shape[1])
        for i in range(1, magnitude.shape[1]):
            # Half-wave rectified difference
            diff = magnitude[:, i] - magnitude[:, i-1]
            spectral_flux[i] = np.sum(np.maximum(0, diff))
        
        # Normalize spectral flux
        if np.max(spectral_flux) > 0:
            spectral_flux = spectral_flux / np.max(spectral_flux)
        
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=audio, frame_length=self.FRAME_LENGTH, hop_length=self.HOP_LENGTH
        )[0]
        
        # Compute energy derivative (rate of energy change)
        energy_derivative = np.abs(np.diff(rms, prepend=rms[0]))
        if np.max(energy_derivative) > 0:
            energy_derivative = energy_derivative / np.max(energy_derivative)
        
        # Ensure arrays are same length
        min_len = min(len(spectral_flux), len(energy_derivative))
        spectral_flux = spectral_flux[:min_len]
        energy_derivative = energy_derivative[:min_len]
        
        # Combined boundary detection score
        boundary_score = 0.7 * spectral_flux + 0.3 * energy_derivative
        
        # Smooth the score slightly
        boundary_score = gaussian_filter1d(boundary_score, sigma=2)

        
        # Find peaks in boundary score
        min_distance_frames = int(self.MIN_BOUNDARY_INTERVAL * sr / self.HOP_LENGTH)
        peaks, properties = find_peaks(
            boundary_score,
            height=self.SPECTRAL_FLUX_THRESHOLD,
            distance=max(1, min_distance_frames)
        )
        
        # Convert frame indices to sample positions
        for peak_frame in peaks:
            sample_pos = peak_frame * self.HOP_LENGTH
            if 0 < sample_pos < len(audio):
                boundaries.append(sample_pos)
        
        logger.debug(f"Detected {len(boundaries)} phoneme boundaries")
        return boundaries
    
    def _analyze_coarticulation_regions(
        self,
        audio: np.ndarray,
        sr: int,
        boundaries: List[int]
    ) -> List[CoarticulationRegion]:
        """
        Analyze each boundary to create coarticulation regions.
        
        Args:
            audio: Audio array
            sr: Sample rate
            boundaries: List of boundary sample positions
            
        Returns:
            List of CoarticulationRegion objects
        """
        regions = []
        
        # Window size for analysis around each boundary
        analysis_window = int(self.MAX_SMOOTHING_WINDOW * sr)
        
        for boundary_sample in boundaries:
            # Define region around boundary
            start_sample = max(0, boundary_sample - analysis_window)
            end_sample = min(len(audio), boundary_sample + analysis_window)

            
            # Extract segments before and after boundary
            before_segment = audio[start_sample:boundary_sample]
            after_segment = audio[boundary_sample:end_sample]
            
            if len(before_segment) < 256 or len(after_segment) < 256:
                continue
            
            # Analyze spectral discontinuity
            discontinuity = self._measure_spectral_discontinuity(
                before_segment, after_segment, sr
            )
            
            # Analyze formant shifts
            formant_shift = self._analyze_formant_shift(
                before_segment, after_segment, sr
            )
            
            # Determine transition type based on discontinuity magnitude
            if discontinuity > 0.6:
                transition_type = 'phoneme_boundary'
            elif discontinuity > 0.3:
                transition_type = 'syllable_boundary'
            else:
                transition_type = 'word_boundary'
            
            # Calculate smoothing strength based on discontinuity
            # Higher discontinuity = more smoothing needed
            smoothing_strength = min(1.0, discontinuity * 1.2)
            
            region = CoarticulationRegion(
                start_time=start_sample / sr,
                end_time=end_sample / sr,
                start_sample=start_sample,
                end_sample=end_sample,
                transition_type=transition_type,
                smoothing_strength=smoothing_strength,
                spectral_discontinuity=discontinuity,
                formant_shift=formant_shift
            )
            
            regions.append(region)
        
        return regions

    
    def _measure_spectral_discontinuity(
        self,
        before_segment: np.ndarray,
        after_segment: np.ndarray,
        sr: int
    ) -> float:
        """
        Measure the spectral discontinuity between two segments.
        
        Args:
            before_segment: Audio segment before boundary
            after_segment: Audio segment after boundary
            sr: Sample rate
            
        Returns:
            Discontinuity score (0-1)
        """
        # Compute spectral features for each segment
        # Use the last part of before and first part of after
        analysis_len = min(len(before_segment), len(after_segment), int(0.02 * sr))
        
        before_end = before_segment[-analysis_len:]
        after_start = after_segment[:analysis_len]
        
        # Compute MFCCs for comparison
        before_mfcc = librosa.feature.mfcc(
            y=before_end, sr=sr, n_mfcc=13, n_fft=512, hop_length=128
        )
        after_mfcc = librosa.feature.mfcc(
            y=after_start, sr=sr, n_mfcc=13, n_fft=512, hop_length=128
        )
        
        # Average MFCCs across time
        before_mfcc_mean = np.mean(before_mfcc, axis=1)
        after_mfcc_mean = np.mean(after_mfcc, axis=1)
        
        # Compute cosine distance
        dot_product = np.dot(before_mfcc_mean, after_mfcc_mean)
        norm_before = np.linalg.norm(before_mfcc_mean)
        norm_after = np.linalg.norm(after_mfcc_mean)
        
        if norm_before > 0 and norm_after > 0:
            cosine_similarity = dot_product / (norm_before * norm_after)
            discontinuity = 1.0 - max(0, cosine_similarity)
        else:
            discontinuity = 0.5
        
        return float(np.clip(discontinuity, 0, 1))

    
    def _analyze_formant_shift(
        self,
        before_segment: np.ndarray,
        after_segment: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """
        Analyze formant frequency shifts across a boundary.
        
        Args:
            before_segment: Audio segment before boundary
            after_segment: Audio segment after boundary
            sr: Sample rate
            
        Returns:
            Dictionary with F1, F2, F3 shift values
        """
        formant_shift = {'F1': 0.0, 'F2': 0.0, 'F3': 0.0}
        
        try:
            # Estimate formants using LPC
            before_formants = self._estimate_formants(before_segment, sr)
            after_formants = self._estimate_formants(after_segment, sr)
            
            # Calculate shifts
            for i, key in enumerate(['F1', 'F2', 'F3']):
                if i < len(before_formants) and i < len(after_formants):
                    shift = after_formants[i] - before_formants[i]
                    formant_shift[key] = float(shift)
        except Exception as e:
            logger.debug(f"Formant analysis failed: {e}")
        
        return formant_shift
    
    def _estimate_formants(
        self,
        segment: np.ndarray,
        sr: int,
        num_formants: int = 3
    ) -> List[float]:
        """
        Estimate formant frequencies using LPC analysis.
        
        Args:
            segment: Audio segment
            sr: Sample rate
            num_formants: Number of formants to estimate
            
        Returns:
            List of formant frequencies
        """
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized = np.append(segment[0], segment[1:] - pre_emphasis * segment[:-1])

        
        # Apply window
        windowed = emphasized * np.hamming(len(emphasized))
        
        # LPC order (rule of thumb: 2 + sr/1000)
        lpc_order = min(2 + int(sr / 1000), len(windowed) - 1)
        
        # Compute LPC coefficients using autocorrelation method
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Levinson-Durbin recursion
        try:
            lpc_coeffs = self._levinson_durbin(autocorr, lpc_order)
        except Exception:
            return [500.0, 1500.0, 2500.0]  # Default formants
        
        # Find roots of LPC polynomial
        roots = np.roots(lpc_coeffs)
        
        # Keep only roots inside unit circle with positive imaginary part
        roots = roots[np.abs(roots) < 1]
        roots = roots[np.imag(roots) > 0]
        
        # Convert to frequencies
        angles = np.angle(roots)
        frequencies = angles * sr / (2 * np.pi)
        
        # Sort and filter to reasonable formant range
        frequencies = sorted([f for f in frequencies if 90 < f < 5000])
        
        # Return requested number of formants
        formants = frequencies[:num_formants]
        
        # Pad with defaults if not enough formants found
        defaults = [500.0, 1500.0, 2500.0, 3500.0, 4500.0]
        while len(formants) < num_formants:
            formants.append(defaults[len(formants)])
        
        return formants[:num_formants]


# =============================================================================
# UNIFIED MICRO-EXPRESSION INJECTOR
# =============================================================================

@dataclass
class MicroExpressionConfig:
    """Configuration for micro-expression injection."""
    enable_breathing: bool = True
    enable_hesitations: bool = True
    enable_lip_smacks: bool = True
    enable_coarticulation: bool = True
    breathing_intensity: float = 0.8
    hesitation_frequency: float = 0.5
    lip_smack_frequency: float = 0.3
    coarticulation_strength: float = 0.7


@dataclass
class MicroExpressionResult:
    """Result of micro-expression injection."""
    audio: np.ndarray
    sample_rate: int
    breaths_injected: int
    hesitations_injected: int
    lip_smacks_injected: int
    coarticulation_regions_smoothed: int
    processing_time: float
    quality_improvement: float


class MicroExpressionInjector:
    """
    Unified micro-expression injector that combines all micro-expression features.
    
    This class orchestrates:
    1. Breathing pattern analysis and injection
    2. Hesitation and filler sound injection
    3. Lip smack and mouth sound injection
    4. Coarticulation smoothing
    
    Goal: Make synthesized speech indistinguishable from real human speech
    by adding subtle human-like details.
    
    Requirements: 5.1, 5.2, 5.3, 5.4
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        config: Optional[MicroExpressionConfig] = None
    ):
        """
        Initialize the unified micro-expression injector.
        
        Args:
            sample_rate: Audio sample rate
            config: Configuration for injection parameters
        """
        self.sample_rate = sample_rate
        self.config = config or MicroExpressionConfig()
        
        # Initialize component analyzers and injectors
        self.breathing_analyzer = BreathingPatternAnalyzer(sample_rate)
        self.breathing_injector = NaturalBreathingInjector(sample_rate)
        self.hesitation_analyzer = HesitationPatternAnalyzer(sample_rate)
        self.hesitation_injector = HesitationInjector(sample_rate)
        self.lip_smack_analyzer = LipSmackPatternAnalyzer(sample_rate)
        self.lip_smack_injector = LipSmackInjector(sample_rate)
        self.coarticulation_smoother = CoarticulationSmoother(sample_rate)
        
        # Cached patterns from reference audio
        self._breathing_pattern: Optional[BreathingPattern] = None
        self._hesitation_pattern: Optional[HesitationPattern] = None
        self._lip_smack_pattern: Optional[LipSmackPattern] = None
        
        logger.info("MicroExpressionInjector initialized")
    
    def analyze_reference_audio(
        self,
        reference_audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze reference audio to extract all micro-expression patterns.
        
        This method extracts breathing, hesitation, and lip smack patterns
        from the reference audio for later injection into synthesized speech.
        
        Args:
            reference_audio: Reference audio array
            sample_rate: Sample rate (uses default if not provided)
            
        Returns:
            Dictionary with analysis results for all micro-expression types
        """
        import time
        start_time = time.time()
        
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            reference_audio = librosa.resample(
                reference_audio, orig_sr=sr, target_sr=self.sample_rate
            )
            sr = self.sample_rate
        
        results = {
            'breathing': None,
            'hesitation': None,
            'lip_smack': None,
            'analysis_time': 0.0
        }
        
        # Analyze breathing patterns
        if self.config.enable_breathing:
            try:
                breathing_result = self.breathing_analyzer.analyze_breathing_patterns(
                    reference_audio, sr
                )
                self._breathing_pattern = breathing_result.pattern
                results['breathing'] = {
                    'num_events': len(breathing_result.pattern.breath_events),
                    'breathing_rate': breathing_result.pattern.breathing_rate,
                    'quality_score': breathing_result.quality_score,
                    'confidence': breathing_result.confidence
                }
            except Exception as e:
                logger.warning(f"Breathing analysis failed: {e}")
        
        # Analyze hesitation patterns
        if self.config.enable_hesitations:
            try:
                hesitation_result = self.hesitation_analyzer.analyze_hesitation_patterns(
                    reference_audio, sr
                )
                self._hesitation_pattern = hesitation_result.pattern
                results['hesitation'] = {
                    'num_events': len(hesitation_result.pattern.hesitation_events),
                    'hesitation_rate': hesitation_result.pattern.hesitation_rate,
                    'quality_score': hesitation_result.quality_score,
                    'confidence': hesitation_result.confidence
                }
            except Exception as e:
                logger.warning(f"Hesitation analysis failed: {e}")
        
        # Analyze lip smack patterns
        if self.config.enable_lip_smacks:
            try:
                lip_smack_result = self.lip_smack_analyzer.analyze_lip_smack_patterns(
                    reference_audio, sr
                )
                self._lip_smack_pattern = lip_smack_result.pattern
                results['lip_smack'] = {
                    'num_events': len(lip_smack_result.pattern.lip_smack_events),
                    'lip_smack_rate': lip_smack_result.pattern.lip_smack_rate,
                    'quality_score': lip_smack_result.quality_score,
                    'confidence': lip_smack_result.confidence
                }
            except Exception as e:
                logger.warning(f"Lip smack analysis failed: {e}")
        
        results['analysis_time'] = time.time() - start_time
        
        logger.info(f"Reference audio analysis complete in {results['analysis_time']:.2f}s")
        return results
    
    def inject_micro_expressions(
        self,
        synthesized_audio: np.ndarray,
        sample_rate: Optional[int] = None,
        reference_audio: Optional[np.ndarray] = None
    ) -> MicroExpressionResult:
        """
        Inject micro-expressions into synthesized audio.
        
        This method applies all enabled micro-expression injections:
        1. Natural breathing at phrase boundaries
        2. Hesitations and fillers at appropriate points
        3. Lip smacks and mouth sounds
        4. Coarticulation smoothing
        
        Args:
            synthesized_audio: Synthesized audio to enhance
            sample_rate: Sample rate (uses default if not provided)
            reference_audio: Optional reference audio for on-the-fly analysis
            
        Returns:
            MicroExpressionResult with enhanced audio and statistics
        """
        import time
        start_time = time.time()
        
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            synthesized_audio = librosa.resample(
                synthesized_audio, orig_sr=sr, target_sr=self.sample_rate
            )
            sr = self.sample_rate
        
        # Analyze reference audio if provided and patterns not cached
        if reference_audio is not None and self._breathing_pattern is None:
            self.analyze_reference_audio(reference_audio, sr)
        
        # Track statistics
        breaths_injected = 0
        hesitations_injected = 0
        lip_smacks_injected = 0
        coarticulation_regions = 0
        
        # Start with the original audio
        enhanced_audio = synthesized_audio.copy()
        
        # 1. Inject breathing
        if self.config.enable_breathing and self._breathing_pattern is not None:
            try:
                breath_result = self.breathing_injector.inject_breathing(
                    enhanced_audio,
                    self._breathing_pattern,
                    sr,
                    intensity_scale=self.config.breathing_intensity
                )
                enhanced_audio = breath_result.audio
                breaths_injected = breath_result.breaths_injected
            except Exception as e:
                logger.warning(f"Breathing injection failed: {e}")
        
        # 2. Inject hesitations
        if self.config.enable_hesitations and self._hesitation_pattern is not None:
            try:
                hesitation_result = self.hesitation_injector.inject_hesitations(
                    enhanced_audio,
                    self._hesitation_pattern,
                    sr,
                    frequency_scale=self.config.hesitation_frequency
                )
                enhanced_audio = hesitation_result.audio
                hesitations_injected = hesitation_result.hesitations_injected
            except Exception as e:
                logger.warning(f"Hesitation injection failed: {e}")
        
        # 3. Inject lip smacks
        if self.config.enable_lip_smacks and self._lip_smack_pattern is not None:
            try:
                lip_smack_result = self.lip_smack_injector.inject_lip_smacks(
                    enhanced_audio,
                    self._lip_smack_pattern,
                    sr,
                    frequency_scale=self.config.lip_smack_frequency
                )
                enhanced_audio = lip_smack_result.audio
                lip_smacks_injected = lip_smack_result.lip_smacks_injected
            except Exception as e:
                logger.warning(f"Lip smack injection failed: {e}")
        
        # 4. Apply coarticulation smoothing
        if self.config.enable_coarticulation:
            try:
                coarticulation_result = self.coarticulation_smoother.smooth_coarticulation(
                    enhanced_audio,
                    sr,
                    strength=self.config.coarticulation_strength
                )
                enhanced_audio = coarticulation_result.audio
                coarticulation_regions = coarticulation_result.regions_smoothed
            except Exception as e:
                logger.warning(f"Coarticulation smoothing failed: {e}")
        
        # Calculate quality improvement
        quality_improvement = self._estimate_quality_improvement(
            synthesized_audio, enhanced_audio, sr
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Micro-expression injection complete: "
            f"{breaths_injected} breaths, {hesitations_injected} hesitations, "
            f"{lip_smacks_injected} lip smacks, {coarticulation_regions} coarticulation regions"
        )
        
        return MicroExpressionResult(
            audio=enhanced_audio,
            sample_rate=sr,
            breaths_injected=breaths_injected,
            hesitations_injected=hesitations_injected,
            lip_smacks_injected=lip_smacks_injected,
            coarticulation_regions_smoothed=coarticulation_regions,
            processing_time=processing_time,
            quality_improvement=quality_improvement
        )
    
    def _estimate_quality_improvement(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        sr: int
    ) -> float:
        """
        Estimate the quality improvement from micro-expression injection.
        
        Args:
            original: Original synthesized audio
            enhanced: Enhanced audio with micro-expressions
            sr: Sample rate
            
        Returns:
            Estimated quality improvement (0-1)
        """
        try:
            # Compare spectral characteristics
            orig_mfcc = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13)
            enh_mfcc = librosa.feature.mfcc(y=enhanced, sr=sr, n_mfcc=13)
            
            # Calculate spectral diversity (more diverse = more natural)
            orig_diversity = np.std(orig_mfcc)
            enh_diversity = np.std(enh_mfcc)
            
            diversity_improvement = (enh_diversity - orig_diversity) / (orig_diversity + 1e-8)
            
            # Normalize to 0-1 range
            improvement = np.clip(diversity_improvement * 0.5 + 0.5, 0, 1)
            
            return float(improvement)
        except Exception:
            return 0.5  # Default moderate improvement
    
    def clear_cached_patterns(self):
        """Clear all cached micro-expression patterns."""
        self._breathing_pattern = None
        self._hesitation_pattern = None
        self._lip_smack_pattern = None
        logger.info("Cached micro-expression patterns cleared")


# Global instance for easy access
micro_expression_injector = MicroExpressionInjector()


async def initialize_micro_expression_service():
    """Initialize the micro-expression injection service."""
    logger.info("Micro-expression injection service initialized")
    return True
