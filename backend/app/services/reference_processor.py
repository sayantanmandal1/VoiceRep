"""
Reference audio processor for state-of-the-art voice cloning.

Cleans and optimizes reference audio using:
- Demucs v4 source separation (vocal isolation)
- Silero VAD (voice activity detection)
- Loudness normalization (pyloudnorm)
- Smart length handling for any duration input
- SHA-256 caching to avoid re-processing
"""

import os
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import librosa
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# Target parameters for XTTS v2
TARGET_SR = 22050
TARGET_LUFS = -20.0
MAX_REFERENCE_DURATION = 30.0  # seconds
MIN_REFERENCE_DURATION = 3.0   # seconds
OPTIMAL_SEGMENT_DURATION = 10.0  # seconds for segment selection


class ReferenceProcessor:
    """Process reference audio for optimal voice cloning quality."""

    def __init__(self, cache_dir: str = "models/ref_cache"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._demucs_model = None
        self._vad_model = None
        self._vad_utils = None
        logger.info("ReferenceProcessor initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, audio_path: str) -> str:
        """
        Process reference audio and return path to clean version.

        Pipeline:
          1. Check SHA-256 cache
          2. Demucs vocal isolation
          3. Silero VAD speech trimming
          4. Smart length handling
          5. Loudness normalization
          6. Write to cache and return path

        Args:
            audio_path: Path to raw reference audio file.

        Returns:
            Path to processed reference audio (WAV, 22050 Hz, mono).
        """
        audio_path = str(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        # 1. Cache check
        file_hash = self._sha256(audio_path)
        cached = self._cache_dir / f"{file_hash}.wav"
        if cached.exists():
            logger.info(f"Using cached processed reference: {cached}")
            return str(cached)

        logger.info(f"Processing reference audio: {audio_path}")

        # Load original audio
        audio, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        if len(audio) == 0:
            raise ValueError("Reference audio is empty")

        # 2. Demucs vocal isolation
        audio = self._isolate_vocals(audio_path)

        # 3. Silero VAD - keep only speech segments
        audio = self._apply_vad(audio, TARGET_SR)
        if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
            # VAD removed everything — fall back to original loaded audio
            logger.warning("VAD removed all audio, falling back to full file")
            audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        # 4. Smart length handling
        audio = self._handle_length(audio, TARGET_SR)

        # 5. Loudness normalization + peak limiting
        audio = self._normalize_loudness(audio, TARGET_SR)

        # 6. Write to cache
        sf.write(str(cached), audio, TARGET_SR, subtype="PCM_16")
        logger.info(f"Processed reference saved: {cached} ({len(audio)/TARGET_SR:.1f}s)")
        return str(cached)

    # ------------------------------------------------------------------
    # Demucs vocal isolation
    # ------------------------------------------------------------------

    def _isolate_vocals(self, audio_path: str) -> np.ndarray:
        """Use Demucs v4 to isolate vocals from background."""
        try:
            from demucs.api import Separator

            if self._demucs_model is None:
                logger.info("Loading Demucs htdemucs_ft model (first run downloads ~300MB)")
                self._demucs_model = Separator(
                    model="htdemucs_ft",
                    segment=7.8,
                    overlap=0.25,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

            # Separate
            _, separated = self._demucs_model.separate_audio_file(audio_path)

            # Extract vocals stem → numpy mono at TARGET_SR
            vocals = separated["vocals"]  # Tensor [channels, samples]
            if vocals.dim() == 2:
                vocals = vocals.mean(dim=0)  # mono
            vocals_np = vocals.cpu().numpy().astype(np.float32)

            # Demucs outputs at 44100 Hz — resample to TARGET_SR
            if len(vocals_np) > 0:
                vocals_np = librosa.resample(vocals_np, orig_sr=44100, target_sr=TARGET_SR)

            logger.info(f"Demucs vocal isolation complete: {len(vocals_np)/TARGET_SR:.1f}s")
            return vocals_np

        except Exception as e:
            logger.warning(f"Demucs vocal isolation failed ({e}), using raw audio")
            audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
            return audio

    # ------------------------------------------------------------------
    # Silero VAD
    # ------------------------------------------------------------------

    def _apply_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove non-speech segments using Silero VAD."""
        try:
            if self._vad_model is None:
                self._vad_model, self._vad_utils = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad",
                    trust_repo=True
                )
                logger.info("Silero VAD model loaded")

            get_speech_timestamps = self._vad_utils[0]

            # Silero VAD expects 16kHz
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            wav_tensor = torch.from_numpy(audio_16k).float()

            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                wav_tensor, self._vad_model,
                sampling_rate=16000,
                threshold=0.3,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
            )

            if not speech_timestamps:
                return audio

            # Convert timestamps from 16kHz to target SR
            ratio = sr / 16000.0
            speech_segments = []
            for ts in speech_timestamps:
                start = int(ts["start"] * ratio)
                end = int(ts["end"] * ratio)
                start = max(0, start)
                end = min(len(audio), end)
                if end > start:
                    speech_segments.append(audio[start:end])

            if not speech_segments:
                return audio

            # Concatenate speech segments with tiny crossfade
            return self._concat_with_crossfade(speech_segments, sr, crossfade_ms=10)

        except Exception as e:
            logger.warning(f"VAD failed ({e}), returning full audio")
            return audio

    # ------------------------------------------------------------------
    # Smart length handling
    # ------------------------------------------------------------------

    def _handle_length(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Handle any reference length:
        - < MIN_REFERENCE_DURATION: use as-is (XTTS minimum)
        - MIN_REFERENCE_DURATION to MAX_REFERENCE_DURATION: optimal, use full
        - > MAX_REFERENCE_DURATION: select best segments by energy/SNR
        """
        duration = len(audio) / sr

        if duration <= MAX_REFERENCE_DURATION:
            return audio

        # Select best segments for long audio
        logger.info(f"Reference is {duration:.1f}s, selecting best segments (max {MAX_REFERENCE_DURATION}s)")
        segment_samples = int(OPTIMAL_SEGMENT_DURATION * sr)
        hop = segment_samples // 2  # 50% overlap for candidate selection

        # Score each candidate segment
        candidates = []
        for start in range(0, len(audio) - segment_samples, hop):
            segment = audio[start : start + segment_samples]
            rms = np.sqrt(np.mean(segment ** 2))
            peak = np.max(np.abs(segment))
            # Prefer segments with good RMS (speech present) and low peak ratio (clean)
            if rms > 1e-4:
                snr_proxy = rms / (peak + 1e-8)  # Higher = more consistent energy
                candidates.append((start, rms * snr_proxy))

        if not candidates:
            # Fallback: just take the first MAX_REFERENCE_DURATION seconds
            return audio[: int(MAX_REFERENCE_DURATION * sr)]

        # Sort by score, take top-3 non-overlapping segments
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_starts = []
        for start, score in candidates:
            # Check no overlap with already selected segments
            overlap = False
            for s in selected_starts:
                if abs(start - s) < segment_samples:
                    overlap = True
                    break
            if not overlap:
                selected_starts.append(start)
            if len(selected_starts) >= 3:
                break

        # Sort by position to maintain temporal order
        selected_starts.sort()

        segments = [audio[s : s + segment_samples] for s in selected_starts]
        return self._concat_with_crossfade(segments, sr, crossfade_ms=20)

    # ------------------------------------------------------------------
    # Loudness normalization
    # ------------------------------------------------------------------

    def _normalize_loudness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize to target LUFS with peak limiting."""
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            current_lufs = meter.integrated_loudness(audio)

            if np.isinf(current_lufs) or np.isnan(current_lufs):
                # Audio too quiet—just peak normalize
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.9
                return audio

            # Normalize to target LUFS
            audio = pyln.normalize.loudness(audio, current_lufs, TARGET_LUFS)

            # Peak limit to -1 dBFS (≈ 0.891)
            peak = np.max(np.abs(audio))
            if peak > 0.891:
                audio = audio * (0.891 / peak)

            return audio

        except Exception as e:
            logger.warning(f"Loudness normalization failed ({e}), using peak normalization")
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.9
            return audio

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(filepath: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:24]  # shortened for filename

    @staticmethod
    def _concat_with_crossfade(
        segments: list, sr: int, crossfade_ms: int = 10
    ) -> np.ndarray:
        """Concatenate audio segments with a short crossfade."""
        if not segments:
            return np.array([], dtype=np.float32)
        if len(segments) == 1:
            return segments[0]

        xf_samples = int(crossfade_ms / 1000.0 * sr)
        xf_samples = max(1, xf_samples)

        result = segments[0].copy()
        for seg in segments[1:]:
            if len(result) < xf_samples or len(seg) < xf_samples:
                result = np.concatenate([result, seg])
                continue

            fade_out = np.linspace(1.0, 0.0, xf_samples, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xf_samples, dtype=np.float32)

            # Crossfade region
            overlap = result[-xf_samples:] * fade_out + seg[:xf_samples] * fade_in
            result = np.concatenate([result[:-xf_samples], overlap, seg[xf_samples:]])

        return result


# Singleton instance
reference_processor = ReferenceProcessor()
