"""
State-of-the-art voice cloner using a single XTTS v2 model.

Design principles:
  - ONE model → ONE voice → ZERO mixing artifacts
  - Demucs reference cleanup → best possible speaker embedding
  - Minimal post-processing: loudness norm, DC offset removal, light denoise
  - Quality assessment is READ-ONLY (never modifies audio)

This replaces the previous 15+ service pipeline (ensemble, RVC, Griffin-Lim,
spectral matching, formant shifting, etc.) that caused dual-voice overlapping.
"""

import os
import time
import logging
import warnings
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Callable

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 22050
TARGET_LUFS = -20.0


class VoiceCloner:
    """
    State-of-the-art voice cloner.

    Loads XTTS v2 once on GPU and keeps it resident for all synthesis calls.
    Reference audio is processed through Demucs + VAD + normalization.
    Post-processing is intentionally minimal to preserve XTTS's HiFi-GAN output.
    """

    def __init__(self):
        self._tts_model = None
        self._device = None
        self._model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self._ref_processor = None
        self._ecapa_model = None
        self._resemblyzer_encoder = None
        self._lock = asyncio.Lock()
        self._latent_cache: Dict[str, Any] = {}  # reference hash → conditioning latents
        logger.info("VoiceCloner created (model loads lazily on first use)")

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_tts_model(self):
        """Load XTTS v2 model on first use. Stays resident in GPU memory."""
        if self._tts_model is not None:
            return

        import torch

        # Handle PyTorch compatibility for TTS library
        try:
            from torch.nn.utils.parametrizations import weight_norm as _wn
        except ImportError:
            try:
                from torch.nn.utils import weight_norm as _wn
                import torch.nn.utils.parametrizations
                torch.nn.utils.parametrizations.weight_norm = _wn
            except ImportError:
                pass

        from TTS.api import TTS

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading XTTS v2 model on {self._device} (first run downloads ~1.8 GB)")

        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        self._tts_model = TTS(model_name=self._model_name).to(self._device)

        logger.info("XTTS v2 model loaded and ready")

    def _ensure_ref_processor(self):
        """Lazy-load the reference processor."""
        if self._ref_processor is not None:
            return
        from app.services.reference_processor import ReferenceProcessor
        self._ref_processor = ReferenceProcessor()
        # Clear cached references on first load so code changes take effect
        self._ref_processor.clear_cache()
        logger.info("Reference processor loaded and cache cleared")

    def _ensure_ecapa(self):
        """Lazy-load ECAPA-TDNN for speaker similarity scoring."""
        if self._ecapa_model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self._ecapa_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self._device or "cpu"},
            )
            logger.info("ECAPA-TDNN speaker encoder loaded")
        except Exception as e:
            logger.warning(f"ECAPA-TDNN not available: {e}")

    def _ensure_resemblyzer(self):
        """Lazy-load Resemblyzer for secondary speaker verification."""
        if self._resemblyzer_encoder is not None:
            return
        try:
            from resemblyzer import VoiceEncoder
            self._resemblyzer_encoder = VoiceEncoder()
            logger.info("Resemblyzer GE2E encoder loaded")
        except Exception as e:
            logger.warning(f"Resemblyzer not available: {e}")

    # ------------------------------------------------------------------
    # Main synthesis API
    # ------------------------------------------------------------------

    async def clone_voice(
        self,
        text: str,
        reference_audio_path: str,
        language: str = "en",
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Clone a voice: produce speech in the style of the reference audio.

        Args:
            text: Text to speak.
            reference_audio_path: Path to reference audio (any duration).
            language: BCP-47 language code (en, es, fr, de, it, pt, ...).
            output_dir: Directory for output file. Defaults to settings.RESULTS_DIR.
            progress_callback: Optional (progress_pct:int, message:str) callback.

        Returns:
            (output_path, metrics_dict)
        """
        async with self._lock:  # serialize GPU access
            return await self._clone_voice_impl(
                text, reference_audio_path, language, output_dir, progress_callback
            )

    async def _clone_voice_impl(
        self,
        text: str,
        reference_audio_path: str,
        language: str,
        output_dir: Optional[str],
        progress_callback: Optional[Callable],
    ) -> Tuple[str, Dict[str, Any]]:
        start = time.time()

        def _progress(pct: int, msg: str):
            if progress_callback:
                try:
                    progress_callback(pct, msg)
                except Exception:
                    pass

        # ---- Step 1: Reference processing (Demucs + VAD + normalization) ----
        _progress(5, "Processing reference audio (vocal isolation)")
        self._ensure_ref_processor()
        clean_ref = await self._ref_processor.process(reference_audio_path)
        _progress(20, "Reference audio processed")

        # ---- Step 2: Ensure XTTS model is loaded ----
        _progress(22, "Loading synthesis model")
        await asyncio.to_thread(self._ensure_tts_model)
        _progress(30, "Synthesis model ready")

        # ---- Step 3: Prepare output path ----
        from app.core.config import settings

        out_dir = Path(output_dir or settings.RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_path = str(out_dir / f"clone_{timestamp}.wav")

        # ---- Step 4: Map language code ----
        lang_code = self._map_language(language)

        # ---- Step 5: XTTS v2 synthesis (best-of-3 with low-level API) ----
        _progress(35, "Synthesizing speech with XTTS v2 (best-of-3 generation)")
        await asyncio.to_thread(
            self._synthesize,
            text=text,
            speaker_wav=clean_ref,
            language=lang_code,
            file_path=output_path,
        )

        # Verify output created
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("XTTS v2 failed to produce audio output")

        _progress(75, "Speech synthesized")

        # ---- Step 6: Light post-processing ----
        _progress(78, "Applying post-processing")
        await asyncio.to_thread(
            self._post_process, output_path, clean_ref
        )
        _progress(88, "Post-processing complete")

        # ---- Step 7: Quality assessment (read-only) ----
        _progress(90, "Measuring speaker similarity")
        metrics = await asyncio.to_thread(
            self._assess_quality, output_path, clean_ref
        )
        metrics["processing_time"] = round(time.time() - start, 2)
        metrics["reference_path"] = reference_audio_path
        metrics["language"] = language
        metrics["text_length"] = len(text)

        _progress(100, "Voice cloning complete")
        logger.info(
            f"Voice cloning done in {metrics['processing_time']}s — "
            f"similarity={metrics.get('similarity_score', 'N/A')}"
        )
        return output_path, metrics

    # ------------------------------------------------------------------
    # Conditioning latent extraction (cached)
    # ------------------------------------------------------------------

    def _get_conditioning_latents(self, speaker_wav: str):
        """
        Compute or retrieve cached conditioning latents for a reference audio.

        Uses the full reference (up to 60s) and computes speaker embedding
        with maximum context for best voice capture.
        """
        # Cache key based on file content
        cache_key = hashlib.sha256(
            open(speaker_wav, "rb").read()
        ).hexdigest()[:16]

        if cache_key in self._latent_cache:
            logger.info("Using cached conditioning latents")
            return self._latent_cache[cache_key]

        model = self._tts_model.synthesizer.tts_model
        logger.info("Computing conditioning latents from reference audio...")

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[speaker_wav],
            gpt_cond_len=30,           # use up to 30s for GPT conditioning
            gpt_cond_chunk_len=4,      # 4s chunks for fine-grained extraction
            max_ref_length=60,         # load up to 60s of reference (default is 10!)
            sound_norm_refs=True,      # normalize reference for consistent embedding
        )

        self._latent_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
        logger.info(
            f"Conditioning latents computed and cached "
            f"(GPT latent shape: {gpt_cond_latent.shape}, "
            f"speaker emb shape: {speaker_embedding.shape})"
        )
        return gpt_cond_latent, speaker_embedding

    # ------------------------------------------------------------------
    # XTTS v2 synthesis — low-level API with best-of-N
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        file_path: str,
    ):
        """
        High-quality synthesis using the low-level XTTS v2 model API.

        Instead of the high-level tts_to_file() wrapper (which uses default
        parameters and only 10s of reference), this method:
          1. Computes conditioning latents with full reference context (up to 60s)
          2. Runs best-of-3 generation with different temperatures
          3. Picks the attempt with highest MFCC similarity to reference
          4. Falls back to tts_to_file() if all else fails
        """
        import torch

        # Try low-level API first
        try:
            model = self._tts_model.synthesizer.tts_model
            gpt_cond_latent, speaker_embedding = self._get_conditioning_latents(speaker_wav)
        except Exception as e:
            logger.warning(f"Low-level XTTS API unavailable ({e}), using tts_to_file fallback")
            self._tts_model.tts_to_file(
                text=text, speaker_wav=speaker_wav,
                language=language, file_path=file_path, split_sentences=True,
            )
            return

        # Best-of-3 with increasing temperature
        # Lower temperature = more deterministic = closer to reference voice
        best_wav = None
        best_similarity = -1.0
        best_temp = None
        temperatures = [0.1, 0.3, 0.65]

        for temp in temperatures:
            try:
                logger.info(f"Synthesis attempt: temperature={temp}")
                out = model.inference(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=temp,
                    length_penalty=1.0,
                    repetition_penalty=10.0,    # high = prevent degenerate loops
                    top_k=50,
                    top_p=0.85,
                    do_sample=True,
                    speed=1.0,
                    enable_text_splitting=True,
                )

                wav = out["wav"]
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                wav = wav.squeeze().astype(np.float32)

                if len(wav) < 2000:  # too short = likely degenerate
                    logger.warning(f"  temp={temp}: output too short ({len(wav)} samples), skipping")
                    continue

                # Quick MFCC similarity to pick best candidate
                sim = self._quick_mfcc_similarity(wav, speaker_wav, sr=24000)
                duration = len(wav) / 24000
                logger.info(
                    f"  temp={temp}: similarity={sim:.4f}, "
                    f"duration={duration:.1f}s"
                )

                if sim > best_similarity:
                    best_similarity = sim
                    best_wav = wav
                    best_temp = temp

                # Early exit if we already have excellent similarity
                if sim > 0.85:
                    logger.info(f"  Excellent similarity at temp={temp}, stopping early")
                    break

            except Exception as e:
                logger.warning(f"  Synthesis attempt at temp={temp} failed: {e}")
                continue

        if best_wav is not None:
            logger.info(
                f"Best-of-{len(temperatures)} result: temp={best_temp}, "
                f"similarity={best_similarity:.4f}"
            )
            sf.write(file_path, best_wav, 24000, subtype="PCM_16")
            return

        # All attempts failed — fall back to high-level API
        logger.warning("All low-level synthesis attempts failed, falling back to tts_to_file")
        self._tts_model.tts_to_file(
            text=text, speaker_wav=speaker_wav,
            language=language, file_path=file_path, split_sentences=True,
        )

    def _quick_mfcc_similarity(
        self, wav: np.ndarray, ref_path: str, sr: int = 24000
    ) -> float:
        """Quick MFCC cosine similarity for best-of-N candidate ranking."""
        try:
            ref, _ = librosa.load(ref_path, sr=sr, mono=True)
            mfcc_a = np.mean(librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20), axis=1)
            mfcc_b = np.mean(librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=20), axis=1)
            cos = float(
                np.dot(mfcc_a, mfcc_b)
                / (np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_b) + 1e-8)
            )
            return max(0.0, min(1.0, cos))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Minimal post-processing
    # ------------------------------------------------------------------

    def _post_process(self, output_path: str, reference_path: str):
        """
        Light post-processing — intentionally minimal:
        1. DC offset removal (20 Hz high-pass)
        2. Light noise reduction
        3. Loudness normalization to match reference
        """
        # Load at native sample rate to avoid quality loss from resampling
        audio, sr = librosa.load(output_path, sr=None, mono=True)

        # 1. DC offset removal — 2nd-order Butterworth at 20 Hz
        sos = butter(2, 20.0, btype="high", fs=sr, output="sos")
        audio = sosfilt(sos, audio).astype(np.float32)

        # 2. Light noise reduction
        try:
            import noisereduce as nr
            audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=0.3,      # gentle — preserve voice character
                stationary=True,
                n_std_thresh_stationary=1.5,
            )
        except Exception as e:
            logger.debug(f"Noise reduction skipped: {e}")

        # 3. Loudness normalization — match reference LUFS
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)

            ref_audio, _ = librosa.load(reference_path, sr=sr, mono=True)
            ref_lufs = meter.integrated_loudness(ref_audio)

            if np.isinf(ref_lufs) or np.isnan(ref_lufs):
                ref_lufs = TARGET_LUFS

            current_lufs = meter.integrated_loudness(audio)
            if not (np.isinf(current_lufs) or np.isnan(current_lufs)):
                audio = pyln.normalize.loudness(audio, current_lufs, ref_lufs)
        except Exception as e:
            logger.debug(f"LUFS normalization skipped: {e}")

        # Peak limit to -1 dBFS
        peak = np.max(np.abs(audio))
        if peak > 0.891:
            audio = audio * (0.891 / peak)

        # Ensure no silence
        if np.max(np.abs(audio)) < 1e-4:
            raise RuntimeError("Post-processed audio is silent")

        sf.write(output_path, audio, sr, subtype="PCM_16")

    # ------------------------------------------------------------------
    # Quality assessment (READ-ONLY — never modifies audio)
    # ------------------------------------------------------------------

    def _assess_quality(self, output_path: str, reference_path: str) -> Dict[str, Any]:
        """
        Measure speaker similarity between output and reference.
        Uses ECAPA-TDNN (60% weight) + Resemblyzer (40% weight).
        """
        metrics: Dict[str, Any] = {
            "similarity_score": 0.0,
            "ecapa_similarity": None,
            "resemblyzer_similarity": None,
            "quality_level": "unknown",
        }

        # ECAPA-TDNN
        ecapa_sim = self._ecapa_similarity(output_path, reference_path)
        metrics["ecapa_similarity"] = ecapa_sim

        # Resemblyzer
        resem_sim = self._resemblyzer_similarity(output_path, reference_path)
        metrics["resemblyzer_similarity"] = resem_sim

        # Weighted combined score
        scores = []
        weights = []
        if ecapa_sim is not None:
            scores.append(ecapa_sim)
            weights.append(0.6)
        if resem_sim is not None:
            scores.append(resem_sim)
            weights.append(0.4)

        if scores:
            total_w = sum(weights)
            combined = sum(s * w for s, w in zip(scores, weights)) / total_w
            metrics["similarity_score"] = round(combined, 4)
        else:
            # Fallback: basic spectral similarity
            metrics["similarity_score"] = self._spectral_similarity(output_path, reference_path)

        # Classify quality level
        sim = metrics["similarity_score"]
        if sim >= 0.85:
            metrics["quality_level"] = "excellent"
        elif sim >= 0.75:
            metrics["quality_level"] = "good"
        elif sim >= 0.60:
            metrics["quality_level"] = "fair"
        else:
            metrics["quality_level"] = "poor"

        return metrics

    def _ecapa_similarity(self, path_a: str, path_b: str) -> Optional[float]:
        """ECAPA-TDNN cosine similarity."""
        try:
            self._ensure_ecapa()
            if self._ecapa_model is None:
                return None

            import torch

            emb_a = self._ecapa_model.encode_batch(
                self._ecapa_model.load_audio(path_a).unsqueeze(0)
            )
            emb_b = self._ecapa_model.encode_batch(
                self._ecapa_model.load_audio(path_b).unsqueeze(0)
            )
            cos_sim = torch.nn.functional.cosine_similarity(emb_a.squeeze(), emb_b.squeeze(), dim=0)
            return round(float(cos_sim.item()), 4)
        except Exception as e:
            logger.debug(f"ECAPA similarity failed: {e}")
            return None

    def _resemblyzer_similarity(self, path_a: str, path_b: str) -> Optional[float]:
        """Resemblyzer GE2E cosine similarity."""
        try:
            self._ensure_resemblyzer()
            if self._resemblyzer_encoder is None:
                return None

            from resemblyzer import preprocess_wav

            wav_a = preprocess_wav(path_a)
            wav_b = preprocess_wav(path_b)
            emb_a = self._resemblyzer_encoder.embed_utterance(wav_a)
            emb_b = self._resemblyzer_encoder.embed_utterance(wav_b)

            cos_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))
            return round(max(0.0, min(1.0, cos_sim)), 4)
        except Exception as e:
            logger.debug(f"Resemblyzer similarity failed: {e}")
            return None

    def _spectral_similarity(self, path_a: str, path_b: str) -> float:
        """Fallback: MFCC-based cosine similarity."""
        try:
            a, _ = librosa.load(path_a, sr=SAMPLE_RATE, mono=True)
            b, _ = librosa.load(path_b, sr=SAMPLE_RATE, mono=True)
            mfcc_a = np.mean(librosa.feature.mfcc(y=a, sr=SAMPLE_RATE, n_mfcc=20), axis=1)
            mfcc_b = np.mean(librosa.feature.mfcc(y=b, sr=SAMPLE_RATE, n_mfcc=20), axis=1)
            cos = float(np.dot(mfcc_a, mfcc_b) / (np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_b) + 1e-8))
            return round(max(0.0, min(1.0, cos)), 4)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Language mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_language(lang: str) -> str:
        """
        Map language names/codes to XTTS v2 two-letter codes.
        XTTS v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko
        """
        mapping = {
            # Full names
            "english": "en", "spanish": "es", "french": "fr",
            "german": "de", "italian": "it", "portuguese": "pt",
            "polish": "pl", "turkish": "tr", "russian": "ru",
            "dutch": "nl", "czech": "cs", "arabic": "ar",
            "chinese": "zh-cn", "japanese": "ja", "hungarian": "hu",
            "korean": "ko",
            # Already correct codes
            "en": "en", "es": "es", "fr": "fr", "de": "de",
            "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
            "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
            "zh-cn": "zh-cn", "ja": "ja", "hu": "hu", "ko": "ko",
            # Common variants
            "zh": "zh-cn", "cn": "zh-cn",
        }
        return mapping.get(lang.lower().strip(), "en")


# Singleton — loads lazily, stays resident
voice_cloner = VoiceCloner()
