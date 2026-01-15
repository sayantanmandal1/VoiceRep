"""
ECAPA-TDNN Speaker Encoder for High-Fidelity Voice Cloning.

This module implements state-of-the-art speaker embedding extraction using
ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN).
ECAPA-TDNN achieves best-in-class speaker verification performance with
attentive statistical pooling and multi-scale feature aggregation.
"""

import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Try to import SpeechBrain for ECAPA-TDNN
SPEECHBRAIN_AVAILABLE = False
try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
    logger.info("SpeechBrain ECAPA-TDNN encoder available")
except ImportError:
    logger.warning("SpeechBrain not available. Install with: pip install speechbrain")


@dataclass
class ECAPAEmbedding:
    """Container for ECAPA-TDNN speaker embedding and metadata."""
    embedding: np.ndarray  # 192-dimensional embedding
    confidence: float  # Embedding quality confidence (0-1)
    audio_duration: float  # Duration of processed audio
    sample_rate: int  # Sample rate used
    model_version: str  # Model version identifier


@dataclass
class ECAPABatchResult:
    """Container for batch embedding extraction results."""
    embeddings: List[ECAPAEmbedding]
    mean_embedding: np.ndarray
    weighted_embedding: np.ndarray  # Weighted by confidence
    overall_confidence: float


class ECAPASpeakerEncoder:
    """
    ECAPA-TDNN Speaker Encoder for extracting high-quality speaker embeddings.
    
    ECAPA-TDNN uses:
    - 1D Squeeze-Excitation (SE) blocks for channel attention
    - Multi-scale feature aggregation with Res2Net-style connections
    - Attentive statistical pooling for utterance-level representation
    
    The model produces 192-dimensional speaker embeddings that capture
    unique voice characteristics with high discriminative power.
    """
    
    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize ECAPA-TDNN speaker encoder.
        
        Args:
            model_source: HuggingFace model identifier or local path
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory for model cache
        """
        self.model_source = model_source
        self.cache_dir = cache_dir or "models/ecapa_cache"
        self.embedding_dim = 192
        self.target_sample_rate = 16000
        self.model_version = "ecapa-tdnn-voxceleb-v1"
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self._initialized = False
        
        logger.info(f"ECAPA Speaker Encoder initialized (device: {self.device})")
    
    def initialize(self) -> bool:
        """
        Initialize the ECAPA-TDNN model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        if not SPEECHBRAIN_AVAILABLE:
            logger.error("SpeechBrain not available. Cannot initialize ECAPA-TDNN.")
            return False
        
        try:
            logger.info(f"Loading ECAPA-TDNN model from {self.model_source}")
            
            # Create cache directory
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Load pretrained ECAPA-TDNN model
            self.model = EncoderClassifier.from_hparams(
                source=self.model_source,
                savedir=self.cache_dir,
                run_opts={"device": self.device}
            )
            
            self._initialized = True
            logger.info("ECAPA-TDNN model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ECAPA-TDNN: {e}")
            return False
    
    def extract_embedding(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None
    ) -> Optional[ECAPAEmbedding]:
        """
        Extract speaker embedding from audio.
        
        Args:
            audio_path: Path to audio file
            audio_array: Audio as numpy array (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array provided)
            
        Returns:
            ECAPAEmbedding object or None if extraction fails
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            # Load audio
            if audio_path is not None:
                waveform, sr = torchaudio.load(audio_path)
            elif audio_array is not None:
                if sample_rate is None:
                    raise ValueError("sample_rate required when providing audio_array")
                waveform = torch.from_numpy(audio_array).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                sr = sample_rate
            else:
                raise ValueError("Either audio_path or audio_array must be provided")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to target sample rate
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Calculate audio duration
            audio_duration = waveform.shape[1] / self.target_sample_rate
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(waveform.to(self.device))
                embedding = embedding.squeeze().cpu().numpy()
            
            # Normalize embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # Calculate confidence based on embedding statistics
            confidence = self._calculate_embedding_confidence(embedding, audio_duration)
            
            return ECAPAEmbedding(
                embedding=embedding,
                confidence=confidence,
                audio_duration=audio_duration,
                sample_rate=self.target_sample_rate,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Failed to extract ECAPA embedding: {e}")
            return None
    
    def extract_batch_embeddings(
        self,
        audio_paths: List[str]
    ) -> Optional[ECAPABatchResult]:
        """
        Extract embeddings from multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            ECAPABatchResult with individual and aggregated embeddings
        """
        if not audio_paths:
            return None
        
        embeddings = []
        for path in audio_paths:
            emb = self.extract_embedding(audio_path=path)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        # Calculate mean embedding
        emb_array = np.array([e.embedding for e in embeddings])
        mean_embedding = np.mean(emb_array, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)
        
        # Calculate confidence-weighted embedding
        confidences = np.array([e.confidence for e in embeddings])
        weights = confidences / (np.sum(confidences) + 1e-8)
        weighted_embedding = np.sum(emb_array * weights[:, np.newaxis], axis=0)
        weighted_embedding = weighted_embedding / (np.linalg.norm(weighted_embedding) + 1e-8)
        
        # Overall confidence
        overall_confidence = np.mean(confidences)
        
        return ECAPABatchResult(
            embeddings=embeddings,
            mean_embedding=mean_embedding,
            weighted_embedding=weighted_embedding,
            overall_confidence=overall_confidence
        )
    
    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, ECAPAEmbedding],
        embedding2: Union[np.ndarray, ECAPAEmbedding]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (array or ECAPAEmbedding)
            embedding2: Second embedding (array or ECAPAEmbedding)
            
        Returns:
            Cosine similarity score (0-1)
        """
        if isinstance(embedding1, ECAPAEmbedding):
            embedding1 = embedding1.embedding
        if isinstance(embedding2, ECAPAEmbedding):
            embedding2 = embedding2.embedding
        
        # Normalize
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to 0-1 range
        return float((similarity + 1) / 2)
    
    def verify_speaker(
        self,
        audio_path1: str,
        audio_path2: str,
        threshold: float = 0.75
    ) -> Tuple[bool, float]:
        """
        Verify if two audio samples are from the same speaker.
        
        Args:
            audio_path1: Path to first audio file
            audio_path2: Path to second audio file
            threshold: Similarity threshold for same-speaker decision
            
        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        emb1 = self.extract_embedding(audio_path=audio_path1)
        emb2 = self.extract_embedding(audio_path=audio_path2)
        
        if emb1 is None or emb2 is None:
            return False, 0.0
        
        similarity = self.compute_similarity(emb1, emb2)
        is_same = similarity >= threshold
        
        return is_same, similarity
    
    def _calculate_embedding_confidence(
        self,
        embedding: np.ndarray,
        audio_duration: float
    ) -> float:
        """
        Calculate confidence score for extracted embedding.
        
        Confidence is based on:
        - Audio duration (longer is better, up to a point)
        - Embedding statistics (variance, sparsity)
        
        Args:
            embedding: Extracted embedding
            audio_duration: Duration of source audio
            
        Returns:
            Confidence score (0-1)
        """
        # Duration factor (optimal: 3-10 seconds)
        if audio_duration < 1.0:
            duration_factor = 0.5
        elif audio_duration < 3.0:
            duration_factor = 0.7 + 0.1 * (audio_duration - 1.0)
        elif audio_duration <= 10.0:
            duration_factor = 0.9 + 0.01 * (audio_duration - 3.0)
        else:
            duration_factor = 0.97
        
        # Embedding quality factor (based on variance)
        emb_variance = np.var(embedding)
        variance_factor = min(1.0, emb_variance * 10)  # Normalize
        
        # Sparsity factor (embeddings shouldn't be too sparse)
        sparsity = np.mean(np.abs(embedding) < 0.01)
        sparsity_factor = 1.0 - sparsity
        
        # Combine factors
        confidence = (
            duration_factor * 0.5 +
            variance_factor * 0.3 +
            sparsity_factor * 0.2
        )
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.embedding_dim
    
    def is_available(self) -> bool:
        """Check if ECAPA-TDNN is available."""
        return SPEECHBRAIN_AVAILABLE


# Global instance for convenience
_ecapa_encoder: Optional[ECAPASpeakerEncoder] = None


def get_ecapa_encoder() -> ECAPASpeakerEncoder:
    """Get or create global ECAPA encoder instance."""
    global _ecapa_encoder
    if _ecapa_encoder is None:
        _ecapa_encoder = ECAPASpeakerEncoder()
    return _ecapa_encoder
