"""
WavLM Speaker Encoder for High-Fidelity Voice Cloning.

This module implements speaker embedding extraction using WavLM, a large-scale
self-supervised pre-trained model for full-stack speech processing. WavLM
captures rich acoustic and speaker information through masked speech denoising
and prediction, making it excellent for voice cloning applications.
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

# Try to import transformers for WavLM
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
    logger.info("WavLM encoder available via transformers")
except ImportError:
    logger.warning("Transformers not available. Install with: pip install transformers")


@dataclass
class WavLMEmbedding:
    """Container for WavLM speaker embedding and metadata."""
    embedding: np.ndarray  # 768-dimensional embedding (or fused multi-layer)
    layer_embeddings: Dict[int, np.ndarray]  # Per-layer embeddings
    confidence: float  # Embedding quality confidence (0-1)
    audio_duration: float  # Duration of processed audio
    sample_rate: int  # Sample rate used
    model_version: str  # Model version identifier


@dataclass
class WavLMBatchResult:
    """Container for batch embedding extraction results."""
    embeddings: List[WavLMEmbedding]
    mean_embedding: np.ndarray
    weighted_embedding: np.ndarray
    overall_confidence: float


class WavLMSpeakerEncoder:
    """
    WavLM Speaker Encoder for extracting rich speaker embeddings.
    
    WavLM uses:
    - Masked speech denoising and prediction for pre-training
    - Transformer architecture with 24 layers
    - Multi-layer feature extraction for comprehensive representation
    
    The model produces 768-dimensional embeddings per layer, which can be
    combined for optimal speaker representation.
    """
    
    # Optimal layers for speaker information (based on research)
    SPEAKER_LAYERS = [6, 12, 18, 24]  # Layers with best speaker info
    
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_weighted_layers: bool = True
    ):
        """
        Initialize WavLM speaker encoder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory for model cache
            use_weighted_layers: Whether to use weighted layer combination
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "models/wavlm_cache"
        self.use_weighted_layers = use_weighted_layers
        self.embedding_dim = 768
        self.target_sample_rate = 16000
        self.model_version = "wavlm-large-v1"
        
        # Layer weights for speaker embedding (learned from speaker verification)
        # Higher layers capture more speaker-specific information
        self.layer_weights = {
            6: 0.15,   # Lower-level acoustic features
            12: 0.25,  # Mid-level features
            18: 0.30,  # Higher-level speaker features
            24: 0.30   # Top-level speaker features
        }
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.feature_extractor = None
        self._initialized = False
        
        logger.info(f"WavLM Speaker Encoder initialized (device: {self.device})")
    
    def initialize(self) -> bool:
        """
        Initialize the WavLM model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available. Cannot initialize WavLM.")
            return False
        
        try:
            logger.info(f"Loading WavLM model from {self.model_name}")
            
            # Create cache directory
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Load feature extractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load WavLM model
            self.model = WavLMModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                output_hidden_states=True  # Enable hidden state output
            )
            self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            logger.info("WavLM model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WavLM: {e}")
            return False
    
    def extract_embedding(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None
    ) -> Optional[WavLMEmbedding]:
        """
        Extract speaker embedding from audio.
        
        Args:
            audio_path: Path to audio file
            audio_array: Audio as numpy array (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array provided)
            
        Returns:
            WavLMEmbedding object or None if extraction fails
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        try:
            # Load audio
            if audio_path is not None:
                waveform, sr = torchaudio.load(audio_path)
                waveform = waveform.squeeze().numpy()
            elif audio_array is not None:
                if sample_rate is None:
                    raise ValueError("sample_rate required when providing audio_array")
                waveform = audio_array
                sr = sample_rate
            else:
                raise ValueError("Either audio_path or audio_array must be provided")
            
            # Convert to mono if stereo
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)
            
            # Resample to target sample rate
            if sr != self.target_sample_rate:
                waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                waveform = resampler(waveform_tensor).squeeze().numpy()
            
            # Calculate audio duration
            audio_duration = len(waveform) / self.target_sample_rate
            
            # Process through feature extractor
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            # Extract hidden states
            with torch.no_grad():
                outputs = self.model(input_values, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
            
            # Extract embeddings from specific layers
            layer_embeddings = {}
            for layer_idx in self.SPEAKER_LAYERS:
                if layer_idx < len(hidden_states):
                    # Mean pooling over time dimension
                    layer_emb = hidden_states[layer_idx].mean(dim=1).squeeze().cpu().numpy()
                    layer_emb = layer_emb / (np.linalg.norm(layer_emb) + 1e-8)
                    layer_embeddings[layer_idx] = layer_emb
            
            # Compute fused embedding
            if self.use_weighted_layers and layer_embeddings:
                fused_embedding = np.zeros(self.embedding_dim)
                total_weight = 0
                for layer_idx, emb in layer_embeddings.items():
                    weight = self.layer_weights.get(layer_idx, 0.25)
                    fused_embedding += weight * emb
                    total_weight += weight
                fused_embedding = fused_embedding / (total_weight + 1e-8)
            else:
                # Use last layer if no weighted fusion
                fused_embedding = hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
            
            # Normalize final embedding
            fused_embedding = fused_embedding / (np.linalg.norm(fused_embedding) + 1e-8)
            
            # Calculate confidence
            confidence = self._calculate_embedding_confidence(
                fused_embedding, layer_embeddings, audio_duration
            )
            
            return WavLMEmbedding(
                embedding=fused_embedding,
                layer_embeddings=layer_embeddings,
                confidence=confidence,
                audio_duration=audio_duration,
                sample_rate=self.target_sample_rate,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Failed to extract WavLM embedding: {e}")
            return None
    
    def extract_batch_embeddings(
        self,
        audio_paths: List[str]
    ) -> Optional[WavLMBatchResult]:
        """
        Extract embeddings from multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            WavLMBatchResult with individual and aggregated embeddings
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
        
        return WavLMBatchResult(
            embeddings=embeddings,
            mean_embedding=mean_embedding,
            weighted_embedding=weighted_embedding,
            overall_confidence=overall_confidence
        )
    
    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, WavLMEmbedding],
        embedding2: Union[np.ndarray, WavLMEmbedding]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (array or WavLMEmbedding)
            embedding2: Second embedding (array or WavLMEmbedding)
            
        Returns:
            Cosine similarity score (0-1)
        """
        if isinstance(embedding1, WavLMEmbedding):
            embedding1 = embedding1.embedding
        if isinstance(embedding2, WavLMEmbedding):
            embedding2 = embedding2.embedding
        
        # Normalize
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to 0-1 range
        return float((similarity + 1) / 2)
    
    def compute_layer_similarities(
        self,
        embedding1: WavLMEmbedding,
        embedding2: WavLMEmbedding
    ) -> Dict[int, float]:
        """
        Compute per-layer similarities between two embeddings.
        
        Args:
            embedding1: First WavLM embedding
            embedding2: Second WavLM embedding
            
        Returns:
            Dictionary of layer index to similarity score
        """
        similarities = {}
        
        for layer_idx in self.SPEAKER_LAYERS:
            if layer_idx in embedding1.layer_embeddings and layer_idx in embedding2.layer_embeddings:
                emb1 = embedding1.layer_embeddings[layer_idx]
                emb2 = embedding2.layer_embeddings[layer_idx]
                
                emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
                emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
                
                similarity = np.dot(emb1_norm, emb2_norm)
                similarities[layer_idx] = float((similarity + 1) / 2)
        
        return similarities
    
    def _calculate_embedding_confidence(
        self,
        embedding: np.ndarray,
        layer_embeddings: Dict[int, np.ndarray],
        audio_duration: float
    ) -> float:
        """
        Calculate confidence score for extracted embedding.
        
        Confidence is based on:
        - Audio duration
        - Layer consistency (similar embeddings across layers = more confident)
        - Embedding statistics
        
        Args:
            embedding: Fused embedding
            layer_embeddings: Per-layer embeddings
            audio_duration: Duration of source audio
            
        Returns:
            Confidence score (0-1)
        """
        # Duration factor
        if audio_duration < 1.0:
            duration_factor = 0.5
        elif audio_duration < 3.0:
            duration_factor = 0.7 + 0.1 * (audio_duration - 1.0)
        elif audio_duration <= 10.0:
            duration_factor = 0.9 + 0.01 * (audio_duration - 3.0)
        else:
            duration_factor = 0.97
        
        # Layer consistency factor
        if len(layer_embeddings) >= 2:
            layer_embs = list(layer_embeddings.values())
            consistencies = []
            for i in range(len(layer_embs)):
                for j in range(i + 1, len(layer_embs)):
                    sim = np.dot(layer_embs[i], layer_embs[j])
                    consistencies.append((sim + 1) / 2)
            consistency_factor = np.mean(consistencies) if consistencies else 0.8
        else:
            consistency_factor = 0.8
        
        # Embedding quality factor
        emb_variance = np.var(embedding)
        variance_factor = min(1.0, emb_variance * 10)
        
        # Combine factors
        confidence = (
            duration_factor * 0.4 +
            consistency_factor * 0.4 +
            variance_factor * 0.2
        )
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.embedding_dim
    
    def is_available(self) -> bool:
        """Check if WavLM is available."""
        return TRANSFORMERS_AVAILABLE


# Global instance for convenience
_wavlm_encoder: Optional[WavLMSpeakerEncoder] = None


def get_wavlm_encoder() -> WavLMSpeakerEncoder:
    """Get or create global WavLM encoder instance."""
    global _wavlm_encoder
    if _wavlm_encoder is None:
        _wavlm_encoder = WavLMSpeakerEncoder()
    return _wavlm_encoder
