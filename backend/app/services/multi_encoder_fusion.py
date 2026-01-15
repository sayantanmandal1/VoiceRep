"""
Multi-Encoder Fusion Network for Perfect Voice Cloning.

This module implements an advanced speaker embedding fusion system that combines
embeddings from multiple state-of-the-art speaker encoders (ECAPA-TDNN, WavLM,
Resemblyzer) to create a comprehensive, high-fidelity speaker representation.

The fusion network uses attention-based combination with encoder confidence
weighting to produce optimal speaker embeddings for voice cloning.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Import individual encoders
from app.services.ecapa_speaker_encoder import (
    ECAPASpeakerEncoder, ECAPAEmbedding, get_ecapa_encoder, SPEECHBRAIN_AVAILABLE
)
from app.services.wavlm_speaker_encoder import (
    WavLMSpeakerEncoder, WavLMEmbedding, get_wavlm_encoder, TRANSFORMERS_AVAILABLE
)

# Try to import Resemblyzer
RESEMBLYZER_AVAILABLE = False
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
    logger.info("Resemblyzer encoder available")
except ImportError:
    logger.warning("Resemblyzer not available. Install with: pip install resemblyzer")


@dataclass
class FusedSpeakerEmbedding:
    """Container for fused multi-encoder speaker embedding."""
    fused_embedding: np.ndarray  # 1024-dimensional fused embedding
    ecapa_embedding: Optional[np.ndarray]  # 192-dim ECAPA embedding
    wavlm_embedding: Optional[np.ndarray]  # 768-dim WavLM embedding
    resemblyzer_embedding: Optional[np.ndarray]  # 256-dim Resemblyzer embedding
    encoder_confidences: Dict[str, float]  # Per-encoder confidence scores
    encoder_weights: Dict[str, float]  # Weights used in fusion
    overall_confidence: float  # Overall embedding confidence
    audio_duration: float  # Duration of processed audio
    encoders_used: List[str]  # List of encoders that contributed


@dataclass
class SpeakerSimilarityResult:
    """Result of speaker similarity comparison."""
    overall_similarity: float  # Fused embedding similarity
    per_encoder_similarity: Dict[str, float]  # Per-encoder similarities
    confidence: float  # Confidence in the comparison


class AttentionFusionNetwork(nn.Module):
    """
    Attention-based fusion network for combining multi-encoder embeddings.
    
    Uses self-attention to learn optimal combination weights for different
    encoder outputs based on the input characteristics.
    """
    
    def __init__(
        self,
        ecapa_dim: int = 192,
        wavlm_dim: int = 768,
        resemblyzer_dim: int = 256,
        output_dim: int = 1024,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.ecapa_dim = ecapa_dim
        self.wavlm_dim = wavlm_dim
        self.resemblyzer_dim = resemblyzer_dim
        self.output_dim = output_dim
        
        # Projection layers to common dimension
        self.ecapa_proj = nn.Linear(ecapa_dim, hidden_dim)
        self.wavlm_proj = nn.Linear(wavlm_dim, hidden_dim)
        self.resemblyzer_proj = nn.Linear(resemblyzer_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        ecapa_emb: Optional[torch.Tensor] = None,
        wavlm_emb: Optional[torch.Tensor] = None,
        resemblyzer_emb: Optional[torch.Tensor] = None,
        encoder_weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Forward pass for fusion network.
        
        Args:
            ecapa_emb: ECAPA embedding (batch, 192)
            wavlm_emb: WavLM embedding (batch, 768)
            resemblyzer_emb: Resemblyzer embedding (batch, 256)
            encoder_weights: Optional weights for each encoder
            
        Returns:
            Fused embedding (batch, 1024)
        """
        embeddings = []
        weights = []
        
        # Project each available embedding
        if ecapa_emb is not None:
            proj_ecapa = self.ecapa_proj(ecapa_emb)
            embeddings.append(proj_ecapa)
            weights.append(encoder_weights.get('ecapa', 1.0) if encoder_weights else 1.0)
        
        if wavlm_emb is not None:
            proj_wavlm = self.wavlm_proj(wavlm_emb)
            embeddings.append(proj_wavlm)
            weights.append(encoder_weights.get('wavlm', 1.0) if encoder_weights else 1.0)
        
        if resemblyzer_emb is not None:
            proj_resemblyzer = self.resemblyzer_proj(resemblyzer_emb)
            embeddings.append(proj_resemblyzer)
            weights.append(encoder_weights.get('resemblyzer', 1.0) if encoder_weights else 1.0)
        
        if not embeddings:
            raise ValueError("At least one embedding must be provided")
        
        # Stack embeddings for attention
        # Shape: (batch, num_encoders, hidden_dim)
        stacked = torch.stack(embeddings, dim=1)
        
        # Apply attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Weighted combination
        weights_tensor = torch.tensor(weights, device=stacked.device).float()
        weights_tensor = weights_tensor / weights_tensor.sum()
        weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(-1)  # (1, num_encoders, 1)
        
        combined = (attended * weights_tensor).sum(dim=1)  # (batch, hidden_dim)
        
        # Output projection
        output = self.output_proj(combined)
        output = self.layer_norm(output)
        
        return output


class MultiEncoderFusion:
    """
    Multi-Encoder Fusion System for comprehensive speaker embedding extraction.
    
    Combines embeddings from:
    - ECAPA-TDNN: Best-in-class speaker verification (192-dim)
    - WavLM: Rich acoustic features from self-supervised learning (768-dim)
    - Resemblyzer: GE2E-trained encoder for voice similarity (256-dim)
    
    The fusion produces a 1024-dimensional embedding that captures all aspects
    of speaker identity for high-fidelity voice cloning.
    """
    
    # Default encoder weights based on voice cloning performance
    DEFAULT_WEIGHTS = {
        'ecapa': 0.35,      # Best for speaker identity
        'wavlm': 0.40,      # Best for acoustic details
        'resemblyzer': 0.25  # Good for voice similarity
    }
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_attention_fusion: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize multi-encoder fusion system.
        
        Args:
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            use_attention_fusion: Whether to use attention-based fusion
            cache_dir: Directory for model cache
        """
        self.cache_dir = cache_dir or "models/fusion_cache"
        self.use_attention_fusion = use_attention_fusion
        self.output_dim = 1024
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize encoders
        self.ecapa_encoder: Optional[ECAPASpeakerEncoder] = None
        self.wavlm_encoder: Optional[WavLMSpeakerEncoder] = None
        self.resemblyzer_encoder = None
        
        # Fusion network
        self.fusion_network: Optional[AttentionFusionNetwork] = None
        
        self._initialized = False
        
        logger.info(f"Multi-Encoder Fusion initialized (device: {self.device})")
    
    def initialize(self) -> bool:
        """
        Initialize all available encoders and fusion network.
        
        Returns:
            True if at least one encoder initialized successfully
        """
        if self._initialized:
            return True
        
        encoders_available = []
        
        # Initialize ECAPA-TDNN
        if SPEECHBRAIN_AVAILABLE:
            try:
                self.ecapa_encoder = ECAPASpeakerEncoder(device=self.device)
                if self.ecapa_encoder.initialize():
                    encoders_available.append('ecapa')
                    logger.info("ECAPA-TDNN encoder initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ECAPA-TDNN: {e}")
        
        # Initialize WavLM
        if TRANSFORMERS_AVAILABLE:
            try:
                self.wavlm_encoder = WavLMSpeakerEncoder(device=self.device)
                if self.wavlm_encoder.initialize():
                    encoders_available.append('wavlm')
                    logger.info("WavLM encoder initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WavLM: {e}")
        
        # Initialize Resemblyzer
        if RESEMBLYZER_AVAILABLE:
            try:
                self.resemblyzer_encoder = VoiceEncoder(device=self.device)
                encoders_available.append('resemblyzer')
                logger.info("Resemblyzer encoder initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Resemblyzer: {e}")
        
        if not encoders_available:
            logger.error("No encoders available. Cannot initialize fusion system.")
            return False
        
        # Initialize fusion network
        if self.use_attention_fusion:
            try:
                self.fusion_network = AttentionFusionNetwork()
                self.fusion_network.to(self.device)
                self.fusion_network.eval()
                logger.info("Attention fusion network initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize fusion network: {e}")
                self.use_attention_fusion = False
        
        self._initialized = True
        logger.info(f"Multi-encoder fusion ready with encoders: {encoders_available}")
        return True
    
    def extract_fused_embedding(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
        encoder_weights: Optional[Dict[str, float]] = None
    ) -> Optional[FusedSpeakerEmbedding]:
        """
        Extract fused speaker embedding from audio.
        
        Args:
            audio_path: Path to audio file
            audio_array: Audio as numpy array (alternative to audio_path)
            sample_rate: Sample rate of audio_array
            encoder_weights: Optional custom weights for each encoder
            
        Returns:
            FusedSpeakerEmbedding object or None if extraction fails
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        weights = encoder_weights or self.DEFAULT_WEIGHTS.copy()
        
        # Extract embeddings from each available encoder
        ecapa_emb = None
        wavlm_emb = None
        resemblyzer_emb = None
        encoder_confidences = {}
        encoders_used = []
        audio_duration = 0.0
        
        # ECAPA-TDNN
        if self.ecapa_encoder is not None:
            try:
                result = self.ecapa_encoder.extract_embedding(
                    audio_path=audio_path,
                    audio_array=audio_array,
                    sample_rate=sample_rate
                )
                if result is not None:
                    ecapa_emb = result.embedding
                    encoder_confidences['ecapa'] = result.confidence
                    encoders_used.append('ecapa')
                    audio_duration = max(audio_duration, result.audio_duration)
            except Exception as e:
                logger.warning(f"ECAPA extraction failed: {e}")
        
        # WavLM
        if self.wavlm_encoder is not None:
            try:
                result = self.wavlm_encoder.extract_embedding(
                    audio_path=audio_path,
                    audio_array=audio_array,
                    sample_rate=sample_rate
                )
                if result is not None:
                    wavlm_emb = result.embedding
                    encoder_confidences['wavlm'] = result.confidence
                    encoders_used.append('wavlm')
                    audio_duration = max(audio_duration, result.audio_duration)
            except Exception as e:
                logger.warning(f"WavLM extraction failed: {e}")
        
        # Resemblyzer
        if self.resemblyzer_encoder is not None:
            try:
                if audio_path is not None:
                    wav = preprocess_wav(audio_path)
                elif audio_array is not None:
                    # Resemblyzer expects 16kHz audio
                    if sample_rate != 16000:
                        import torchaudio
                        wav_tensor = torch.from_numpy(audio_array).float()
                        if wav_tensor.dim() == 1:
                            wav_tensor = wav_tensor.unsqueeze(0)
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        wav = resampler(wav_tensor).squeeze().numpy()
                    else:
                        wav = audio_array
                else:
                    wav = None
                
                if wav is not None:
                    resemblyzer_emb = self.resemblyzer_encoder.embed_utterance(wav)
                    # Estimate confidence based on audio quality
                    encoder_confidences['resemblyzer'] = self._estimate_resemblyzer_confidence(
                        resemblyzer_emb, len(wav) / 16000
                    )
                    encoders_used.append('resemblyzer')
                    audio_duration = max(audio_duration, len(wav) / 16000)
            except Exception as e:
                logger.warning(f"Resemblyzer extraction failed: {e}")
        
        if not encoders_used:
            logger.error("No embeddings extracted from any encoder")
            return None
        
        # Adjust weights based on available encoders
        available_weights = {k: v for k, v in weights.items() if k in encoders_used}
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v / total_weight for k, v in available_weights.items()}
        
        # Fuse embeddings
        fused_embedding = self._fuse_embeddings(
            ecapa_emb=ecapa_emb,
            wavlm_emb=wavlm_emb,
            resemblyzer_emb=resemblyzer_emb,
            weights=normalized_weights,
            confidences=encoder_confidences
        )
        
        # Calculate overall confidence
        overall_confidence = sum(
            encoder_confidences.get(enc, 0) * normalized_weights.get(enc, 0)
            for enc in encoders_used
        )
        
        return FusedSpeakerEmbedding(
            fused_embedding=fused_embedding,
            ecapa_embedding=ecapa_emb,
            wavlm_embedding=wavlm_emb,
            resemblyzer_embedding=resemblyzer_emb,
            encoder_confidences=encoder_confidences,
            encoder_weights=normalized_weights,
            overall_confidence=overall_confidence,
            audio_duration=audio_duration,
            encoders_used=encoders_used
        )
    
    def _fuse_embeddings(
        self,
        ecapa_emb: Optional[np.ndarray],
        wavlm_emb: Optional[np.ndarray],
        resemblyzer_emb: Optional[np.ndarray],
        weights: Dict[str, float],
        confidences: Dict[str, float]
    ) -> np.ndarray:
        """
        Fuse embeddings from multiple encoders.
        
        Args:
            ecapa_emb: ECAPA embedding
            wavlm_emb: WavLM embedding
            resemblyzer_emb: Resemblyzer embedding
            weights: Encoder weights
            confidences: Encoder confidences
            
        Returns:
            Fused 1024-dimensional embedding
        """
        if self.use_attention_fusion and self.fusion_network is not None:
            return self._attention_fusion(ecapa_emb, wavlm_emb, resemblyzer_emb, weights)
        else:
            return self._weighted_concatenation_fusion(
                ecapa_emb, wavlm_emb, resemblyzer_emb, weights, confidences
            )
    
    def _attention_fusion(
        self,
        ecapa_emb: Optional[np.ndarray],
        wavlm_emb: Optional[np.ndarray],
        resemblyzer_emb: Optional[np.ndarray],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Fuse embeddings using attention network."""
        with torch.no_grad():
            ecapa_tensor = torch.from_numpy(ecapa_emb).float().unsqueeze(0).to(self.device) if ecapa_emb is not None else None
            wavlm_tensor = torch.from_numpy(wavlm_emb).float().unsqueeze(0).to(self.device) if wavlm_emb is not None else None
            resemblyzer_tensor = torch.from_numpy(resemblyzer_emb).float().unsqueeze(0).to(self.device) if resemblyzer_emb is not None else None
            
            fused = self.fusion_network(
                ecapa_emb=ecapa_tensor,
                wavlm_emb=wavlm_tensor,
                resemblyzer_emb=resemblyzer_tensor,
                encoder_weights=weights
            )
            
            fused_np = fused.squeeze().cpu().numpy()
            return fused_np / (np.linalg.norm(fused_np) + 1e-8)
    
    def _weighted_concatenation_fusion(
        self,
        ecapa_emb: Optional[np.ndarray],
        wavlm_emb: Optional[np.ndarray],
        resemblyzer_emb: Optional[np.ndarray],
        weights: Dict[str, float],
        confidences: Dict[str, float]
    ) -> np.ndarray:
        """
        Fuse embeddings using weighted concatenation and projection.
        
        This is a fallback method when attention fusion is not available.
        """
        # Concatenate available embeddings with padding
        parts = []
        
        if ecapa_emb is not None:
            # Pad ECAPA (192) to 256
            padded = np.zeros(256)
            padded[:192] = ecapa_emb * weights.get('ecapa', 1.0) * confidences.get('ecapa', 1.0)
            parts.append(padded)
        else:
            parts.append(np.zeros(256))
        
        if wavlm_emb is not None:
            # Truncate WavLM (768) to 512
            truncated = wavlm_emb[:512] * weights.get('wavlm', 1.0) * confidences.get('wavlm', 1.0)
            parts.append(truncated)
        else:
            parts.append(np.zeros(512))
        
        if resemblyzer_emb is not None:
            # Use Resemblyzer (256) as-is
            weighted = resemblyzer_emb * weights.get('resemblyzer', 1.0) * confidences.get('resemblyzer', 1.0)
            parts.append(weighted)
        else:
            parts.append(np.zeros(256))
        
        # Concatenate to 1024 dimensions
        fused = np.concatenate(parts)
        
        # Normalize
        return fused / (np.linalg.norm(fused) + 1e-8)
    
    def _estimate_resemblyzer_confidence(
        self,
        embedding: np.ndarray,
        audio_duration: float
    ) -> float:
        """Estimate confidence for Resemblyzer embedding."""
        # Duration factor
        if audio_duration < 1.0:
            duration_factor = 0.5
        elif audio_duration < 3.0:
            duration_factor = 0.7 + 0.1 * (audio_duration - 1.0)
        else:
            duration_factor = min(0.95, 0.9 + 0.01 * (audio_duration - 3.0))
        
        # Embedding quality factor
        emb_variance = np.var(embedding)
        variance_factor = min(1.0, emb_variance * 10)
        
        return float(duration_factor * 0.7 + variance_factor * 0.3)
    
    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, FusedSpeakerEmbedding],
        embedding2: Union[np.ndarray, FusedSpeakerEmbedding]
    ) -> SpeakerSimilarityResult:
        """
        Compute comprehensive similarity between two speaker embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            SpeakerSimilarityResult with overall and per-encoder similarities
        """
        # Extract fused embeddings
        if isinstance(embedding1, FusedSpeakerEmbedding):
            fused1 = embedding1.fused_embedding
            emb1_obj = embedding1
        else:
            fused1 = embedding1
            emb1_obj = None
        
        if isinstance(embedding2, FusedSpeakerEmbedding):
            fused2 = embedding2.fused_embedding
            emb2_obj = embedding2
        else:
            fused2 = embedding2
            emb2_obj = None
        
        # Overall similarity
        fused1_norm = fused1 / (np.linalg.norm(fused1) + 1e-8)
        fused2_norm = fused2 / (np.linalg.norm(fused2) + 1e-8)
        overall_similarity = float((np.dot(fused1_norm, fused2_norm) + 1) / 2)
        
        # Per-encoder similarities
        per_encoder = {}
        
        if emb1_obj is not None and emb2_obj is not None:
            # ECAPA similarity
            if emb1_obj.ecapa_embedding is not None and emb2_obj.ecapa_embedding is not None:
                ecapa1 = emb1_obj.ecapa_embedding / (np.linalg.norm(emb1_obj.ecapa_embedding) + 1e-8)
                ecapa2 = emb2_obj.ecapa_embedding / (np.linalg.norm(emb2_obj.ecapa_embedding) + 1e-8)
                per_encoder['ecapa'] = float((np.dot(ecapa1, ecapa2) + 1) / 2)
            
            # WavLM similarity
            if emb1_obj.wavlm_embedding is not None and emb2_obj.wavlm_embedding is not None:
                wavlm1 = emb1_obj.wavlm_embedding / (np.linalg.norm(emb1_obj.wavlm_embedding) + 1e-8)
                wavlm2 = emb2_obj.wavlm_embedding / (np.linalg.norm(emb2_obj.wavlm_embedding) + 1e-8)
                per_encoder['wavlm'] = float((np.dot(wavlm1, wavlm2) + 1) / 2)
            
            # Resemblyzer similarity
            if emb1_obj.resemblyzer_embedding is not None and emb2_obj.resemblyzer_embedding is not None:
                res1 = emb1_obj.resemblyzer_embedding / (np.linalg.norm(emb1_obj.resemblyzer_embedding) + 1e-8)
                res2 = emb2_obj.resemblyzer_embedding / (np.linalg.norm(emb2_obj.resemblyzer_embedding) + 1e-8)
                per_encoder['resemblyzer'] = float((np.dot(res1, res2) + 1) / 2)
        
        # Confidence based on available encoders
        confidence = len(per_encoder) / 3.0 if per_encoder else 0.5
        
        return SpeakerSimilarityResult(
            overall_similarity=overall_similarity,
            per_encoder_similarity=per_encoder,
            confidence=confidence
        )
    
    def get_available_encoders(self) -> List[str]:
        """Get list of available encoders."""
        available = []
        if self.ecapa_encoder is not None:
            available.append('ecapa')
        if self.wavlm_encoder is not None:
            available.append('wavlm')
        if self.resemblyzer_encoder is not None:
            available.append('resemblyzer')
        return available
    
    def get_output_dim(self) -> int:
        """Get output embedding dimensionality."""
        return self.output_dim


# Global instance
_multi_encoder_fusion: Optional[MultiEncoderFusion] = None


def get_multi_encoder_fusion() -> MultiEncoderFusion:
    """Get or create global multi-encoder fusion instance."""
    global _multi_encoder_fusion
    if _multi_encoder_fusion is None:
        _multi_encoder_fusion = MultiEncoderFusion()
    return _multi_encoder_fusion
