"""
Optimized Voice Analysis Service with performance optimization integration.

This service integrates with the performance optimization system to achieve
30-second analysis targets for 5-minute audio while maintaining quality.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import librosa
import concurrent.futures

from app.services.multi_dimensional_voice_analyzer import MultiDimensionalVoiceAnalyzer
from app.services.performance_optimization_service import performance_optimization_service
from app.schemas.voice import VoiceProfileSchema, VoiceCharacteristics
from app.core.config import settings

logger = logging.getLogger(__name__)


class OptimizedVoiceAnalysisService:
    """
    Optimized voice analysis service that leverages performance optimization
    to meet strict timing requirements while maintaining analysis quality.
    """
    
    def __init__(self):
        self.analyzer = MultiDimensionalVoiceAnalyzer()
        self.performance_service = performance_optimization_service
        
        # Analysis optimization parameters
        self.target_analysis_time = 30.0  # seconds for 5-minute audio
        self.quality_threshold = 0.8
        self.max_concurrent_analyses = 4
        
        # Adaptive analysis parameters
        self.adaptive_parameters = {
            "hop_length_multiplier": 1.0,
            "n_fft_reduction": 1.0,
            "feature_subset_ratio": 1.0,
            "precision_level": "high"
        }
        
        logger.info("Optimized Voice Analysis Service initialized")
    
    async def analyze_voice_optimized(
        self, 
        audio_path: str,
        progress_callback: Optional[Callable] = None,
        priority: int = 5,
        target_time: Optional[float] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Perform optimized voice analysis with performance targets.
        
        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback
            priority: Task priority (1=highest, 10=lowest)
            target_time: Optional custom target time (defaults to 30s)
            
        Returns:
            Tuple of (analysis_results, processing_time)
        """
        target_time = target_time or self.target_analysis_time
        
        try:
            if progress_callback:
                progress_callback(5, "Starting optimized voice analysis")
            
            # Use performance optimization service
            result, processing_time = await self.performance_service.optimize_voice_analysis(
                analysis_func=self._perform_comprehensive_analysis,
                audio_path=audio_path,
                target_time_seconds=target_time,
                progress_callback=progress_callback,
                priority=priority
            )
            
            return result, processing_time
            
        except Exception as e:
            logger.error(f"Optimized voice analysis failed: {e}")
            raise
    
    def _perform_comprehensive_analysis(
        self, 
        audio_path: str,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform comprehensive voice analysis with adaptive optimization.
        
        This method is called by the performance optimization service
        and implements adaptive analysis strategies based on audio characteristics.
        """
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(10, "Loading and preprocessing audio")
            
            # Load audio with optimization
            audio_data, sample_rate, audio_info = self._load_audio_optimized(audio_path)
            
            if progress_callback:
                progress_callback(20, "Determining analysis strategy")
            
            # Determine optimal analysis strategy based on audio characteristics
            analysis_strategy = self._determine_analysis_strategy(audio_data, sample_rate, audio_info)
            
            if progress_callback:
                progress_callback(30, f"Executing {analysis_strategy['name']} analysis")
            
            # Execute analysis with selected strategy
            if analysis_strategy["type"] == "fast":
                results = self._perform_fast_analysis(audio_data, sample_rate, progress_callback)
            elif analysis_strategy["type"] == "adaptive":
                results = self._perform_adaptive_analysis(audio_data, sample_rate, progress_callback)
            else:  # comprehensive
                results = self._perform_full_analysis(audio_data, sample_rate, progress_callback)
            
            if progress_callback:
                progress_callback(90, "Finalizing analysis results")
            
            # Add metadata
            processing_time = time.time() - start_time
            results.update({
                "analysis_strategy": analysis_strategy,
                "processing_time": processing_time,
                "audio_info": audio_info,
                "optimization_applied": True,
                "target_time_met": processing_time <= self.target_analysis_time
            })
            
            if progress_callback:
                progress_callback(100, f"Analysis complete ({processing_time:.1f}s)")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            raise
    
    def _load_audio_optimized(self, audio_path: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Load audio with optimization for analysis speed."""
        try:
            # Get audio file info first
            audio_info = librosa.get_duration(path=audio_path)
            duration = audio_info
            
            # Determine optimal loading strategy
            if duration > 300:  # > 5 minutes
                # For very long audio, use chunked loading
                audio_data, sample_rate = self._load_audio_chunked(audio_path)
                strategy = "chunked"
            elif duration > 60:  # > 1 minute
                # For medium audio, use optimized sample rate
                audio_data, sample_rate = librosa.load(audio_path, sr=16000)  # Lower sample rate for speed
                strategy = "downsampled"
            else:
                # For short audio, use full quality
                audio_data, sample_rate = librosa.load(audio_path, sr=22050)
                strategy = "full_quality"
            
            audio_metadata = {
                "duration": duration,
                "loading_strategy": strategy,
                "sample_rate": sample_rate,
                "samples": len(audio_data)
            }
            
            return audio_data, sample_rate, audio_metadata
            
        except Exception as e:
            logger.error(f"Optimized audio loading failed: {e}")
            raise
    
    def _load_audio_chunked(self, audio_path: str, chunk_duration: float = 60.0) -> Tuple[np.ndarray, int]:
        """Load audio in chunks and concatenate representative segments."""
        try:
            # Get total duration
            total_duration = librosa.get_duration(path=audio_path)
            
            # Select representative chunks (beginning, middle, end)
            chunk_times = [
                0,  # Beginning
                total_duration / 2 - chunk_duration / 2,  # Middle
                max(0, total_duration - chunk_duration)  # End
            ]
            
            chunks = []
            sample_rate = None
            
            for start_time in chunk_times:
                chunk, sr = librosa.load(
                    audio_path, 
                    sr=16000,  # Optimized sample rate
                    offset=start_time,
                    duration=min(chunk_duration, total_duration - start_time)
                )
                chunks.append(chunk)
                if sample_rate is None:
                    sample_rate = sr
            
            # Concatenate chunks with small gaps
            gap_samples = int(0.1 * sample_rate)  # 100ms gap
            gap = np.zeros(gap_samples)
            
            audio_data = np.concatenate([
                chunks[0], gap, chunks[1], gap, chunks[2]
            ])
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Chunked audio loading failed: {e}")
            raise
    
    def _determine_analysis_strategy(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        audio_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine optimal analysis strategy based on audio characteristics."""
        try:
            duration = len(audio_data) / sample_rate
            
            # Calculate audio complexity metrics
            rms_energy = np.sqrt(np.mean(audio_data**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
            
            # Determine complexity score
            complexity_score = (
                (rms_energy * 2) +
                (zero_crossing_rate * 5) +
                (spectral_centroid / 2000)
            ) / 3
            
            # Select strategy based on duration and complexity
            if duration <= 30 and complexity_score < 0.3:
                # Short, simple audio - use fast analysis
                strategy = {
                    "type": "fast",
                    "name": "Fast Analysis",
                    "features": ["basic_pitch", "basic_formants", "basic_timbre"],
                    "expected_time": 5.0,
                    "quality_level": 0.7
                }
            elif duration <= 120 or complexity_score < 0.6:
                # Medium audio or moderate complexity - use adaptive analysis
                strategy = {
                    "type": "adaptive",
                    "name": "Adaptive Analysis",
                    "features": ["pitch", "formants", "timbre", "basic_prosody"],
                    "expected_time": 15.0,
                    "quality_level": 0.85
                }
            else:
                # Long or complex audio - use comprehensive analysis with optimization
                strategy = {
                    "type": "comprehensive",
                    "name": "Optimized Comprehensive Analysis",
                    "features": ["all_features"],
                    "expected_time": 25.0,
                    "quality_level": 0.95
                }
            
            strategy.update({
                "audio_duration": duration,
                "complexity_score": complexity_score,
                "sample_rate": sample_rate
            })
            
            return strategy
            
        except Exception as e:
            logger.error(f"Analysis strategy determination failed: {e}")
            # Fallback to adaptive strategy
            return {
                "type": "adaptive",
                "name": "Fallback Adaptive Analysis",
                "features": ["pitch", "formants", "timbre"],
                "expected_time": 20.0,
                "quality_level": 0.8
            }
    
    def _perform_fast_analysis(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Perform fast analysis for simple, short audio."""
        try:
            results = {}
            
            if progress_callback:
                progress_callback(40, "Fast pitch analysis")
            
            # Basic pitch analysis
            f0, voiced_flag, _ = librosa.pyin(
                audio_data, 
                fmin=80, 
                fmax=400, 
                sr=sample_rate,
                hop_length=512  # Larger hop for speed
            )
            
            f0_clean = f0[voiced_flag & ~np.isnan(f0)]
            if len(f0_clean) > 0:
                results["pitch_features"] = {
                    "mean_f0": float(np.mean(f0_clean)),
                    "std_f0": float(np.std(f0_clean)),
                    "min_f0": float(np.min(f0_clean)),
                    "max_f0": float(np.max(f0_clean)),
                    "voiced_ratio": float(np.sum(voiced_flag) / len(voiced_flag))
                }
            
            if progress_callback:
                progress_callback(60, "Fast formant analysis")
            
            # Basic formant analysis (simplified)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=5)
            results["formant_features"] = {
                "mfcc_means": mfccs.mean(axis=1).tolist(),
                "mfcc_stds": mfccs.std(axis=1).tolist()
            }
            
            if progress_callback:
                progress_callback(80, "Fast timbre analysis")
            
            # Basic timbre analysis
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            
            results["timbre_features"] = {
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "brightness": float(np.mean(spectral_centroid) / 1000)
            }
            
            # Quality assessment
            results["quality_metrics"] = {
                "analysis_type": "fast",
                "feature_completeness": 0.6,
                "estimated_accuracy": 0.7
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Fast analysis failed: {e}")
            raise
    
    def _perform_adaptive_analysis(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Perform adaptive analysis balancing speed and quality."""
        try:
            results = {}
            
            if progress_callback:
                progress_callback(40, "Adaptive pitch analysis")
            
            # Enhanced pitch analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=50, 
                fmax=500, 
                sr=sample_rate,
                hop_length=256
            )
            
            f0_clean = f0[voiced_flag & ~np.isnan(f0)]
            if len(f0_clean) > 0:
                results["pitch_features"] = {
                    "mean_f0": float(np.mean(f0_clean)),
                    "std_f0": float(np.std(f0_clean)),
                    "min_f0": float(np.min(f0_clean)),
                    "max_f0": float(np.max(f0_clean)),
                    "voiced_ratio": float(np.sum(voiced_flag) / len(voiced_flag)),
                    "pitch_range_semitones": float(12 * np.log2(np.max(f0_clean) / np.min(f0_clean))),
                    "pitch_stability": float(1.0 - (np.std(f0_clean) / np.mean(f0_clean)))
                }
            
            if progress_callback:
                progress_callback(55, "Adaptive formant analysis")
            
            # Enhanced formant analysis
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            results["formant_features"] = {
                "mfcc_means": mfccs.mean(axis=1).tolist(),
                "mfcc_stds": mfccs.std(axis=1).tolist(),
                "mfcc_deltas": np.mean(np.diff(mfccs, axis=1), axis=1).tolist()
            }
            
            if progress_callback:
                progress_callback(70, "Adaptive timbre analysis")
            
            # Enhanced timbre analysis
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            results["timbre_features"] = {
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_centroid_std": float(np.std(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_flatness_mean": float(np.mean(spectral_flatness)),
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
                "brightness": float(np.mean(spectral_centroid) / 1000),
                "roughness": float(np.std(spectral_flatness))
            }
            
            if progress_callback:
                progress_callback(85, "Basic prosody analysis")
            
            # Basic prosody analysis
            rms_energy = librosa.feature.rms(y=audio_data)
            results["prosody_features"] = {
                "energy_mean": float(np.mean(rms_energy)),
                "energy_std": float(np.std(rms_energy)),
                "dynamic_range": float(20 * np.log10(np.max(rms_energy) / (np.mean(rms_energy) + 1e-8)))
            }
            
            # Quality assessment
            results["quality_metrics"] = {
                "analysis_type": "adaptive",
                "feature_completeness": 0.8,
                "estimated_accuracy": 0.85
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Adaptive analysis failed: {e}")
            raise
    
    def _perform_full_analysis(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis with optimization."""
        try:
            if progress_callback:
                progress_callback(40, "Comprehensive analysis with optimization")
            
            # Use the full multi-dimensional analyzer but with optimized parameters
            analyzer = MultiDimensionalVoiceAnalyzer(sample_rate=sample_rate)
            
            # Create temporary file for analysis
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sample_rate)
                tmp_path = tmp_file.name
            
            try:
                # Perform comprehensive analysis
                results = analyzer.analyze_voice_comprehensive(tmp_path)
                
                if progress_callback:
                    progress_callback(85, "Processing comprehensive results")
                
                # Convert complex objects to serializable format
                serializable_results = self._make_results_serializable(results)
                
                # Add quality assessment
                serializable_results["quality_metrics"] = {
                    "analysis_type": "comprehensive",
                    "feature_completeness": 1.0,
                    "estimated_accuracy": 0.95
                }
                
                return serializable_results
                
            finally:
                # Clean up temporary file
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Full analysis failed: {e}")
            raise
    
    def _make_results_serializable(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert analysis results to JSON-serializable format."""
        try:
            serializable = {}
            
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable[key] = float(value)
                elif hasattr(value, '__dict__'):
                    # Convert dataclass or object to dict
                    serializable[key] = self._convert_object_to_dict(value)
                elif isinstance(value, dict):
                    serializable[key] = self._make_results_serializable(value)
                elif isinstance(value, list):
                    serializable[key] = [
                        item.tolist() if isinstance(item, np.ndarray) else 
                        float(item) if isinstance(item, (np.integer, np.floating)) else
                        self._convert_object_to_dict(item) if hasattr(item, '__dict__') else
                        item
                        for item in value
                    ]
                else:
                    serializable[key] = value
            
            return serializable
            
        except Exception as e:
            logger.error(f"Result serialization failed: {e}")
            return {"error": f"Serialization failed: {e}"}
    
    def _convert_object_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert object with attributes to dictionary."""
        try:
            if hasattr(obj, '__dict__'):
                result = {}
                for attr_name, attr_value in obj.__dict__.items():
                    if isinstance(attr_value, np.ndarray):
                        result[attr_name] = attr_value.tolist()
                    elif isinstance(attr_value, (np.integer, np.floating)):
                        result[attr_name] = float(attr_value)
                    elif isinstance(attr_value, dict):
                        result[attr_name] = self._make_results_serializable(attr_value)
                    else:
                        result[attr_name] = attr_value
                return result
            else:
                return str(obj)
                
        except Exception as e:
            logger.error(f"Object conversion failed: {e}")
            return {"error": f"Conversion failed: {e}"}
    
    async def batch_analyze_voices(
        self, 
        audio_paths: List[str],
        progress_callback: Optional[Callable] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Perform batch voice analysis with concurrent processing.
        
        Args:
            audio_paths: List of audio file paths
            progress_callback: Optional progress callback
            max_concurrent: Maximum concurrent analyses (defaults to service limit)
            
        Returns:
            List of (audio_path, results, processing_time) tuples
        """
        try:
            max_concurrent = max_concurrent or self.max_concurrent_analyses
            
            if progress_callback:
                progress_callback(5, f"Starting batch analysis of {len(audio_paths)} files")
            
            # Create analysis tasks
            tasks = []
            for i, audio_path in enumerate(audio_paths):
                task_func = lambda path=audio_path: self._perform_comprehensive_analysis(path)
                tasks.append((task_func, (), {}))
            
            # Execute batch with concurrency control
            results = await self.performance_service.concurrency_manager.submit_batch_tasks(
                tasks=tasks,
                task_type="analysis",
                max_concurrent=max_concurrent
            )
            
            # Process results
            batch_results = []
            for i, (audio_path, result) in enumerate(zip(audio_paths, results)):
                if isinstance(result, Exception):
                    logger.error(f"Batch analysis failed for {audio_path}: {result}")
                    batch_results.append((audio_path, {"error": str(result)}, 0.0))
                else:
                    processing_time = result.get("processing_time", 0.0)
                    batch_results.append((audio_path, result, processing_time))
                
                if progress_callback:
                    progress = 10 + (i + 1) * 80 // len(audio_paths)
                    progress_callback(progress, f"Completed {i + 1}/{len(audio_paths)} analyses")
            
            if progress_callback:
                progress_callback(100, f"Batch analysis complete")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch voice analysis failed: {e}")
            raise
    
    def get_analysis_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for voice analysis operations."""
        try:
            # Get performance metrics from optimization service
            report = self.performance_service.get_performance_report()
            
            analysis_stats = report.get("analysis_performance", {})
            
            # Calculate target achievement rate
            analysis_times = self.performance_service.metrics.analysis_times
            target_met_count = len([t for t in analysis_times if t <= self.target_analysis_time])
            total_analyses = len(analysis_times)
            
            target_achievement_rate = target_met_count / total_analyses if total_analyses > 0 else 0.0
            
            return {
                "target_analysis_time": self.target_analysis_time,
                "target_achievement_rate": target_achievement_rate,
                "total_analyses": total_analyses,
                "analyses_meeting_target": target_met_count,
                "performance_stats": analysis_stats,
                "cache_hit_rate": report.get("cache_hit_rates", {}).get("voice_profile", 0.0),
                "average_analysis_time": analysis_stats.get("mean", 0.0),
                "fastest_analysis_time": analysis_stats.get("min", 0.0),
                "slowest_analysis_time": analysis_stats.get("max", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis performance stats: {e}")
            return {"error": str(e)}


# Global service instance
optimized_voice_analysis_service = OptimizedVoiceAnalysisService()


async def initialize_optimized_voice_analysis():
    """Initialize optimized voice analysis service."""
    try:
        logger.info("Optimized Voice Analysis Service initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Optimized voice analysis initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize optimized voice analysis: {e}")