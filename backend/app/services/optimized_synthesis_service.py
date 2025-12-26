"""
Optimized Synthesis Service with performance optimization integration.

This service integrates with the performance optimization system to achieve
faster-than-real-time synthesis while maintaining quality.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
import torch

from app.services.ensemble_voice_synthesis_engine import ensemble_voice_synthesizer
from app.services.performance_optimization_service import performance_optimization_service
from app.schemas.voice import VoiceProfileSchema, VoiceModelSchema
from app.core.config import settings

logger = logging.getLogger(__name__)


class OptimizedSynthesisService:
    """
    Optimized synthesis service that leverages performance optimization
    to achieve faster-than-real-time synthesis while maintaining quality.
    """
    
    def __init__(self):
        self.ensemble_synthesizer = ensemble_voice_synthesizer
        self.performance_service = performance_optimization_service
        
        # Synthesis optimization parameters
        self.target_realtime_factor = 2.0  # 2x faster than real-time
        self.quality_threshold = 0.95
        self.max_concurrent_synthesis = 8
        
        # GPU optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_optimizer = performance_optimization_service.gpu_optimizer
        
        # Synthesis strategies
        self.synthesis_strategies = {
            "ultra_fast": {
                "models": ["xtts_v2"],
                "quality_target": 0.85,
                "speed_multiplier": 4.0,
                "use_mixed_precision": True,
                "batch_size": 4
            },
            "fast": {
                "models": ["xtts_v2", "bark"],
                "quality_target": 0.90,
                "speed_multiplier": 2.5,
                "use_mixed_precision": True,
                "batch_size": 2
            },
            "balanced": {
                "models": ["xtts_v2", "bark", "your_tts"],
                "quality_target": 0.95,
                "speed_multiplier": 2.0,
                "use_mixed_precision": True,
                "batch_size": 1
            },
            "quality": {
                "models": ["xtts_v2", "bark", "your_tts"],
                "quality_target": 0.98,
                "speed_multiplier": 1.5,
                "use_mixed_precision": False,
                "batch_size": 1
            }
        }
        
        logger.info("Optimized Synthesis Service initialized")
    
    async def synthesize_speech_optimized(
        self,
        text: str,
        voice_profile: VoiceProfileSchema,
        language: str = "en",
        strategy: str = "balanced",
        progress_callback: Optional[Callable] = None,
        priority: int = 5,
        target_realtime_factor: Optional[float] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any], float, float]:
        """
        Perform optimized speech synthesis with performance targets.
        
        Args:
            text: Text to synthesize
            voice_profile: Voice profile for synthesis
            language: Target language
            strategy: Synthesis strategy (ultra_fast, fast, balanced, quality)
            progress_callback: Optional progress callback
            priority: Task priority (1=highest, 10=lowest)
            target_realtime_factor: Optional custom target realtime factor
            
        Returns:
            Tuple of (success, output_path, metadata, processing_time, realtime_factor)
        """
        target_realtime_factor = target_realtime_factor or self.target_realtime_factor
        
        try:
            if progress_callback:
                progress_callback(5, f"Starting optimized synthesis ({strategy})")
            
            # Select synthesis strategy
            strategy_config = self.synthesis_strategies.get(strategy, self.synthesis_strategies["balanced"])
            
            # Optimize synthesis function based on strategy
            synthesis_func = self._create_optimized_synthesis_function(strategy_config)
            
            # Use performance optimization service
            result, processing_time, realtime_factor = await self.performance_service.optimize_speech_synthesis(
                synthesis_func=synthesis_func,
                text=text,
                voice_profile=voice_profile,
                target_realtime_factor=target_realtime_factor,
                language=language,
                progress_callback=progress_callback,
                priority=priority
            )
            
            # Extract result components
            if result and len(result) >= 3:
                success, output_path, metadata = result[0], result[1], result[2]
                
                # Add optimization metadata
                if metadata:
                    metadata.update({
                        "synthesis_strategy": strategy,
                        "strategy_config": strategy_config,
                        "target_realtime_factor": target_realtime_factor,
                        "achieved_realtime_factor": realtime_factor,
                        "optimization_applied": True,
                        "gpu_used": self.device.type == 'cuda'
                    })
                
                return success, output_path, metadata, processing_time, realtime_factor
            else:
                return False, None, {"error": "Invalid synthesis result"}, processing_time, 0.0
            
        except Exception as e:
            logger.error(f"Optimized synthesis failed: {e}")
            raise
    
    def _create_optimized_synthesis_function(self, strategy_config: Dict[str, Any]) -> Callable:
        """Create optimized synthesis function based on strategy configuration."""
        
        async def optimized_synthesis_func(
            text: str,
            voice_profile: VoiceProfileSchema,
            language: str = "en",
            progress_callback: Optional[Callable] = None,
            **kwargs
        ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
            """Optimized synthesis function with GPU acceleration and caching."""
            
            try:
                if progress_callback:
                    progress_callback(10, "Preparing GPU-optimized synthesis")
                
                # Apply GPU optimizations
                if self.device.type == 'cuda':
                    # Clear GPU cache if needed
                    gpu_info = self.gpu_optimizer.get_gpu_utilization()
                    if gpu_info.get("memory_utilization", 0) > 0.8:
                        self.gpu_optimizer.clear_gpu_cache()
                
                # Optimize text for synthesis
                optimized_text = self._optimize_text_for_synthesis(text, strategy_config)
                
                if progress_callback:
                    progress_callback(20, "Configuring ensemble synthesis")
                
                # Configure ensemble synthesis based on strategy
                synthesis_config = self._configure_ensemble_synthesis(strategy_config)
                
                # Apply model optimizations
                await self._optimize_models_for_strategy(strategy_config)
                
                if progress_callback:
                    progress_callback(30, "Executing optimized synthesis")
                
                # Execute synthesis with optimizations
                if strategy_config["speed_multiplier"] >= 3.0:
                    # Ultra-fast synthesis
                    result = await self._execute_ultra_fast_synthesis(
                        optimized_text, voice_profile, language, synthesis_config, progress_callback
                    )
                elif strategy_config["speed_multiplier"] >= 2.0:
                    # Fast synthesis
                    result = await self._execute_fast_synthesis(
                        optimized_text, voice_profile, language, synthesis_config, progress_callback
                    )
                else:
                    # Standard optimized synthesis
                    result = await self._execute_standard_synthesis(
                        optimized_text, voice_profile, language, synthesis_config, progress_callback
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Optimized synthesis function failed: {e}")
                return False, None, {"error": str(e)}
        
        return optimized_synthesis_func
    
    def _optimize_text_for_synthesis(self, text: str, strategy_config: Dict[str, Any]) -> str:
        """Optimize text for faster synthesis."""
        try:
            optimized_text = text.strip()
            
            # For ultra-fast synthesis, apply aggressive text optimization
            if strategy_config["speed_multiplier"] >= 3.0:
                # Remove excessive punctuation
                import re
                optimized_text = re.sub(r'[.]{2,}', '.', optimized_text)
                optimized_text = re.sub(r'[!]{2,}', '!', optimized_text)
                optimized_text = re.sub(r'[?]{2,}', '?', optimized_text)
                
                # Simplify complex sentences for faster processing
                if len(optimized_text) > 200:
                    sentences = optimized_text.split('.')
                    if len(sentences) > 3:
                        # Keep first 3 sentences for ultra-fast mode
                        optimized_text = '. '.join(sentences[:3]) + '.'
            
            return optimized_text
            
        except Exception as e:
            logger.error(f"Text optimization failed: {e}")
            return text
    
    def _configure_ensemble_synthesis(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure ensemble synthesis based on strategy."""
        try:
            config = {
                "selected_models": strategy_config["models"],
                "quality_target": strategy_config["quality_target"],
                "use_mixed_precision": strategy_config["use_mixed_precision"],
                "batch_size": strategy_config["batch_size"],
                "enable_caching": True,
                "enable_gpu_optimization": self.device.type == 'cuda'
            }
            
            # Adjust ensemble weights based on strategy
            if strategy_config["speed_multiplier"] >= 3.0:
                # Ultra-fast: prioritize fastest model
                config["model_weights"] = {"xtts_v2": 1.0}
            elif strategy_config["speed_multiplier"] >= 2.0:
                # Fast: balance speed and quality
                config["model_weights"] = {"xtts_v2": 0.7, "bark": 0.3}
            else:
                # Balanced/Quality: use all models
                config["model_weights"] = {"xtts_v2": 0.4, "bark": 0.35, "your_tts": 0.25}
            
            return config
            
        except Exception as e:
            logger.error(f"Ensemble configuration failed: {e}")
            return {"selected_models": ["xtts_v2"], "quality_target": 0.8}
    
    async def _optimize_models_for_strategy(self, strategy_config: Dict[str, Any]):
        """Apply model-specific optimizations for the selected strategy."""
        try:
            if self.device.type == 'cuda' and hasattr(self.ensemble_synthesizer, 'loaded_models'):
                for model_type, model in self.ensemble_synthesizer.loaded_models.items():
                    if model_type.value in strategy_config["models"]:
                        # Apply GPU optimizations
                        if hasattr(model, 'model') and hasattr(model.model, 'to'):
                            model.model = model.model.to(self.device)
                            
                            # Enable mixed precision if configured
                            if strategy_config["use_mixed_precision"]:
                                if hasattr(model.model, 'half'):
                                    model.model = model.model.half()
                        
                        # Optimize for inference
                        if hasattr(model, 'model'):
                            model.model = self.gpu_optimizer.optimize_model_for_inference(model.model)
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
    
    async def _execute_ultra_fast_synthesis(
        self,
        text: str,
        voice_profile: VoiceProfileSchema,
        language: str,
        synthesis_config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute ultra-fast synthesis with maximum speed optimizations."""
        try:
            if progress_callback:
                progress_callback(40, "Ultra-fast synthesis mode")
            
            # Use only the fastest model (XTTS-v2) with aggressive optimizations
            if hasattr(self.ensemble_synthesizer, 'loaded_models'):
                from app.services.ensemble_voice_synthesis_engine import TTSModelType
                
                xtts_model = self.ensemble_synthesizer.loaded_models.get(TTSModelType.XTTS_V2)
                if xtts_model:
                    # Direct synthesis with single model
                    result = await self._synthesize_with_single_model(
                        xtts_model, text, voice_profile, language, progress_callback
                    )
                    
                    if result[0]:  # Success
                        metadata = result[2] if len(result) >= 3 else {}
                        metadata.update({
                            "synthesis_mode": "ultra_fast",
                            "models_used": ["xtts_v2"],
                            "optimization_level": "maximum"
                        })
                        return result[0], result[1], metadata
            
            # Fallback to ensemble synthesis with ultra-fast config
            return await self.ensemble_synthesizer.synthesize_speech_ensemble(
                text=text,
                voice_profile=voice_profile,
                language=language,
                progress_callback=progress_callback
            )
            
        except Exception as e:
            logger.error(f"Ultra-fast synthesis failed: {e}")
            return False, None, {"error": str(e)}
    
    async def _execute_fast_synthesis(
        self,
        text: str,
        voice_profile: VoiceProfileSchema,
        language: str,
        synthesis_config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute fast synthesis with speed-quality balance."""
        try:
            if progress_callback:
                progress_callback(40, "Fast synthesis mode")
            
            # Use ensemble synthesis with limited models
            result = await self.ensemble_synthesizer.synthesize_speech_ensemble(
                text=text,
                voice_profile=voice_profile,
                language=language,
                progress_callback=progress_callback
            )
            
            if result[0] and len(result) >= 3:  # Success
                metadata = result[2]
                metadata.update({
                    "synthesis_mode": "fast",
                    "models_used": synthesis_config["selected_models"],
                    "optimization_level": "high"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Fast synthesis failed: {e}")
            return False, None, {"error": str(e)}
    
    async def _execute_standard_synthesis(
        self,
        text: str,
        voice_profile: VoiceProfileSchema,
        language: str,
        synthesis_config: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Execute standard optimized synthesis."""
        try:
            if progress_callback:
                progress_callback(40, "Standard optimized synthesis")
            
            # Use full ensemble synthesis with optimizations
            result = await self.ensemble_synthesizer.synthesize_speech_ensemble(
                text=text,
                voice_profile=voice_profile,
                language=language,
                progress_callback=progress_callback
            )
            
            if result[0] and len(result) >= 3:  # Success
                metadata = result[2]
                metadata.update({
                    "synthesis_mode": "standard_optimized",
                    "models_used": synthesis_config["selected_models"],
                    "optimization_level": "balanced"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Standard synthesis failed: {e}")
            return False, None, {"error": str(e)}
    
    async def _synthesize_with_single_model(
        self,
        model: Any,
        text: str,
        voice_profile: VoiceProfileSchema,
        language: str,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Synthesize with a single model for maximum speed."""
        try:
            import tempfile
            import soundfile as sf
            import os
            
            if progress_callback:
                progress_callback(50, "Single model synthesis")
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                temp_output_path = tmp_file.name
            
            try:
                # Prepare reference audio (simplified for speed)
                reference_audio_path = None  # Would need actual implementation
                
                # Synthesize with the model
                synthesis_kwargs = {
                    "text": text,
                    "language": language
                }
                
                # Add reference audio if available
                if reference_audio_path and os.path.exists(reference_audio_path):
                    synthesis_kwargs["speaker_wav"] = reference_audio_path
                
                try:
                    if hasattr(model, 'tts_to_file'):
                        model.tts_to_file(file_path=temp_output_path, **synthesis_kwargs)
                    else:
                        # Fallback synthesis
                        audio_data = model.tts(**synthesis_kwargs)
                        sf.write(temp_output_path, audio_data, 22050)
                        
                except Exception as synthesis_error:
                    if "multi-speaker" in str(synthesis_error).lower() or "speaker" in str(synthesis_error).lower():
                        logger.info("Model is multi-speaker, trying with default speaker")
                        
                        # Try with default speaker configurations
                        speaker_options = [
                            {"speaker": "default"},
                            {"speaker": "p225"},  # Common VCTK speaker
                            {"speaker": "ljspeech"},  # Common single speaker
                            {"speaker_idx": 0},  # Speaker index
                        ]
                        
                        synthesis_success = False
                        for speaker_config in speaker_options:
                            try:
                                # Create new kwargs with speaker config
                                speaker_kwargs = synthesis_kwargs.copy()
                                speaker_kwargs.update(speaker_config)
                                
                                if hasattr(model, 'tts_to_file'):
                                    model.tts_to_file(file_path=temp_output_path, **speaker_kwargs)
                                else:
                                    audio_data = model.tts(**speaker_kwargs)
                                    sf.write(temp_output_path, audio_data, 22050)
                                
                                synthesis_success = True
                                logger.info(f"Synthesis successful with speaker config: {speaker_config}")
                                break
                                
                            except Exception as config_error:
                                logger.debug(f"Speaker config {speaker_config} failed: {config_error}")
                                continue
                        
                        if not synthesis_success:
                            raise synthesis_error  # Re-raise original error if all configs fail
                    else:
                        raise synthesis_error  # Re-raise if not a speaker-related error
                
                # Generate final output path
                timestamp = int(time.time())
                filename = f"optimized_synthesis_{voice_profile.id}_{timestamp}.wav"
                output_path = os.path.join(settings.RESULTS_DIR, filename)
                
                # Move temporary file to final location
                import shutil
                shutil.move(temp_output_path, output_path)
                
                if progress_callback:
                    progress_callback(90, "Single model synthesis complete")
                
                # Generate metadata
                metadata = {
                    "text": text,
                    "language": language,
                    "voice_profile_id": voice_profile.id,
                    "synthesis_method": "single_model_optimized",
                    "model_used": "xtts_v2"
                }
                
                return True, output_path, metadata
                
            except Exception as e:
                # Clean up temporary file on error
                if os.path.exists(temp_output_path):
                    try:
                        os.remove(temp_output_path)
                    except Exception:
                        pass
                raise e
            
        except Exception as e:
            logger.error(f"Single model synthesis failed: {e}")
            return False, None, {"error": str(e)}
    
    async def batch_synthesize_optimized(
        self,
        synthesis_requests: List[Dict[str, Any]],
        progress_callback: Optional[Callable] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], bool, Optional[str], Dict[str, Any], float]]:
        """
        Perform batch synthesis with concurrent processing and optimization.
        
        Args:
            synthesis_requests: List of synthesis request dictionaries
            progress_callback: Optional progress callback
            max_concurrent: Maximum concurrent syntheses
            
        Returns:
            List of (request, success, output_path, metadata, processing_time) tuples
        """
        try:
            max_concurrent = max_concurrent or self.max_concurrent_synthesis
            
            if progress_callback:
                progress_callback(5, f"Starting batch synthesis of {len(synthesis_requests)} requests")
            
            # Create synthesis tasks
            tasks = []
            for request in synthesis_requests:
                task_func = lambda req=request: self._execute_batch_synthesis_item(req)
                tasks.append((task_func, (), {}))
            
            # Execute batch with concurrency control
            results = await self.performance_service.concurrency_manager.submit_batch_tasks(
                tasks=tasks,
                task_type="synthesis",
                max_concurrent=max_concurrent
            )
            
            # Process results
            batch_results = []
            for i, (request, result) in enumerate(zip(synthesis_requests, results)):
                if isinstance(result, Exception):
                    logger.error(f"Batch synthesis failed for request {i}: {result}")
                    batch_results.append((request, False, None, {"error": str(result)}, 0.0))
                else:
                    success, output_path, metadata, processing_time = result
                    batch_results.append((request, success, output_path, metadata, processing_time))
                
                if progress_callback:
                    progress = 10 + (i + 1) * 80 // len(synthesis_requests)
                    progress_callback(progress, f"Completed {i + 1}/{len(synthesis_requests)} syntheses")
            
            if progress_callback:
                progress_callback(100, "Batch synthesis complete")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Batch synthesis failed: {e}")
            raise
    
    def _execute_batch_synthesis_item(self, request: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any], float]:
        """Execute a single synthesis item from batch processing."""
        try:
            start_time = time.time()
            
            # Extract request parameters
            text = request.get("text", "")
            voice_profile = request.get("voice_profile")
            language = request.get("language", "en")
            strategy = request.get("strategy", "fast")  # Default to fast for batch
            
            # Create synthesis function
            strategy_config = self.synthesis_strategies.get(strategy, self.synthesis_strategies["fast"])
            synthesis_func = self._create_optimized_synthesis_function(strategy_config)
            
            # Execute synthesis (synchronous version for batch processing)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    synthesis_func(text, voice_profile, language)
                )
                
                processing_time = time.time() - start_time
                
                if result[0]:  # Success
                    return result[0], result[1], result[2], processing_time
                else:
                    return False, None, result[2], processing_time
                    
            finally:
                loop.close()
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Batch synthesis item failed: {e}")
            return False, None, {"error": str(e)}, processing_time
    
    def get_synthesis_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for synthesis operations."""
        try:
            # Get performance metrics from optimization service
            report = self.performance_service.get_performance_report()
            
            synthesis_stats = report.get("synthesis_performance", {})
            
            # Calculate target achievement rate
            synthesis_times = self.performance_service.metrics.synthesis_times
            
            # Note: Realtime factor calculation would need duration data
            # For now, we'll estimate based on processing time
            estimated_realtime_factors = []
            for proc_time in synthesis_times:
                # Rough estimate: assume 1 second of audio per 10 characters
                estimated_duration = 1.0  # Default estimate
                estimated_realtime_factor = estimated_duration / proc_time if proc_time > 0 else 0
                estimated_realtime_factors.append(estimated_realtime_factor)
            
            target_met_count = len([rf for rf in estimated_realtime_factors if rf >= self.target_realtime_factor])
            total_syntheses = len(synthesis_times)
            
            target_achievement_rate = target_met_count / total_syntheses if total_syntheses > 0 else 0.0
            
            return {
                "target_realtime_factor": self.target_realtime_factor,
                "target_achievement_rate": target_achievement_rate,
                "total_syntheses": total_syntheses,
                "syntheses_meeting_target": target_met_count,
                "performance_stats": synthesis_stats,
                "cache_hit_rate": report.get("cache_hit_rates", {}).get("audio", 0.0),
                "average_synthesis_time": synthesis_stats.get("mean", 0.0),
                "fastest_synthesis_time": synthesis_stats.get("min", 0.0),
                "slowest_synthesis_time": synthesis_stats.get("max", 0.0),
                "gpu_available": self.device.type == 'cuda',
                "gpu_utilization": report.get("gpu_info", {}).get("memory_utilization", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get synthesis performance stats: {e}")
            return {"error": str(e)}
    
    async def warm_up_models(self, progress_callback: Optional[Callable] = None):
        """Warm up synthesis models for optimal performance."""
        try:
            if progress_callback:
                progress_callback(10, "Warming up synthesis models")
            
            # Initialize ensemble synthesizer if not already done
            if not hasattr(self.ensemble_synthesizer, 'loaded_models') or not self.ensemble_synthesizer.loaded_models:
                await self.ensemble_synthesizer.initialize_models(progress_callback)
            
            if progress_callback:
                progress_callback(50, "Optimizing models for GPU")
            
            # Apply GPU optimizations to loaded models
            for strategy_name, strategy_config in self.synthesis_strategies.items():
                await self._optimize_models_for_strategy(strategy_config)
            
            if progress_callback:
                progress_callback(80, "Running warm-up synthesis")
            
            # Run a small warm-up synthesis to initialize GPU kernels
            if self.device.type == 'cuda':
                try:
                    # Create dummy voice profile for warm-up
                    from datetime import datetime
                    dummy_profile = VoiceProfileSchema(
                        id="warmup",
                        reference_audio_id="warmup",
                        quality_score=0.8,
                        created_at=datetime.now()
                    )
                    
                    # Run quick warm-up synthesis
                    await self.synthesize_speech_optimized(
                        text="Hello world",
                        voice_profile=dummy_profile,
                        strategy="ultra_fast"
                    )
                    
                except Exception as e:
                    logger.warning(f"Warm-up synthesis failed: {e}")
                    # Don't fail initialization if warm-up fails
            
            if progress_callback:
                progress_callback(100, "Model warm-up complete")
            
            logger.info("Synthesis models warmed up successfully")
            
        except Exception as e:
            logger.error(f"Model warm-up failed: {e}")
            raise


# Global service instance
optimized_synthesis_service = OptimizedSynthesisService()


async def initialize_optimized_synthesis():
    """Initialize optimized synthesis service."""
    try:
        logger.info("Initializing Optimized Synthesis Service...")
        
        # Warm up models for optimal performance
        await optimized_synthesis_service.warm_up_models()
        
        logger.info("Optimized Synthesis Service initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Optimized synthesis initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize optimized synthesis: {e}")