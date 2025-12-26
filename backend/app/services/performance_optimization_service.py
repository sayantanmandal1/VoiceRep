"""
Performance Optimization Service for high-fidelity voice cloning system.

This module implements GPU optimization, intelligent caching, concurrent processing,
and performance monitoring to achieve faster-than-real-time synthesis and meet
30-second analysis targets for 5-minute audio.
"""

import asyncio
import logging
import time
import threading
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import pickle
from pathlib import Path
import numpy as np
from collections import OrderedDict, defaultdict
import weakref

# GPU and ML optimization imports
import torch
import torch.cuda
from torch.amp import GradScaler

# Internal imports
from app.core.config import settings
from app.schemas.voice import VoiceProfileSchema, VoiceModelSchema

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class GPUConfiguration:
    """GPU optimization configuration."""
    device: torch.device
    memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    enable_tensor_cores: bool = True
    batch_size_multiplier: float = 1.0
    memory_pool_size: Optional[int] = None
    cuda_streams: int = 4


@dataclass
class CacheConfiguration:
    """Intelligent caching configuration."""
    max_voice_profiles: int = 100
    max_audio_cache_mb: int = 1024  # 1GB
    max_model_cache_mb: int = 2048  # 2GB
    ttl_seconds: int = 3600  # 1 hour
    enable_disk_cache: bool = True
    cache_compression: bool = True


@dataclass
class ConcurrencyConfiguration:
    """Concurrent processing configuration."""
    max_concurrent_synthesis: int = 8
    max_concurrent_analysis: int = 4
    thread_pool_size: int = 16
    process_pool_size: int = 4
    queue_timeout: int = 300  # 5 minutes
    enable_priority_queue: bool = True


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    synthesis_times: List[float] = field(default_factory=list)
    analysis_times: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cache_hit_rates: Dict[str, float] = field(default_factory=dict)
    concurrent_tasks: List[int] = field(default_factory=list)
    queue_lengths: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))


class GPUOptimizer:
    """GPU optimization for maximum synthesis speed."""
    
    def __init__(self, config: GPUConfiguration):
        self.config = config
        self.device = config.device
        self.scaler = GradScaler('cuda') if config.enable_mixed_precision and config.device.type == 'cuda' else None
        self.cuda_streams = []
        
        # Initialize GPU optimization
        self._initialize_gpu_optimization()
    
    def _initialize_gpu_optimization(self):
        """Initialize GPU optimization settings."""
        try:
            if self.device.type == 'cuda':
                # Set memory fraction
                if self.config.memory_fraction < 1.0:
                    torch.cuda.set_per_process_memory_fraction(
                        self.config.memory_fraction, 
                        device=self.device.index if self.device.index is not None else 0
                    )
                
                # Enable tensor cores if available
                if self.config.enable_tensor_cores and torch.cuda.get_device_capability()[0] >= 7:
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                
                # Create CUDA streams for parallel processing
                for _ in range(self.config.cuda_streams):
                    stream = torch.cuda.Stream(device=self.device)
                    self.cuda_streams.append(stream)
                
                # Set memory pool size if specified
                if self.config.memory_pool_size:
                    torch.cuda.set_memory_pool_size(self.config.memory_pool_size)
                
                logger.info(f"GPU optimization initialized: {torch.cuda.get_device_name()}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f}GB")
                
            else:
                logger.info("GPU not available, using CPU optimization")
                
        except Exception as e:
            logger.error(f"GPU optimization initialization failed: {e}")
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize model for faster inference."""
        try:
            # Move model to GPU
            model = model.to(self.device)
            
            # Set to evaluation mode
            model.eval()
            
            # Disable gradient computation
            for param in model.parameters():
                param.requires_grad = False
            
            # Apply torch.jit compilation if possible
            try:
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 80, 100).to(self.device)  # Typical mel-spectrogram shape
                model = torch.jit.trace(model, dummy_input)
                logger.info("Model successfully compiled with TorchScript")
            except Exception as e:
                logger.warning(f"TorchScript compilation failed: {e}")
            
            # Apply additional optimizations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def get_optimal_batch_size(self, model_size_mb: float, input_size: Tuple[int, ...]) -> int:
        """Calculate optimal batch size based on GPU memory."""
        try:
            if self.device.type != 'cuda':
                return 1
            
            # Get available GPU memory
            available_memory = torch.cuda.get_device_properties(self.device).total_memory
            available_memory *= self.config.memory_fraction
            
            # Estimate memory per sample (rough calculation)
            input_memory = np.prod(input_size) * 4  # 4 bytes per float32
            model_memory = model_size_mb * 1024 * 1024
            
            # Conservative estimate: 3x memory overhead for forward pass
            memory_per_sample = (input_memory + model_memory) * 3
            
            # Calculate batch size
            batch_size = max(1, int(available_memory / memory_per_sample))
            batch_size = min(batch_size, 32)  # Cap at reasonable maximum
            
            # Apply multiplier from config
            batch_size = max(1, int(batch_size * self.config.batch_size_multiplier))
            
            return batch_size
            
        except Exception as e:
            logger.error(f"Batch size calculation failed: {e}")
            return 1
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache."""
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
        except Exception as e:
            logger.error(f"GPU cache clearing failed: {e}")
    
    def get_gpu_utilization(self) -> Dict[str, float]:
        """Get current GPU utilization metrics."""
        try:
            if self.device.type != 'cuda':
                return {"gpu_available": False}
            
            # Get GPU memory info
            memory_allocated = torch.cuda.memory_allocated(self.device)
            memory_reserved = torch.cuda.memory_reserved(self.device)
            memory_total = torch.cuda.get_device_properties(self.device).total_memory
            
            return {
                "gpu_available": True,
                "memory_allocated_mb": memory_allocated / 1024 / 1024,
                "memory_reserved_mb": memory_reserved / 1024 / 1024,
                "memory_total_mb": memory_total / 1024 / 1024,
                "memory_utilization": memory_allocated / memory_total,
                "device_name": torch.cuda.get_device_name(self.device)
            }
            
        except Exception as e:
            logger.error(f"GPU utilization check failed: {e}")
            return {"gpu_available": False, "error": str(e)}


class IntelligentCache:
    """Intelligent caching system for repeated voice profiles and models."""
    
    def __init__(self, config: CacheConfiguration):
        self.config = config
        self.voice_profile_cache = OrderedDict()
        self.audio_cache = OrderedDict()
        self.model_cache = OrderedDict()
        self.cache_stats = defaultdict(int)
        self.cache_lock = threading.RLock()
        
        # Initialize disk cache if enabled
        if config.enable_disk_cache:
            self.disk_cache_dir = Path(settings.MODELS_DIR) / "cache"
            self.disk_cache_dir.mkdir(exist_ok=True)
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, (str, int, float)):
            return str(data)
        
        # For complex objects, use hash of serialized data
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def cache_voice_profile(self, profile: VoiceProfileSchema, analysis_data: Dict[str, Any]):
        """Cache voice profile analysis data."""
        try:
            with self.cache_lock:
                cache_key = self._generate_cache_key(profile.id)
                
                # Prepare cache entry
                cache_entry = {
                    "profile": profile,
                    "analysis_data": analysis_data,
                    "timestamp": time.time(),
                    "access_count": 0
                }
                
                # Add to memory cache
                self.voice_profile_cache[cache_key] = cache_entry
                
                # Enforce size limits
                self._enforce_voice_profile_cache_limits()
                
                # Save to disk cache if enabled
                if self.config.enable_disk_cache:
                    self._save_to_disk_cache("voice_profiles", cache_key, cache_entry)
                
                self.cache_stats["voice_profile_stores"] += 1
                logger.debug(f"Cached voice profile: {profile.id}")
                
        except Exception as e:
            logger.error(f"Voice profile caching failed: {e}")
    
    def get_cached_voice_profile(self, profile_id: str) -> Optional[Tuple[VoiceProfileSchema, Dict[str, Any]]]:
        """Retrieve cached voice profile analysis data."""
        try:
            with self.cache_lock:
                cache_key = self._generate_cache_key(profile_id)
                
                # Check memory cache first
                if cache_key in self.voice_profile_cache:
                    entry = self.voice_profile_cache[cache_key]
                    
                    # Check TTL
                    if time.time() - entry["timestamp"] < self.config.ttl_seconds:
                        entry["access_count"] += 1
                        # Move to end (LRU)
                        self.voice_profile_cache.move_to_end(cache_key)
                        
                        self.cache_stats["voice_profile_hits"] += 1
                        return entry["profile"], entry["analysis_data"]
                    else:
                        # Expired, remove from cache
                        del self.voice_profile_cache[cache_key]
                
                # Check disk cache if enabled
                if self.config.enable_disk_cache:
                    disk_entry = self._load_from_disk_cache("voice_profiles", cache_key)
                    if disk_entry and time.time() - disk_entry["timestamp"] < self.config.ttl_seconds:
                        # Restore to memory cache
                        self.voice_profile_cache[cache_key] = disk_entry
                        self.cache_stats["voice_profile_disk_hits"] += 1
                        return disk_entry["profile"], disk_entry["analysis_data"]
                
                self.cache_stats["voice_profile_misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Voice profile cache retrieval failed: {e}")
            return None
    
    def cache_audio_data(self, audio_id: str, audio_data: np.ndarray, metadata: Dict[str, Any]):
        """Cache processed audio data."""
        try:
            with self.cache_lock:
                cache_key = self._generate_cache_key(audio_id)
                
                # Calculate memory usage
                audio_size_mb = audio_data.nbytes / 1024 / 1024
                
                # Check if audio fits in cache
                if audio_size_mb > self.config.max_audio_cache_mb / 2:
                    logger.warning(f"Audio too large for cache: {audio_size_mb:.1f}MB")
                    return
                
                cache_entry = {
                    "audio_data": audio_data,
                    "metadata": metadata,
                    "timestamp": time.time(),
                    "size_mb": audio_size_mb,
                    "access_count": 0
                }
                
                self.audio_cache[cache_key] = cache_entry
                self._enforce_audio_cache_limits()
                
                self.cache_stats["audio_stores"] += 1
                logger.debug(f"Cached audio data: {audio_id} ({audio_size_mb:.1f}MB)")
                
        except Exception as e:
            logger.error(f"Audio caching failed: {e}")
    
    def get_cached_audio_data(self, audio_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve cached audio data."""
        try:
            with self.cache_lock:
                cache_key = self._generate_cache_key(audio_id)
                
                if cache_key in self.audio_cache:
                    entry = self.audio_cache[cache_key]
                    
                    # Check TTL
                    if time.time() - entry["timestamp"] < self.config.ttl_seconds:
                        entry["access_count"] += 1
                        self.audio_cache.move_to_end(cache_key)
                        
                        self.cache_stats["audio_hits"] += 1
                        return entry["audio_data"], entry["metadata"]
                    else:
                        del self.audio_cache[cache_key]
                
                self.cache_stats["audio_misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Audio cache retrieval failed: {e}")
            return None
    
    def cache_model(self, model_id: str, model_data: Any, model_size_mb: float):
        """Cache model data."""
        try:
            with self.cache_lock:
                cache_key = self._generate_cache_key(model_id)
                
                # Check if model fits in cache
                if model_size_mb > self.config.max_model_cache_mb / 2:
                    logger.warning(f"Model too large for cache: {model_size_mb:.1f}MB")
                    return
                
                cache_entry = {
                    "model_data": model_data,
                    "timestamp": time.time(),
                    "size_mb": model_size_mb,
                    "access_count": 0
                }
                
                self.model_cache[cache_key] = cache_entry
                self._enforce_model_cache_limits()
                
                self.cache_stats["model_stores"] += 1
                logger.debug(f"Cached model: {model_id} ({model_size_mb:.1f}MB)")
                
        except Exception as e:
            logger.error(f"Model caching failed: {e}")
    
    def get_cached_model(self, model_id: str) -> Optional[Any]:
        """Retrieve cached model data."""
        try:
            with self.cache_lock:
                cache_key = self._generate_cache_key(model_id)
                
                if cache_key in self.model_cache:
                    entry = self.model_cache[cache_key]
                    
                    # Check TTL
                    if time.time() - entry["timestamp"] < self.config.ttl_seconds:
                        entry["access_count"] += 1
                        self.model_cache.move_to_end(cache_key)
                        
                        self.cache_stats["model_hits"] += 1
                        return entry["model_data"]
                    else:
                        del self.model_cache[cache_key]
                
                self.cache_stats["model_misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Model cache retrieval failed: {e}")
            return None
    
    def _enforce_voice_profile_cache_limits(self):
        """Enforce voice profile cache size limits."""
        while len(self.voice_profile_cache) > self.config.max_voice_profiles:
            # Remove least recently used
            self.voice_profile_cache.popitem(last=False)
    
    def _enforce_audio_cache_limits(self):
        """Enforce audio cache size limits."""
        total_size = sum(entry["size_mb"] for entry in self.audio_cache.values())
        
        while total_size > self.config.max_audio_cache_mb and self.audio_cache:
            # Remove least recently used
            _, removed_entry = self.audio_cache.popitem(last=False)
            total_size -= removed_entry["size_mb"]
    
    def _enforce_model_cache_limits(self):
        """Enforce model cache size limits."""
        total_size = sum(entry["size_mb"] for entry in self.model_cache.values())
        
        while total_size > self.config.max_model_cache_mb and self.model_cache:
            # Remove least recently used
            _, removed_entry = self.model_cache.popitem(last=False)
            total_size -= removed_entry["size_mb"]
    
    def _save_to_disk_cache(self, cache_type: str, cache_key: str, data: Any):
        """Save data to disk cache."""
        try:
            if not self.config.enable_disk_cache:
                return
            
            cache_dir = self.disk_cache_dir / cache_type
            cache_dir.mkdir(exist_ok=True)
            
            cache_file = cache_dir / f"{cache_key}.pkl"
            
            with open(cache_file, 'wb') as f:
                if self.config.cache_compression:
                    import gzip
                    with gzip.open(f, 'wb') as gz_f:
                        pickle.dump(data, gz_f)
                else:
                    pickle.dump(data, f)
                    
        except Exception as e:
            logger.error(f"Disk cache save failed: {e}")
    
    def _load_from_disk_cache(self, cache_type: str, cache_key: str) -> Optional[Any]:
        """Load data from disk cache."""
        try:
            if not self.config.enable_disk_cache:
                return None
            
            cache_file = self.disk_cache_dir / cache_type / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                if self.config.cache_compression:
                    import gzip
                    with gzip.open(f, 'rb') as gz_f:
                        return pickle.load(gz_f)
                else:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Disk cache load failed: {e}")
            return None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.cache_lock:
            stats = dict(self.cache_stats)
            
            # Calculate hit rates
            for cache_type in ["voice_profile", "audio", "model"]:
                hits = stats.get(f"{cache_type}_hits", 0)
                misses = stats.get(f"{cache_type}_misses", 0)
                total = hits + misses
                
                if total > 0:
                    stats[f"{cache_type}_hit_rate"] = hits / total
                else:
                    stats[f"{cache_type}_hit_rate"] = 0.0
            
            # Add cache sizes
            stats["voice_profile_cache_size"] = len(self.voice_profile_cache)
            stats["audio_cache_size"] = len(self.audio_cache)
            stats["model_cache_size"] = len(self.model_cache)
            
            # Add memory usage
            audio_memory_mb = sum(entry["size_mb"] for entry in self.audio_cache.values())
            model_memory_mb = sum(entry["size_mb"] for entry in self.model_cache.values())
            
            stats["audio_cache_memory_mb"] = audio_memory_mb
            stats["model_cache_memory_mb"] = model_memory_mb
            
            return stats
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache data."""
        with self.cache_lock:
            if cache_type == "voice_profiles" or cache_type is None:
                self.voice_profile_cache.clear()
            
            if cache_type == "audio" or cache_type is None:
                self.audio_cache.clear()
            
            if cache_type == "models" or cache_type is None:
                self.model_cache.clear()
            
            logger.info(f"Cleared cache: {cache_type or 'all'}")


class ConcurrentProcessingManager:
    """Concurrent request handling with quality maintenance."""
    
    def __init__(self, config: ConcurrencyConfiguration):
        self.config = config
        
        # Thread pools for different types of work
        self.synthesis_executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_synthesis,
            thread_name_prefix="synthesis"
        )
        
        self.analysis_executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_analysis,
            thread_name_prefix="analysis"
        )
        
        self.general_executor = ThreadPoolExecutor(
            max_workers=config.thread_pool_size,
            thread_name_prefix="general"
        )
        
        # Process pool for CPU-intensive tasks
        self.process_executor = ProcessPoolExecutor(
            max_workers=config.process_pool_size
        )
        
        # Task queues with priority support
        self.synthesis_queue = asyncio.PriorityQueue()
        self.analysis_queue = asyncio.PriorityQueue()
        
        # Active task tracking
        self.active_tasks = {}
        self.task_lock = threading.Lock()
        
        # Performance monitoring
        self.queue_metrics = defaultdict(list)
        
    async def submit_synthesis_task(
        self, 
        task_func: Callable, 
        *args, 
        priority: int = 5,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Submit synthesis task with priority and timeout."""
        try:
            task_id = f"synthesis_{int(time.time() * 1000000)}"
            
            # Create task wrapper
            async def task_wrapper():
                start_time = time.time()
                
                with self.task_lock:
                    self.active_tasks[task_id] = {
                        "type": "synthesis",
                        "start_time": start_time,
                        "priority": priority
                    }
                
                try:
                    # Execute task in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.synthesis_executor,
                        lambda: task_func(*args, **kwargs)
                    )
                    
                    processing_time = time.time() - start_time
                    logger.debug(f"Synthesis task {task_id} completed in {processing_time:.2f}s")
                    
                    return result
                    
                finally:
                    with self.task_lock:
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]
            
            # Apply timeout if specified
            if timeout:
                return await asyncio.wait_for(task_wrapper(), timeout=timeout)
            else:
                return await task_wrapper()
                
        except asyncio.TimeoutError:
            logger.error(f"Synthesis task {task_id} timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Synthesis task {task_id} failed: {e}")
            raise
    
    async def submit_analysis_task(
        self, 
        task_func: Callable, 
        *args, 
        priority: int = 5,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Submit analysis task with priority and timeout."""
        try:
            task_id = f"analysis_{int(time.time() * 1000000)}"
            
            async def task_wrapper():
                start_time = time.time()
                
                with self.task_lock:
                    self.active_tasks[task_id] = {
                        "type": "analysis",
                        "start_time": start_time,
                        "priority": priority
                    }
                
                try:
                    # Execute task in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.analysis_executor,
                        lambda: task_func(*args, **kwargs)
                    )
                    
                    processing_time = time.time() - start_time
                    logger.debug(f"Analysis task {task_id} completed in {processing_time:.2f}s")
                    
                    return result
                    
                finally:
                    with self.task_lock:
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]
            
            # Apply timeout if specified
            if timeout:
                return await asyncio.wait_for(task_wrapper(), timeout=timeout)
            else:
                return await task_wrapper()
                
        except asyncio.TimeoutError:
            logger.error(f"Analysis task {task_id} timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Analysis task {task_id} failed: {e}")
            raise
    
    async def submit_batch_tasks(
        self, 
        tasks: List[Tuple[Callable, tuple, dict]], 
        task_type: str = "general",
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Submit batch of tasks for concurrent execution."""
        try:
            if max_concurrent is None:
                max_concurrent = self.config.thread_pool_size
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def execute_task(task_func, args, kwargs):
                async with semaphore:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        self.general_executor,
                        lambda: task_func(*args, **kwargs)
                    )
            
            # Execute all tasks concurrently
            task_coroutines = [
                execute_task(task_func, args, kwargs)
                for task_func, args, kwargs in tasks
            ]
            
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch task {i} failed: {result}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch task execution failed: {e}")
            raise
    
    def get_active_task_count(self) -> Dict[str, int]:
        """Get count of active tasks by type."""
        with self.task_lock:
            task_counts = defaultdict(int)
            
            for task_info in self.active_tasks.values():
                task_counts[task_info["type"]] += 1
            
            task_counts["total"] = len(self.active_tasks)
            
            return dict(task_counts)
    
    def get_executor_status(self) -> Dict[str, Any]:
        """Get status of thread/process executors."""
        return {
            "synthesis_executor": {
                "max_workers": self.config.max_concurrent_synthesis,
                "active_threads": getattr(self.synthesis_executor, '_threads', 0)
            },
            "analysis_executor": {
                "max_workers": self.config.max_concurrent_analysis,
                "active_threads": getattr(self.analysis_executor, '_threads', 0)
            },
            "general_executor": {
                "max_workers": self.config.thread_pool_size,
                "active_threads": getattr(self.general_executor, '_threads', 0)
            },
            "process_executor": {
                "max_workers": self.config.process_pool_size,
                "active_processes": len(getattr(self.process_executor, '_processes', {}))
            }
        }
    
    def shutdown(self):
        """Shutdown all executors."""
        try:
            self.synthesis_executor.shutdown(wait=True)
            self.analysis_executor.shutdown(wait=True)
            self.general_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            logger.info("All executors shut down successfully")
            
        except Exception as e:
            logger.error(f"Executor shutdown failed: {e}")


class PerformanceOptimizationService:
    """Main performance optimization service coordinating all optimization components."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        
        # Initialize configurations based on optimization level
        self.gpu_config = self._create_gpu_config()
        self.cache_config = self._create_cache_config()
        self.concurrency_config = self._create_concurrency_config()
        
        # Initialize components
        self.gpu_optimizer = GPUOptimizer(self.gpu_config)
        self.cache = IntelligentCache(self.cache_config)
        self.concurrency_manager = ConcurrentProcessingManager(self.concurrency_config)
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.monitoring_task = None
        
        logger.info(f"Performance optimization service initialized with {optimization_level.value} level")
    
    def _create_gpu_config(self) -> GPUConfiguration:
        """Create GPU configuration based on optimization level."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        configs = {
            OptimizationLevel.CONSERVATIVE: GPUConfiguration(
                device=device,
                memory_fraction=0.6,
                enable_mixed_precision=False,
                enable_tensor_cores=False,
                batch_size_multiplier=0.8,
                cuda_streams=2
            ),
            OptimizationLevel.BALANCED: GPUConfiguration(
                device=device,
                memory_fraction=0.8,
                enable_mixed_precision=True,
                enable_tensor_cores=True,
                batch_size_multiplier=1.0,
                cuda_streams=4
            ),
            OptimizationLevel.AGGRESSIVE: GPUConfiguration(
                device=device,
                memory_fraction=0.9,
                enable_mixed_precision=True,
                enable_tensor_cores=True,
                batch_size_multiplier=1.2,
                cuda_streams=6
            ),
            OptimizationLevel.MAXIMUM: GPUConfiguration(
                device=device,
                memory_fraction=0.95,
                enable_mixed_precision=True,
                enable_tensor_cores=True,
                batch_size_multiplier=1.5,
                cuda_streams=8
            )
        }
        
        return configs[self.optimization_level]
    
    def _create_cache_config(self) -> CacheConfiguration:
        """Create cache configuration based on optimization level."""
        configs = {
            OptimizationLevel.CONSERVATIVE: CacheConfiguration(
                max_voice_profiles=50,
                max_audio_cache_mb=512,
                max_model_cache_mb=1024,
                ttl_seconds=1800,  # 30 minutes
                enable_disk_cache=False,
                cache_compression=True
            ),
            OptimizationLevel.BALANCED: CacheConfiguration(
                max_voice_profiles=100,
                max_audio_cache_mb=1024,
                max_model_cache_mb=2048,
                ttl_seconds=3600,  # 1 hour
                enable_disk_cache=True,
                cache_compression=True
            ),
            OptimizationLevel.AGGRESSIVE: CacheConfiguration(
                max_voice_profiles=200,
                max_audio_cache_mb=2048,
                max_model_cache_mb=4096,
                ttl_seconds=7200,  # 2 hours
                enable_disk_cache=True,
                cache_compression=False
            ),
            OptimizationLevel.MAXIMUM: CacheConfiguration(
                max_voice_profiles=500,
                max_audio_cache_mb=4096,
                max_model_cache_mb=8192,
                ttl_seconds=14400,  # 4 hours
                enable_disk_cache=True,
                cache_compression=False
            )
        }
        
        return configs[self.optimization_level]
    
    def _create_concurrency_config(self) -> ConcurrencyConfiguration:
        """Create concurrency configuration based on optimization level."""
        cpu_count = psutil.cpu_count()
        
        configs = {
            OptimizationLevel.CONSERVATIVE: ConcurrencyConfiguration(
                max_concurrent_synthesis=2,
                max_concurrent_analysis=2,
                thread_pool_size=max(4, cpu_count // 2),
                process_pool_size=max(2, cpu_count // 4),
                queue_timeout=300,
                enable_priority_queue=False
            ),
            OptimizationLevel.BALANCED: ConcurrencyConfiguration(
                max_concurrent_synthesis=4,
                max_concurrent_analysis=4,
                thread_pool_size=max(8, cpu_count),
                process_pool_size=max(4, cpu_count // 2),
                queue_timeout=300,
                enable_priority_queue=True
            ),
            OptimizationLevel.AGGRESSIVE: ConcurrencyConfiguration(
                max_concurrent_synthesis=8,
                max_concurrent_analysis=6,
                thread_pool_size=max(16, cpu_count * 2),
                process_pool_size=max(6, cpu_count),
                queue_timeout=600,
                enable_priority_queue=True
            ),
            OptimizationLevel.MAXIMUM: ConcurrencyConfiguration(
                max_concurrent_synthesis=12,
                max_concurrent_analysis=8,
                thread_pool_size=max(24, cpu_count * 3),
                process_pool_size=max(8, cpu_count),
                queue_timeout=900,
                enable_priority_queue=True
            )
        }
        
        return configs[self.optimization_level]
    
    async def optimize_voice_analysis(
        self, 
        analysis_func: Callable, 
        audio_path: str,
        target_time_seconds: float = 30.0,
        **kwargs
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize voice analysis to meet 30-second target for 5-minute audio."""
        start_time = time.time()
        
        try:
            # Check cache first
            audio_id = f"analysis_{hashlib.md5(audio_path.encode()).hexdigest()}"
            cached_result = self.cache.get_cached_voice_profile(audio_id)
            
            if cached_result:
                profile, analysis_data = cached_result
                processing_time = time.time() - start_time
                logger.info(f"Voice analysis cache hit: {processing_time:.3f}s")
                return analysis_data, processing_time
            
            # Submit analysis task with timeout
            timeout = max(target_time_seconds, 30)  # Minimum 30 seconds
            
            result = await self.concurrency_manager.submit_analysis_task(
                analysis_func,
                audio_path,
                timeout=timeout,
                priority=1,  # High priority
                **kwargs
            )
            
            processing_time = time.time() - start_time
            
            # Cache result if successful
            if result and isinstance(result, dict):
                # Create dummy profile for caching
                from app.schemas.voice import VoiceProfileSchema
                profile = VoiceProfileSchema(
                    id=audio_id,
                    reference_audio_id=audio_id,
                    quality_score=result.get("quality_score", 0.8),
                    created_at=time.time()
                )
                
                self.cache.cache_voice_profile(profile, result)
            
            # Record metrics
            self.metrics.analysis_times.append(processing_time)
            
            # Check if target time was met
            if processing_time <= target_time_seconds:
                logger.info(f"Voice analysis completed within target: {processing_time:.2f}s <= {target_time_seconds}s")
            else:
                logger.warning(f"Voice analysis exceeded target: {processing_time:.2f}s > {target_time_seconds}s")
            
            return result, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Voice analysis optimization failed after {processing_time:.2f}s: {e}")
            raise
    
    async def optimize_speech_synthesis(
        self, 
        synthesis_func: Callable,
        text: str,
        voice_profile: VoiceProfileSchema,
        target_realtime_factor: float = 2.0,  # 2x faster than real-time
        **kwargs
    ) -> Tuple[Any, float, float]:
        """Optimize speech synthesis for faster-than-real-time generation."""
        start_time = time.time()
        
        try:
            # Estimate target audio duration
            estimated_duration = len(text) * 0.08  # ~80ms per character (rough estimate)
            target_synthesis_time = estimated_duration / target_realtime_factor
            
            # Check cache first
            synthesis_id = f"synthesis_{hashlib.md5((text + voice_profile.id).encode()).hexdigest()}"
            cached_audio = self.cache.get_cached_audio_data(synthesis_id)
            
            if cached_audio:
                audio_data, metadata = cached_audio
                processing_time = time.time() - start_time
                actual_duration = metadata.get("duration", estimated_duration)
                realtime_factor = actual_duration / processing_time if processing_time > 0 else float('inf')
                
                logger.info(f"Speech synthesis cache hit: {processing_time:.3f}s, {realtime_factor:.1f}x real-time")
                return (True, None, metadata), processing_time, realtime_factor
            
            # Submit synthesis task with timeout
            timeout = max(target_synthesis_time * 3, 60)  # Allow 3x target time, minimum 60s
            
            # Remove priority from kwargs to avoid duplicate parameter
            synthesis_kwargs = {k: v for k, v in kwargs.items() if k != 'priority'}
            
            result = await self.concurrency_manager.submit_synthesis_task(
                synthesis_func,
                text,
                voice_profile,
                timeout=timeout,
                priority=2,  # Medium-high priority
                **synthesis_kwargs
            )
            
            processing_time = time.time() - start_time
            
            # Calculate actual realtime factor
            if result and len(result) >= 3 and result[2]:
                actual_duration = result[2].get("duration", estimated_duration)
            else:
                actual_duration = estimated_duration
            
            realtime_factor = actual_duration / processing_time if processing_time > 0 else float('inf')
            
            # Cache result if successful
            if result and result[0] and result[1]:  # success and output_path
                try:
                    # Load and cache audio data
                    import librosa
                    audio_data, sr = librosa.load(result[1], sr=22050)
                    metadata = result[2] if len(result) >= 3 else {}
                    metadata["duration"] = len(audio_data) / sr
                    
                    self.cache.cache_audio_data(synthesis_id, audio_data, metadata)
                except Exception as e:
                    logger.warning(f"Failed to cache synthesis result: {e}")
            
            # Record metrics
            self.metrics.synthesis_times.append(processing_time)
            
            # Check if target realtime factor was met
            if realtime_factor >= target_realtime_factor:
                logger.info(f"Speech synthesis achieved target: {realtime_factor:.1f}x >= {target_realtime_factor}x real-time")
            else:
                logger.warning(f"Speech synthesis below target: {realtime_factor:.1f}x < {target_realtime_factor}x real-time")
            
            return result, processing_time, realtime_factor
            
        except Exception as e:
            processing_time = time.time() - start_time
            realtime_factor = 0.0
            logger.error(f"Speech synthesis optimization failed after {processing_time:.2f}s: {e}")
            raise
    
    def start_performance_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        async def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Collect GPU metrics
                    gpu_metrics = self.gpu_optimizer.get_gpu_utilization()
                    if gpu_metrics.get("gpu_available"):
                        self.metrics.gpu_utilization.append(gpu_metrics["memory_utilization"])
                    
                    # Collect system memory metrics
                    memory_info = psutil.virtual_memory()
                    self.metrics.memory_usage.append(memory_info.percent / 100.0)
                    
                    # Collect cache metrics
                    cache_stats = self.cache.get_cache_statistics()
                    for cache_type in ["voice_profile", "audio", "model"]:
                        hit_rate_key = f"{cache_type}_hit_rate"
                        if hit_rate_key in cache_stats:
                            self.metrics.cache_hit_rates[cache_type] = cache_stats[hit_rate_key]
                    
                    # Collect concurrency metrics
                    active_tasks = self.concurrency_manager.get_active_task_count()
                    self.metrics.concurrent_tasks.append(active_tasks.get("total", 0))
                    
                    # Limit metrics history
                    max_history = 1000
                    for metric_list in [
                        self.metrics.synthesis_times,
                        self.metrics.analysis_times,
                        self.metrics.gpu_utilization,
                        self.metrics.memory_usage,
                        self.metrics.concurrent_tasks
                    ]:
                        if len(metric_list) > max_history:
                            metric_list[:] = metric_list[-max_history:]
                    
                    await asyncio.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    await asyncio.sleep(10)  # Wait longer on error
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info("Performance monitoring started")
    
    def stop_performance_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        
        logger.info("Performance monitoring stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            # Calculate statistics
            def calc_stats(values):
                if not values:
                    return {"count": 0, "mean": 0, "min": 0, "max": 0}
                
                return {
                    "count": len(values),
                    "mean": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values)
                }
            
            report = {
                "optimization_level": self.optimization_level.value,
                "timestamp": time.time(),
                
                # Performance metrics
                "synthesis_performance": calc_stats(self.metrics.synthesis_times),
                "analysis_performance": calc_stats(self.metrics.analysis_times),
                
                # Resource utilization
                "gpu_utilization": calc_stats(self.metrics.gpu_utilization),
                "memory_utilization": calc_stats(self.metrics.memory_usage),
                "concurrent_tasks": calc_stats(self.metrics.concurrent_tasks),
                
                # Cache performance
                "cache_hit_rates": dict(self.metrics.cache_hit_rates),
                "cache_statistics": self.cache.get_cache_statistics(),
                
                # System information
                "gpu_info": self.gpu_optimizer.get_gpu_utilization(),
                "executor_status": self.concurrency_manager.get_executor_status(),
                
                # Performance targets
                "targets": {
                    "analysis_time_target_seconds": 30.0,
                    "synthesis_realtime_factor_target": 2.0,
                    "analysis_time_achieved": len([t for t in self.metrics.analysis_times if t <= 30.0]) / max(1, len(self.metrics.analysis_times)),
                    "synthesis_realtime_factor_achieved": "calculated_per_synthesis"  # Would need duration data
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def optimize_for_target_performance(self, target_analysis_time: float = 30.0, target_realtime_factor: float = 2.0):
        """Dynamically optimize configuration for target performance."""
        try:
            # Analyze recent performance
            recent_analysis_times = self.metrics.analysis_times[-10:] if self.metrics.analysis_times else []
            recent_synthesis_times = self.metrics.synthesis_times[-10:] if self.metrics.synthesis_times else []
            
            # Check if we need to adjust optimization level
            analysis_performance_ok = True
            synthesis_performance_ok = True
            
            if recent_analysis_times:
                avg_analysis_time = np.mean(recent_analysis_times)
                analysis_performance_ok = avg_analysis_time <= target_analysis_time
            
            # Note: Synthesis realtime factor would need duration data to calculate properly
            
            # Adjust optimization level if needed
            current_level_index = list(OptimizationLevel).index(self.optimization_level)
            
            if not analysis_performance_ok and current_level_index < len(OptimizationLevel) - 1:
                # Increase optimization level
                new_level = list(OptimizationLevel)[current_level_index + 1]
                logger.info(f"Increasing optimization level to {new_level.value} for better performance")
                self._update_optimization_level(new_level)
            
            elif analysis_performance_ok and synthesis_performance_ok and current_level_index > 0:
                # Consider decreasing optimization level for stability
                new_level = list(OptimizationLevel)[current_level_index - 1]
                logger.info(f"Considering optimization level decrease to {new_level.value} for stability")
            
        except Exception as e:
            logger.error(f"Performance optimization adjustment failed: {e}")
    
    def _update_optimization_level(self, new_level: OptimizationLevel):
        """Update optimization level and reconfigure components."""
        try:
            self.optimization_level = new_level
            
            # Update configurations
            self.gpu_config = self._create_gpu_config()
            self.cache_config = self._create_cache_config()
            self.concurrency_config = self._create_concurrency_config()
            
            # Reinitialize GPU optimizer
            self.gpu_optimizer = GPUOptimizer(self.gpu_config)
            
            # Update cache limits
            self.cache.config = self.cache_config
            
            # Note: Concurrency manager would need restart to apply new config
            # This is a simplified implementation
            
            logger.info(f"Optimization level updated to {new_level.value}")
            
        except Exception as e:
            logger.error(f"Optimization level update failed: {e}")
    
    def cleanup_resources(self):
        """Cleanup resources and shutdown components."""
        try:
            # Stop monitoring
            self.stop_performance_monitoring()
            
            # Clear GPU cache
            self.gpu_optimizer.clear_gpu_cache()
            
            # Clear caches
            self.cache.clear_cache()
            
            # Shutdown concurrency manager
            self.concurrency_manager.shutdown()
            
            logger.info("Performance optimization service cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")


# Global service instance
performance_optimization_service = PerformanceOptimizationService()


async def initialize_performance_optimization():
    """Initialize performance optimization service."""
    try:
        logger.info("Initializing Performance Optimization Service...")
        
        # Start performance monitoring
        performance_optimization_service.start_performance_monitoring()
        
        logger.info("Performance Optimization Service initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Performance optimization initialization failed: {e}")
        raise RuntimeError(f"Failed to initialize performance optimization: {e}")