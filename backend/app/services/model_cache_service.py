"""
Advanced model caching service for optimized AI model loading and inference.
"""

import os
import time
import json
import pickle
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import weakref

from app.core.config import settings
from app.services.performance_monitoring_service import performance_monitor

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """Cached model data structure."""
    model_id: str
    model_data: Any
    model_type: str
    size_bytes: int
    load_time: float
    last_accessed: datetime
    access_count: int
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_load_time: float = 0.0
    average_load_time: float = 0.0
    cache_size_bytes: int = 0
    cache_utilization: float = 0.0


class LRUCache:
    """Thread-safe LRU cache implementation for models."""
    
    def __init__(self, max_size_bytes: int = 2 * 1024 * 1024 * 1024):  # 2GB default
        self.max_size_bytes = max_size_bytes
        self.cache: OrderedDict[str, CachedModel] = OrderedDict()
        self.current_size_bytes = 0
        self._lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[CachedModel]:
        """Get model from cache."""
        with self._lock:
            self.stats.total_requests += 1
            
            if key in self.cache:
                # Move to end (most recently used)
                cached_model = self.cache.pop(key)
                self.cache[key] = cached_model
                
                # Update access statistics
                cached_model.last_accessed = datetime.now()
                cached_model.access_count += 1
                
                self.stats.cache_hits += 1
                return cached_model
            else:
                self.stats.cache_misses += 1
                return None
    
    def put(self, key: str, cached_model: CachedModel) -> bool:
        """Put model in cache with eviction if necessary."""
        with self._lock:
            # Remove existing entry if present
            if key in self.cache:
                old_model = self.cache.pop(key)
                self.current_size_bytes -= old_model.size_bytes
            
            # Check if we need to evict
            while (self.current_size_bytes + cached_model.size_bytes > self.max_size_bytes 
                   and self.cache):
                self._evict_lru()
            
            # Add new model if it fits
            if cached_model.size_bytes <= self.max_size_bytes:
                self.cache[key] = cached_model
                self.current_size_bytes += cached_model.size_bytes
                return True
            else:
                logger.warning(f"Model {key} too large for cache: {cached_model.size_bytes} bytes")
                return False
    
    def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if self.cache:
            key, cached_model = self.cache.popitem(last=False)  # Remove first (oldest)
            self.current_size_bytes -= cached_model.size_bytes
            self.stats.evictions += 1
            logger.info(f"Evicted model {key} from cache")
    
    def remove(self, key: str) -> bool:
        """Remove specific model from cache."""
        with self._lock:
            if key in self.cache:
                cached_model = self.cache.pop(key)
                self.current_size_bytes -= cached_model.size_bytes
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            stats = self.stats
            stats.cache_size_bytes = self.current_size_bytes
            stats.cache_utilization = self.current_size_bytes / self.max_size_bytes
            
            if stats.cache_misses > 0:
                stats.average_load_time = stats.total_load_time / stats.cache_misses
            
            return stats
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            models_info = []
            for key, cached_model in self.cache.items():
                models_info.append({
                    'model_id': cached_model.model_id,
                    'model_type': cached_model.model_type,
                    'size_mb': cached_model.size_bytes / (1024 * 1024),
                    'load_time': cached_model.load_time,
                    'last_accessed': cached_model.last_accessed.isoformat(),
                    'access_count': cached_model.access_count,
                    'quality_score': cached_model.quality_score
                })
            
            return {
                'cache_stats': self.get_stats().__dict__,
                'cached_models': models_info,
                'cache_size_mb': self.current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'model_count': len(self.cache)
            }


class ModelCacheService:
    """Advanced model caching service with optimization strategies."""
    
    def __init__(self):
        # Initialize cache with configurable size
        cache_size_gb = getattr(settings, 'MODEL_CACHE_SIZE_GB', 2)
        self.cache = LRUCache(max_size_bytes=cache_size_gb * 1024 * 1024 * 1024)
        
        # Preloading configuration
        self.preload_enabled = getattr(settings, 'ENABLE_MODEL_PRELOADING', True)
        self.preload_popular_models = getattr(settings, 'PRELOAD_POPULAR_MODELS', True)
        
        # Model optimization settings
        self.enable_quantization = getattr(settings, 'ENABLE_MODEL_QUANTIZATION', False)
        self.enable_pruning = getattr(settings, 'ENABLE_MODEL_PRUNING', False)
        
        # Persistence settings
        self.cache_persistence_enabled = getattr(settings, 'ENABLE_CACHE_PERSISTENCE', True)
        self.cache_persistence_path = Path(settings.MODELS_DIR) / "cache"
        self.cache_persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Background optimization
        self._optimization_thread: Optional[threading.Thread] = None
        self._optimization_active = False
        
        # Model usage tracking for intelligent caching
        self.usage_tracker: Dict[str, Dict[str, Any]] = {}
        self._usage_lock = threading.Lock()
        
        # Load persistent cache if enabled
        if self.cache_persistence_enabled:
            self._load_persistent_cache()
    
    def get_model(self, model_id: str, model_type: str = "voice_model") -> Optional[Any]:
        """
        Get model from cache or load if not cached.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model (voice_model, synthesis_model, etc.)
            
        Returns:
            Model data or None if loading fails
        """
        start_time = time.time()
        operation_id = f"model_cache_get_{model_id}_{int(time.time() * 1000)}"
        
        # Start performance monitoring
        performance_monitor.start_operation(
            'model_cache_access',
            operation_id,
            {'model_id': model_id, 'model_type': model_type}
        )
        
        try:
            # Try cache first
            cached_model = self.cache.get(model_id)
            if cached_model:
                # Update usage tracking
                self._update_usage_stats(model_id, cache_hit=True)
                
                # End performance monitoring
                performance_monitor.end_operation(
                    operation_id,
                    success=True,
                    additional_metadata={
                        'cache_hit': True,
                        'load_time': time.time() - start_time,
                        'model_size_mb': cached_model.size_bytes / (1024 * 1024)
                    }
                )
                
                logger.info(f"Cache hit for model {model_id}")
                return cached_model.model_data
            
            # Cache miss - load model
            logger.info(f"Cache miss for model {model_id}, loading...")
            model_data = self._load_model_from_storage(model_id, model_type)
            
            if model_data is None:
                performance_monitor.end_operation(
                    operation_id,
                    success=False,
                    error_message="Failed to load model from storage"
                )
                return None
            
            # Calculate model size and create cached model
            model_size = self._estimate_model_size(model_data)
            load_time = time.time() - start_time
            
            cached_model = CachedModel(
                model_id=model_id,
                model_data=model_data,
                model_type=model_type,
                size_bytes=model_size,
                load_time=load_time,
                last_accessed=datetime.now(),
                access_count=1,
                quality_score=self._assess_model_quality(model_data),
                metadata={'loaded_at': datetime.now().isoformat()}
            )
            
            # Add to cache
            if self.cache.put(model_id, cached_model):
                logger.info(f"Model {model_id} cached successfully")
            else:
                logger.warning(f"Failed to cache model {model_id}")
            
            # Update usage tracking
            self._update_usage_stats(model_id, cache_hit=False, load_time=load_time)
            
            # Update cache statistics
            self.cache.stats.total_load_time += load_time
            
            # End performance monitoring
            performance_monitor.end_operation(
                operation_id,
                success=True,
                additional_metadata={
                    'cache_hit': False,
                    'load_time': load_time,
                    'model_size_mb': model_size / (1024 * 1024)
                }
            )
            
            return model_data
            
        except Exception as e:
            performance_monitor.end_operation(
                operation_id,
                success=False,
                error_message=str(e)
            )
            logger.error(f"Error getting model {model_id}: {str(e)}")
            return None
    
    def preload_models(self, model_ids: List[str]) -> Dict[str, bool]:
        """
        Preload multiple models into cache.
        
        Args:
            model_ids: List of model IDs to preload
            
        Returns:
            Dictionary mapping model_id to success status
        """
        results = {}
        
        for model_id in model_ids:
            try:
                model_data = self.get_model(model_id)
                results[model_id] = model_data is not None
                
                if model_data:
                    logger.info(f"Successfully preloaded model {model_id}")
                else:
                    logger.warning(f"Failed to preload model {model_id}")
                    
            except Exception as e:
                logger.error(f"Error preloading model {model_id}: {str(e)}")
                results[model_id] = False
        
        return results
    
    def optimize_model(self, model_id: str, optimization_level: str = "balanced") -> bool:
        """
        Apply optimization techniques to a cached model.
        
        Args:
            model_id: Model to optimize
            optimization_level: "fast", "balanced", or "quality"
            
        Returns:
            True if optimization successful
        """
        try:
            cached_model = self.cache.get(model_id)
            if not cached_model:
                logger.warning(f"Model {model_id} not in cache for optimization")
                return False
            
            logger.info(f"Optimizing model {model_id} with level {optimization_level}")
            
            # Apply optimization based on level
            optimized_data = self._apply_model_optimization(
                cached_model.model_data, 
                optimization_level
            )
            
            if optimized_data:
                # Update cached model with optimized version
                original_size = cached_model.size_bytes
                optimized_size = self._estimate_model_size(optimized_data)
                
                cached_model.model_data = optimized_data
                cached_model.size_bytes = optimized_size
                cached_model.metadata['optimized'] = True
                cached_model.metadata['optimization_level'] = optimization_level
                cached_model.metadata['size_reduction'] = (original_size - optimized_size) / original_size
                
                # Update cache size tracking
                self.cache.current_size_bytes += (optimized_size - original_size)
                
                logger.info(f"Model {model_id} optimized: {original_size} -> {optimized_size} bytes")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_id}: {str(e)}")
            return False
    
    def start_background_optimization(self) -> None:
        """Start background model optimization thread."""
        if self._optimization_active:
            return
        
        self._optimization_active = True
        self._optimization_thread = threading.Thread(
            target=self._background_optimization_worker,
            daemon=True
        )
        self._optimization_thread.start()
        logger.info("Background model optimization started")
    
    def stop_background_optimization(self) -> None:
        """Stop background model optimization."""
        self._optimization_active = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=10.0)
        logger.info("Background model optimization stopped")
    
    def _background_optimization_worker(self) -> None:
        """Background worker for model optimization."""
        while self._optimization_active:
            try:
                # Find models that could benefit from optimization
                candidates = self._find_optimization_candidates()
                
                for model_id in candidates:
                    if not self._optimization_active:
                        break
                    
                    self.optimize_model(model_id, "balanced")
                    time.sleep(1)  # Prevent overwhelming the system
                
                # Sleep between optimization cycles
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in background optimization: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _find_optimization_candidates(self) -> List[str]:
        """Find models that would benefit from optimization."""
        candidates = []
        
        with self.cache._lock:
            for model_id, cached_model in self.cache.cache.items():
                # Skip already optimized models
                if cached_model.metadata.get('optimized', False):
                    continue
                
                # Consider models with high access count
                if cached_model.access_count >= 5:
                    candidates.append(model_id)
                
                # Consider large models
                if cached_model.size_bytes > 100 * 1024 * 1024:  # >100MB
                    candidates.append(model_id)
        
        return candidates
    
    def _load_model_from_storage(self, model_id: str, model_type: str) -> Optional[Any]:
        """Load model from persistent storage."""
        try:
            # This is a placeholder for actual model loading
            # In production, this would load from the actual model storage
            
            if model_type == "voice_model":
                # Simulate loading voice model characteristics
                model_data = {
                    "id": model_id,
                    "type": model_type,
                    "characteristics": {
                        "fundamental_frequency_range": {"min": 80, "max": 300, "mean": 150},
                        "formant_frequencies": [500, 1500, 2500, 3500],
                        "spectral_characteristics": {
                            "centroid": 2000,
                            "rolloff": 4000,
                            "bandwidth": 1000
                        }
                    },
                    "quality_score": 0.8,
                    "loaded_at": datetime.now().isoformat()
                }
                
                # Simulate loading time based on model complexity
                time.sleep(0.1)  # Simulate I/O delay
                
                return model_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None
    
    def _estimate_model_size(self, model_data: Any) -> int:
        """Estimate model size in bytes."""
        try:
            # Simple estimation based on serialized size
            if isinstance(model_data, dict):
                return len(json.dumps(model_data, default=str).encode('utf-8'))
            else:
                # For other types, use pickle size estimation
                return len(pickle.dumps(model_data))
        except:
            return 1024 * 1024  # Default 1MB estimate
    
    def _assess_model_quality(self, model_data: Any) -> float:
        """Assess model quality score."""
        try:
            if isinstance(model_data, dict) and 'quality_score' in model_data:
                return float(model_data['quality_score'])
            return 0.8  # Default quality score
        except:
            return 0.5  # Conservative default
    
    def _apply_model_optimization(self, model_data: Any, optimization_level: str) -> Optional[Any]:
        """Apply optimization techniques to model data."""
        try:
            # This is a placeholder for actual model optimization
            # In production, this would apply techniques like:
            # - Quantization
            # - Pruning
            # - Knowledge distillation
            # - Model compression
            
            if isinstance(model_data, dict):
                optimized_data = model_data.copy()
                
                if optimization_level == "fast":
                    # Aggressive optimization for speed
                    optimized_data['optimization'] = 'fast'
                    optimized_data['precision'] = 'int8'
                elif optimization_level == "balanced":
                    # Balanced optimization
                    optimized_data['optimization'] = 'balanced'
                    optimized_data['precision'] = 'fp16'
                elif optimization_level == "quality":
                    # Conservative optimization preserving quality
                    optimized_data['optimization'] = 'quality'
                    optimized_data['precision'] = 'fp32'
                
                return optimized_data
            
            return model_data
            
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            return None
    
    def _update_usage_stats(self, model_id: str, cache_hit: bool, load_time: float = 0.0) -> None:
        """Update model usage statistics."""
        with self._usage_lock:
            if model_id not in self.usage_tracker:
                self.usage_tracker[model_id] = {
                    'total_requests': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'total_load_time': 0.0,
                    'last_accessed': datetime.now(),
                    'first_accessed': datetime.now()
                }
            
            stats = self.usage_tracker[model_id]
            stats['total_requests'] += 1
            stats['last_accessed'] = datetime.now()
            
            if cache_hit:
                stats['cache_hits'] += 1
            else:
                stats['cache_misses'] += 1
                stats['total_load_time'] += load_time
    
    def _load_persistent_cache(self) -> None:
        """Load cache from persistent storage."""
        try:
            cache_file = self.cache_persistence_path / "cache_metadata.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_metadata = json.load(f)
                
                logger.info(f"Loaded cache metadata for {len(cache_metadata)} models")
                # In production, this would restore actual cached models
                
        except Exception as e:
            logger.error(f"Failed to load persistent cache: {str(e)}")
    
    def save_persistent_cache(self) -> None:
        """Save cache metadata to persistent storage."""
        try:
            cache_metadata = {}
            
            with self.cache._lock:
                for model_id, cached_model in self.cache.cache.items():
                    cache_metadata[model_id] = {
                        'model_type': cached_model.model_type,
                        'size_bytes': cached_model.size_bytes,
                        'quality_score': cached_model.quality_score,
                        'access_count': cached_model.access_count,
                        'last_accessed': cached_model.last_accessed.isoformat(),
                        'metadata': cached_model.metadata
                    }
            
            cache_file = self.cache_persistence_path / "cache_metadata.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_metadata, f, indent=2)
            
            logger.info(f"Saved cache metadata for {len(cache_metadata)} models")
            
        except Exception as e:
            logger.error(f"Failed to save persistent cache: {str(e)}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_stats = self.cache.get_cache_info()
        
        with self._usage_lock:
            usage_stats = {
                'tracked_models': len(self.usage_tracker),
                'total_model_requests': sum(stats['total_requests'] for stats in self.usage_tracker.values()),
                'most_popular_models': sorted(
                    [(model_id, stats['total_requests']) for model_id, stats in self.usage_tracker.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
        
        return {
            'cache_info': cache_stats,
            'usage_statistics': usage_stats,
            'optimization_active': self._optimization_active,
            'preload_enabled': self.preload_enabled,
            'cache_persistence_enabled': self.cache_persistence_enabled
        }
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        with self._usage_lock:
            self.usage_tracker.clear()
        logger.info("Model cache cleared")


# Global service instance
model_cache_service = ModelCacheService()