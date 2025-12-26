"""
Performance Optimization API endpoints.

This module provides REST API endpoints for performance optimization,
monitoring, and configuration management.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.performance_optimization_service import (
    performance_optimization_service,
    OptimizationLevel
)

logger = logging.getLogger(__name__)

router = APIRouter()


class OptimizationConfigRequest(BaseModel):
    """Request model for optimization configuration."""
    optimization_level: str = Field(..., description="Optimization level: conservative, balanced, aggressive, maximum")
    target_analysis_time: Optional[float] = Field(30.0, description="Target analysis time in seconds")
    target_realtime_factor: Optional[float] = Field(2.0, description="Target synthesis realtime factor")


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    optimization_level: str
    timestamp: float
    synthesis_performance: Dict[str, float]
    analysis_performance: Dict[str, float]
    gpu_utilization: Dict[str, float]
    memory_utilization: Dict[str, float]
    concurrent_tasks: Dict[str, float]
    cache_hit_rates: Dict[str, float]
    cache_statistics: Dict[str, Any]
    gpu_info: Dict[str, Any]
    executor_status: Dict[str, Any]
    targets: Dict[str, Any]


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    voice_profile_cache_size: int
    audio_cache_size: int
    model_cache_size: int
    voice_profile_hit_rate: float
    audio_hit_rate: float
    model_hit_rate: float
    audio_cache_memory_mb: float
    model_cache_memory_mb: float
    total_stores: int
    total_hits: int
    total_misses: int


class GPUStatusResponse(BaseModel):
    """Response model for GPU status."""
    gpu_available: bool
    device_name: Optional[str] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    memory_utilization: Optional[float] = None
    error: Optional[str] = None


class ConcurrencyStatusResponse(BaseModel):
    """Response model for concurrency status."""
    active_synthesis_tasks: int
    active_analysis_tasks: int
    total_active_tasks: int
    synthesis_executor_status: Dict[str, Any]
    analysis_executor_status: Dict[str, Any]
    general_executor_status: Dict[str, Any]
    process_executor_status: Dict[str, Any]


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """
    Get comprehensive performance metrics and statistics.
    
    Returns detailed performance information including:
    - Synthesis and analysis performance statistics
    - GPU and memory utilization
    - Cache hit rates and statistics
    - Concurrent task information
    - Target achievement rates
    """
    try:
        report = performance_optimization_service.get_performance_report()
        
        return PerformanceMetricsResponse(
            optimization_level=report["optimization_level"],
            timestamp=report["timestamp"],
            synthesis_performance=report["synthesis_performance"],
            analysis_performance=report["analysis_performance"],
            gpu_utilization=report["gpu_utilization"],
            memory_utilization=report["memory_utilization"],
            concurrent_tasks=report["concurrent_tasks"],
            cache_hit_rates=report["cache_hit_rates"],
            cache_statistics=report["cache_statistics"],
            gpu_info=report["gpu_info"],
            executor_status=report["executor_status"],
            targets=report["targets"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_statistics():
    """
    Get detailed cache performance statistics.
    
    Returns information about:
    - Cache sizes and hit rates
    - Memory usage by cache type
    - Total operations (stores, hits, misses)
    """
    try:
        stats = performance_optimization_service.cache.get_cache_statistics()
        
        return CacheStatsResponse(
            voice_profile_cache_size=stats.get("voice_profile_cache_size", 0),
            audio_cache_size=stats.get("audio_cache_size", 0),
            model_cache_size=stats.get("model_cache_size", 0),
            voice_profile_hit_rate=stats.get("voice_profile_hit_rate", 0.0),
            audio_hit_rate=stats.get("audio_hit_rate", 0.0),
            model_hit_rate=stats.get("model_hit_rate", 0.0),
            audio_cache_memory_mb=stats.get("audio_cache_memory_mb", 0.0),
            model_cache_memory_mb=stats.get("model_cache_memory_mb", 0.0),
            total_stores=stats.get("voice_profile_stores", 0) + stats.get("audio_stores", 0) + stats.get("model_stores", 0),
            total_hits=stats.get("voice_profile_hits", 0) + stats.get("audio_hits", 0) + stats.get("model_hits", 0),
            total_misses=stats.get("voice_profile_misses", 0) + stats.get("audio_misses", 0) + stats.get("model_misses", 0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )


@router.get("/gpu/status", response_model=GPUStatusResponse)
async def get_gpu_status():
    """
    Get current GPU status and utilization information.
    
    Returns GPU availability, memory usage, and device information.
    """
    try:
        gpu_info = performance_optimization_service.gpu_optimizer.get_gpu_utilization()
        
        return GPUStatusResponse(
            gpu_available=gpu_info.get("gpu_available", False),
            device_name=gpu_info.get("device_name"),
            memory_allocated_mb=gpu_info.get("memory_allocated_mb"),
            memory_reserved_mb=gpu_info.get("memory_reserved_mb"),
            memory_total_mb=gpu_info.get("memory_total_mb"),
            memory_utilization=gpu_info.get("memory_utilization"),
            error=gpu_info.get("error")
        )
        
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve GPU status: {str(e)}"
        )


@router.get("/concurrency/status", response_model=ConcurrencyStatusResponse)
async def get_concurrency_status():
    """
    Get current concurrency and task execution status.
    
    Returns information about active tasks and executor status.
    """
    try:
        active_tasks = performance_optimization_service.concurrency_manager.get_active_task_count()
        executor_status = performance_optimization_service.concurrency_manager.get_executor_status()
        
        return ConcurrencyStatusResponse(
            active_synthesis_tasks=active_tasks.get("synthesis", 0),
            active_analysis_tasks=active_tasks.get("analysis", 0),
            total_active_tasks=active_tasks.get("total", 0),
            synthesis_executor_status=executor_status.get("synthesis_executor", {}),
            analysis_executor_status=executor_status.get("analysis_executor", {}),
            general_executor_status=executor_status.get("general_executor", {}),
            process_executor_status=executor_status.get("process_executor", {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get concurrency status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve concurrency status: {str(e)}"
        )


@router.post("/configuration")
async def update_optimization_configuration(
    config: OptimizationConfigRequest,
    background_tasks: BackgroundTasks
):
    """
    Update performance optimization configuration.
    
    Allows changing optimization level and performance targets.
    Changes are applied in the background to avoid disrupting active operations.
    """
    try:
        # Validate optimization level
        try:
            optimization_level = OptimizationLevel(config.optimization_level.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization level: {config.optimization_level}. "
                       f"Valid options: {[level.value for level in OptimizationLevel]}"
            )
        
        # Update configuration in background
        def update_config():
            try:
                performance_optimization_service._update_optimization_level(optimization_level)
                
                if config.target_analysis_time or config.target_realtime_factor:
                    performance_optimization_service.optimize_for_target_performance(
                        target_analysis_time=config.target_analysis_time or 30.0,
                        target_realtime_factor=config.target_realtime_factor or 2.0
                    )
                
                logger.info(f"Optimization configuration updated: {config.optimization_level}")
                
            except Exception as e:
                logger.error(f"Configuration update failed: {e}")
        
        background_tasks.add_task(update_config)
        
        return {
            "message": "Optimization configuration update initiated",
            "optimization_level": config.optimization_level,
            "target_analysis_time": config.target_analysis_time,
            "target_realtime_factor": config.target_realtime_factor
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update optimization configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_cache(cache_type: Optional[str] = None):
    """
    Clear performance caches.
    
    Args:
        cache_type: Optional cache type to clear ("voice_profiles", "audio", "models").
                   If not specified, clears all caches.
    
    Returns confirmation of cache clearing operation.
    """
    try:
        # Validate cache type if specified
        valid_cache_types = ["voice_profiles", "audio", "models"]
        if cache_type and cache_type not in valid_cache_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cache type: {cache_type}. Valid options: {valid_cache_types}"
            )
        
        # Get cache stats before clearing
        stats_before = performance_optimization_service.cache.get_cache_statistics()
        
        # Clear cache
        performance_optimization_service.cache.clear_cache(cache_type)
        
        # Get cache stats after clearing
        stats_after = performance_optimization_service.cache.get_cache_statistics()
        
        return {
            "message": f"Cache cleared successfully: {cache_type or 'all'}",
            "cache_type": cache_type or "all",
            "stats_before": {
                "voice_profile_cache_size": stats_before.get("voice_profile_cache_size", 0),
                "audio_cache_size": stats_before.get("audio_cache_size", 0),
                "model_cache_size": stats_before.get("model_cache_size", 0)
            },
            "stats_after": {
                "voice_profile_cache_size": stats_after.get("voice_profile_cache_size", 0),
                "audio_cache_size": stats_after.get("audio_cache_size", 0),
                "model_cache_size": stats_after.get("model_cache_size", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/gpu/clear-cache")
async def clear_gpu_cache():
    """
    Clear GPU memory cache to free up VRAM.
    
    Useful when GPU memory is running low or after processing large models.
    """
    try:
        # Get GPU status before clearing
        gpu_info_before = performance_optimization_service.gpu_optimizer.get_gpu_utilization()
        
        # Clear GPU cache
        performance_optimization_service.gpu_optimizer.clear_gpu_cache()
        
        # Get GPU status after clearing
        gpu_info_after = performance_optimization_service.gpu_optimizer.get_gpu_utilization()
        
        return {
            "message": "GPU cache cleared successfully",
            "gpu_available": gpu_info_after.get("gpu_available", False),
            "memory_freed_mb": (
                gpu_info_before.get("memory_allocated_mb", 0) - 
                gpu_info_after.get("memory_allocated_mb", 0)
            ) if gpu_info_before.get("gpu_available") and gpu_info_after.get("gpu_available") else 0,
            "memory_before_mb": gpu_info_before.get("memory_allocated_mb", 0),
            "memory_after_mb": gpu_info_after.get("memory_allocated_mb", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to clear GPU cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear GPU cache: {str(e)}"
        )


@router.post("/optimize/auto")
async def auto_optimize_performance(
    background_tasks: BackgroundTasks,
    target_analysis_time: float = 30.0,
    target_realtime_factor: float = 2.0
):
    """
    Automatically optimize performance configuration based on current metrics.
    
    Analyzes recent performance data and adjusts optimization settings
    to meet specified targets.
    
    Args:
        target_analysis_time: Target time for voice analysis (seconds)
        target_realtime_factor: Target synthesis speed relative to real-time
    """
    try:
        # Validate targets
        if target_analysis_time <= 0 or target_analysis_time > 300:
            raise HTTPException(
                status_code=400,
                detail="Target analysis time must be between 0 and 300 seconds"
            )
        
        if target_realtime_factor <= 0 or target_realtime_factor > 10:
            raise HTTPException(
                status_code=400,
                detail="Target realtime factor must be between 0 and 10"
            )
        
        # Get current performance metrics
        current_report = performance_optimization_service.get_performance_report()
        
        # Perform optimization in background
        def auto_optimize():
            try:
                performance_optimization_service.optimize_for_target_performance(
                    target_analysis_time=target_analysis_time,
                    target_realtime_factor=target_realtime_factor
                )
                logger.info(f"Auto-optimization completed for targets: {target_analysis_time}s analysis, {target_realtime_factor}x synthesis")
                
            except Exception as e:
                logger.error(f"Auto-optimization failed: {e}")
        
        background_tasks.add_task(auto_optimize)
        
        return {
            "message": "Auto-optimization initiated",
            "target_analysis_time": target_analysis_time,
            "target_realtime_factor": target_realtime_factor,
            "current_optimization_level": current_report.get("optimization_level", "unknown"),
            "current_performance": {
                "avg_analysis_time": current_report.get("analysis_performance", {}).get("mean", 0),
                "analysis_count": current_report.get("analysis_performance", {}).get("count", 0),
                "synthesis_count": current_report.get("synthesis_performance", {}).get("count", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate auto-optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate auto-optimization: {str(e)}"
        )


@router.get("/health")
async def get_performance_health():
    """
    Get performance optimization service health status.
    
    Returns overall health and status of optimization components.
    """
    try:
        # Check GPU availability
        gpu_info = performance_optimization_service.gpu_optimizer.get_gpu_utilization()
        gpu_healthy = gpu_info.get("gpu_available", False) and not gpu_info.get("error")
        
        # Check cache health
        cache_stats = performance_optimization_service.cache.get_cache_statistics()
        cache_healthy = True  # Cache is always considered healthy if accessible
        
        # Check concurrency health
        executor_status = performance_optimization_service.concurrency_manager.get_executor_status()
        concurrency_healthy = all(
            status.get("max_workers", 0) > 0 
            for status in executor_status.values()
        )
        
        # Check monitoring status
        monitoring_healthy = performance_optimization_service.monitoring_active
        
        # Overall health
        overall_healthy = all([
            cache_healthy,
            concurrency_healthy,
            monitoring_healthy
        ])
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": time.time(),
            "components": {
                "gpu_optimizer": "healthy" if gpu_healthy else "degraded",
                "cache_system": "healthy" if cache_healthy else "unhealthy",
                "concurrency_manager": "healthy" if concurrency_healthy else "unhealthy",
                "performance_monitoring": "healthy" if monitoring_healthy else "unhealthy"
            },
            "optimization_level": performance_optimization_service.optimization_level.value,
            "monitoring_active": monitoring_healthy,
            "gpu_available": gpu_info.get("gpu_available", False)
        }
        
    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "components": {
                "gpu_optimizer": "unknown",
                "cache_system": "unknown",
                "concurrency_manager": "unknown",
                "performance_monitoring": "unknown"
            }
        }


# Import time for health endpoint
import time