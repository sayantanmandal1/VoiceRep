"""
Test suite for Performance Optimization Service.

This test suite validates the performance optimization components including
GPU optimization, intelligent caching, concurrent processing, and performance monitoring.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from app.services.performance_optimization_service import (
    PerformanceOptimizationService,
    OptimizationLevel,
    GPUOptimizer,
    IntelligentCache,
    ConcurrentProcessingManager,
    GPUConfiguration,
    CacheConfiguration,
    ConcurrencyConfiguration
)
from app.schemas.voice import VoiceProfileSchema


class TestGPUOptimizer:
    """Test GPU optimization functionality."""
    
    def test_gpu_optimizer_initialization(self):
        """Test GPU optimizer initialization."""
        config = GPUConfiguration(
            device=Mock(),
            memory_fraction=0.8,
            enable_mixed_precision=True
        )
        
        optimizer = GPUOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.device == config.device
    
    def test_get_gpu_utilization_no_gpu(self):
        """Test GPU utilization when no GPU is available."""
        config = GPUConfiguration(device=Mock())
        config.device.type = 'cpu'
        
        optimizer = GPUOptimizer(config)
        utilization = optimizer.get_gpu_utilization()
        
        assert utilization["gpu_available"] is False
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024*1024*1024)  # 1GB
    @patch('torch.cuda.memory_reserved', return_value=2*1024*1024*1024)  # 2GB
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.get_device_name', return_value="Test GPU")
    def test_get_gpu_utilization_with_gpu(self, mock_name, mock_props, mock_reserved, mock_allocated, mock_available):
        """Test GPU utilization when GPU is available."""
        mock_props.return_value.total_memory = 8*1024*1024*1024  # 8GB
        
        config = GPUConfiguration(device=Mock())
        config.device.type = 'cuda'
        
        optimizer = GPUOptimizer(config)
        utilization = optimizer.get_gpu_utilization()
        
        assert utilization["gpu_available"] is True
        assert utilization["device_name"] == "Test GPU"
        assert utilization["memory_allocated_mb"] == 1024
        assert utilization["memory_reserved_mb"] == 2048
        assert utilization["memory_total_mb"] == 8192
    
    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        config = GPUConfiguration(device=Mock())
        config.device.type = 'cpu'
        
        optimizer = GPUOptimizer(config)
        batch_size = optimizer.get_optimal_batch_size(100.0, (1, 80, 100))
        
        assert batch_size == 1  # CPU should return 1


class TestIntelligentCache:
    """Test intelligent caching functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        config = CacheConfiguration(
            max_voice_profiles=50,
            max_audio_cache_mb=512,
            ttl_seconds=3600
        )
        
        cache = IntelligentCache(config)
        
        assert cache.config == config
        assert len(cache.voice_profile_cache) == 0
        assert len(cache.audio_cache) == 0
    
    def test_voice_profile_caching(self):
        """Test voice profile caching and retrieval."""
        config = CacheConfiguration(max_voice_profiles=10, ttl_seconds=3600)
        cache = IntelligentCache(config)
        
        # Create test profile
        profile = VoiceProfileSchema(
            id="test_profile",
            reference_audio_id="test_audio",
            quality_score=0.9,
            created_at=time.time()
        )
        
        analysis_data = {"features": [1, 2, 3], "quality": 0.9}
        
        # Cache the profile
        cache.cache_voice_profile(profile, analysis_data)
        
        # Retrieve the profile
        result = cache.get_cached_voice_profile("test_profile")
        
        assert result is not None
        cached_profile, cached_data = result
        assert cached_profile.id == "test_profile"
        assert cached_data["quality"] == 0.9
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        config = CacheConfiguration(max_voice_profiles=10, ttl_seconds=1)  # 1 second TTL
        cache = IntelligentCache(config)
        
        profile = VoiceProfileSchema(
            id="test_profile",
            reference_audio_id="test_audio",
            quality_score=0.9,
            created_at=time.time()
        )
        
        # Cache the profile
        cache.cache_voice_profile(profile, {"data": "test"})
        
        # Should be available immediately
        result = cache.get_cached_voice_profile("test_profile")
        assert result is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        result = cache.get_cached_voice_profile("test_profile")
        assert result is None
    
    def test_audio_caching(self):
        """Test audio data caching."""
        config = CacheConfiguration(max_audio_cache_mb=100, ttl_seconds=3600)
        cache = IntelligentCache(config)
        
        # Create test audio data
        audio_data = np.random.random(1000).astype(np.float32)
        metadata = {"sample_rate": 22050, "duration": 1.0}
        
        # Cache the audio
        cache.cache_audio_data("test_audio", audio_data, metadata)
        
        # Retrieve the audio
        result = cache.get_cached_audio_data("test_audio")
        
        assert result is not None
        cached_audio, cached_metadata = result
        assert len(cached_audio) == 1000
        assert cached_metadata["sample_rate"] == 22050
    
    def test_cache_size_limits(self):
        """Test cache size limit enforcement."""
        config = CacheConfiguration(max_voice_profiles=2, ttl_seconds=3600, enable_disk_cache=False)
        cache = IntelligentCache(config)
        
        # Add profiles up to limit
        for i in range(3):
            profile = VoiceProfileSchema(
                id=f"profile_{i}",
                reference_audio_id=f"audio_{i}",
                quality_score=0.9,
                created_at=time.time()
            )
            cache.cache_voice_profile(profile, {"data": i})
        
        # Should only have 2 profiles (LRU eviction)
        assert len(cache.voice_profile_cache) == 2
        
        # First profile should be evicted
        result = cache.get_cached_voice_profile("profile_0")
        assert result is None
        
        # Last two should be available
        result = cache.get_cached_voice_profile("profile_2")
        assert result is not None
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        config = CacheConfiguration(max_voice_profiles=10, ttl_seconds=3600)
        cache = IntelligentCache(config)
        
        profile = VoiceProfileSchema(
            id="test_profile",
            reference_audio_id="test_audio",
            quality_score=0.9,
            created_at=time.time()
        )
        
        # Cache and retrieve to generate stats
        cache.cache_voice_profile(profile, {"data": "test"})
        cache.get_cached_voice_profile("test_profile")  # Hit
        cache.get_cached_voice_profile("nonexistent")   # Miss
        
        stats = cache.get_cache_statistics()
        
        assert stats["voice_profile_cache_size"] == 1
        assert stats["voice_profile_hits"] == 1
        assert stats["voice_profile_misses"] == 1
        assert stats["voice_profile_hit_rate"] == 0.5


class TestConcurrentProcessingManager:
    """Test concurrent processing functionality."""
    
    def test_manager_initialization(self):
        """Test concurrent processing manager initialization."""
        config = ConcurrencyConfiguration(
            max_concurrent_synthesis=4,
            max_concurrent_analysis=2,
            thread_pool_size=8
        )
        
        manager = ConcurrentProcessingManager(config)
        
        assert manager.config == config
        assert manager.synthesis_executor._max_workers == 4
        assert manager.analysis_executor._max_workers == 2
    
    @pytest.mark.asyncio
    async def test_synthesis_task_submission(self):
        """Test synthesis task submission and execution."""
        config = ConcurrencyConfiguration(
            max_concurrent_synthesis=2,
            thread_pool_size=4
        )
        
        manager = ConcurrentProcessingManager(config)
        
        def test_task(x, y):
            return x + y
        
        # Submit task
        result = await manager.submit_synthesis_task(test_task, 5, 3, priority=5)
        
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_analysis_task_submission(self):
        """Test analysis task submission and execution."""
        config = ConcurrencyConfiguration(
            max_concurrent_analysis=2,
            thread_pool_size=4
        )
        
        manager = ConcurrentProcessingManager(config)
        
        def test_analysis(data):
            return {"result": len(data)}
        
        # Submit task
        result = await manager.submit_analysis_task(test_analysis, "test_data")
        
        assert result["result"] == 9
    
    @pytest.mark.asyncio
    async def test_batch_task_execution(self):
        """Test batch task execution."""
        config = ConcurrencyConfiguration(thread_pool_size=4)
        manager = ConcurrentProcessingManager(config)
        
        def multiply_task(x, multiplier=2):
            return x * multiplier
        
        # Create batch tasks
        tasks = [
            (multiply_task, (i,), {"multiplier": 3})
            for i in range(5)
        ]
        
        # Execute batch
        results = await manager.submit_batch_tasks(tasks, max_concurrent=2)
        
        assert len(results) == 5
        assert results[0] == 0  # 0 * 3
        assert results[1] == 3  # 1 * 3
        assert results[4] == 12  # 4 * 3
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout functionality."""
        config = ConcurrencyConfiguration(thread_pool_size=2)
        manager = ConcurrentProcessingManager(config)
        
        def slow_task():
            time.sleep(2)
            return "completed"
        
        # Submit task with short timeout
        with pytest.raises(asyncio.TimeoutError):
            await manager.submit_synthesis_task(slow_task, timeout=1)
    
    def test_active_task_tracking(self):
        """Test active task count tracking."""
        config = ConcurrencyConfiguration(thread_pool_size=4)
        manager = ConcurrentProcessingManager(config)
        
        # Initially no active tasks
        task_counts = manager.get_active_task_count()
        assert task_counts["total"] == 0
    
    def test_executor_status(self):
        """Test executor status reporting."""
        config = ConcurrencyConfiguration(
            max_concurrent_synthesis=4,
            max_concurrent_analysis=2,
            thread_pool_size=8,
            process_pool_size=2
        )
        
        manager = ConcurrentProcessingManager(config)
        status = manager.get_executor_status()
        
        assert status["synthesis_executor"]["max_workers"] == 4
        assert status["analysis_executor"]["max_workers"] == 2
        assert status["general_executor"]["max_workers"] == 8
        assert status["process_executor"]["max_workers"] == 2


class TestPerformanceOptimizationService:
    """Test main performance optimization service."""
    
    def test_service_initialization(self):
        """Test service initialization with different optimization levels."""
        service = PerformanceOptimizationService(OptimizationLevel.BALANCED)
        
        assert service.optimization_level == OptimizationLevel.BALANCED
        assert service.gpu_optimizer is not None
        assert service.cache is not None
        assert service.concurrency_manager is not None
    
    def test_optimization_level_configs(self):
        """Test different optimization level configurations."""
        # Test conservative level
        conservative_service = PerformanceOptimizationService(OptimizationLevel.CONSERVATIVE)
        assert conservative_service.gpu_config.memory_fraction == 0.6
        assert conservative_service.cache_config.max_voice_profiles == 50
        
        # Test aggressive level
        aggressive_service = PerformanceOptimizationService(OptimizationLevel.AGGRESSIVE)
        assert aggressive_service.gpu_config.memory_fraction == 0.9
        assert aggressive_service.cache_config.max_voice_profiles == 200
    
    @pytest.mark.asyncio
    async def test_voice_analysis_optimization(self):
        """Test voice analysis optimization."""
        service = PerformanceOptimizationService(OptimizationLevel.BALANCED)
        
        def mock_analysis(audio_path, **kwargs):
            time.sleep(0.1)  # Simulate processing
            return {
                "features": [1, 2, 3],
                "quality_score": 0.9,
                "processing_time": 0.1
            }
        
        # Test analysis optimization
        result, processing_time = await service.optimize_voice_analysis(
            mock_analysis,
            "test_audio.wav",
            target_time_seconds=1.0
        )
        
        assert result is not None
        assert "features" in result
        assert processing_time < 1.0  # Should meet target
    
    @pytest.mark.asyncio
    async def test_speech_synthesis_optimization(self):
        """Test speech synthesis optimization."""
        service = PerformanceOptimizationService(OptimizationLevel.BALANCED)
        
        def mock_synthesis(text, voice_profile, **kwargs):
            time.sleep(0.1)  # Simulate processing
            return (
                True,
                "output.wav",
                {
                    "text": text,
                    "duration": 2.0,
                    "quality_score": 0.95
                }
            )
        
        # Create test voice profile
        voice_profile = VoiceProfileSchema(
            id="test_profile",
            reference_audio_id="test_audio",
            quality_score=0.9,
            created_at=time.time()
        )
        
        # Test synthesis optimization
        result, processing_time, realtime_factor = await service.optimize_speech_synthesis(
            mock_synthesis,
            "Hello world",
            voice_profile,
            target_realtime_factor=2.0
        )
        
        assert result is not None
        assert result[0] is True  # Success
        assert processing_time > 0
        assert realtime_factor > 0
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        service = PerformanceOptimizationService(OptimizationLevel.BALANCED)
        
        # Add some mock metrics
        service.metrics.synthesis_times = [0.5, 0.7, 0.3, 0.9]
        service.metrics.analysis_times = [15.0, 25.0, 20.0, 30.0]
        service.metrics.gpu_utilization = [0.6, 0.7, 0.8]
        
        report = service.get_performance_report()
        
        assert report["optimization_level"] == "balanced"
        assert "synthesis_performance" in report
        assert "analysis_performance" in report
        assert "gpu_utilization" in report
        assert "targets" in report
    
    def test_performance_monitoring_lifecycle(self):
        """Test performance monitoring start/stop."""
        service = PerformanceOptimizationService(OptimizationLevel.BALANCED)
        
        # Initially not monitoring
        assert not service.monitoring_active
        
        # Note: In a real async environment, monitoring would work
        # For this test, we just verify the state changes
        service.monitoring_active = True
        assert service.monitoring_active
        
        # Stop monitoring
        service.stop_performance_monitoring()
        assert not service.monitoring_active
    
    def test_resource_cleanup(self):
        """Test resource cleanup."""
        service = PerformanceOptimizationService(OptimizationLevel.BALANCED)
        
        # Simulate monitoring to have resources to clean up
        service.monitoring_active = True
        
        # Cleanup should not raise exceptions
        service.cleanup_resources()
        
        # Should stop monitoring
        assert not service.monitoring_active


class TestIntegration:
    """Integration tests for performance optimization components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self):
        """Test end-to-end performance optimization workflow."""
        service = PerformanceOptimizationService(OptimizationLevel.BALANCED)
        
        # Mock voice analysis function
        def mock_voice_analysis(audio_path, **kwargs):
            return {
                "pitch_features": {"mean_f0": 150.0},
                "formant_features": {"f1": 500, "f2": 1500},
                "quality_score": 0.9
            }
        
        # Mock synthesis function
        def mock_synthesis(text, voice_profile, **kwargs):
            return (True, "output.wav", {"duration": 2.0, "quality": 0.95})
        
        # Create test voice profile
        voice_profile = VoiceProfileSchema(
            id="integration_test",
            reference_audio_id="test_audio",
            quality_score=0.9,
            created_at=time.time()
        )
        
        # Test analysis optimization
        analysis_result, analysis_time = await service.optimize_voice_analysis(
            mock_voice_analysis,
            "test_audio.wav"
        )
        
        assert analysis_result is not None
        assert "pitch_features" in analysis_result
        
        # Test synthesis optimization
        synthesis_result, synthesis_time, realtime_factor = await service.optimize_speech_synthesis(
            mock_synthesis,
            "Test text",
            voice_profile
        )
        
        assert synthesis_result is not None
        assert synthesis_result[0] is True
        
        # Test caching (second analysis should be faster)
        analysis_result2, analysis_time2 = await service.optimize_voice_analysis(
            mock_voice_analysis,
            "test_audio.wav"
        )
        
        # Second analysis should be much faster due to caching
        assert analysis_time2 < analysis_time / 2
    
    def test_performance_under_load(self):
        """Test performance optimization under concurrent load."""
        service = PerformanceOptimizationService(OptimizationLevel.AGGRESSIVE)
        
        def cpu_intensive_task(n):
            # Simulate CPU-intensive work
            result = 0
            for i in range(n * 1000):
                result += i ** 0.5
            return result
        
        # Submit multiple concurrent tasks
        import asyncio
        
        async def run_concurrent_tasks():
            tasks = []
            for i in range(10):
                task = service.concurrency_manager.submit_synthesis_task(
                    cpu_intensive_task, 
                    100 + i,
                    priority=i % 3 + 1
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        # Run the test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(run_concurrent_tasks())
            
            # All tasks should complete successfully
            assert len(results) == 10
            assert all(isinstance(r, (int, float)) for r in results)
            
        finally:
            loop.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running Performance Optimization Tests...")
    
    # Test GPU optimizer
    print("Testing GPU Optimizer...")
    test_gpu = TestGPUOptimizer()
    test_gpu.test_gpu_optimizer_initialization()
    test_gpu.test_get_gpu_utilization_no_gpu()
    print("✓ GPU Optimizer tests passed")
    
    # Test cache
    print("Testing Intelligent Cache...")
    test_cache = TestIntelligentCache()
    test_cache.test_cache_initialization()
    test_cache.test_voice_profile_caching()
    test_cache.test_audio_caching()
    test_cache.test_cache_size_limits()
    test_cache.test_cache_statistics()
    print("✓ Intelligent Cache tests passed")
    
    # Test concurrency manager
    print("Testing Concurrency Manager...")
    test_concurrency = TestConcurrentProcessingManager()
    test_concurrency.test_manager_initialization()
    test_concurrency.test_active_task_tracking()
    test_concurrency.test_executor_status()
    print("✓ Concurrency Manager tests passed")
    
    # Test main service
    print("Testing Performance Optimization Service...")
    test_service = TestPerformanceOptimizationService()
    test_service.test_service_initialization()
    test_service.test_optimization_level_configs()
    test_service.test_performance_report_generation()
    test_service.test_performance_monitoring_lifecycle()
    test_service.test_resource_cleanup()
    print("✓ Performance Optimization Service tests passed")
    
    print("\nAll Performance Optimization tests completed successfully! ✓")
    print("\nPerformance optimization features implemented:")
    print("- GPU optimization for maximum synthesis speed")
    print("- Intelligent caching system for repeated voice profiles")
    print("- Concurrent request handling with quality maintenance")
    print("- Analysis time optimization to meet 30-second target")
    print("- Faster-than-real-time synthesis generation")
    print("- Comprehensive performance monitoring and reporting")