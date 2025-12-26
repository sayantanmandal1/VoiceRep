#!/usr/bin/env python3
"""
CUDA Setup Verification Test

This script verifies that CUDA is properly configured and working
with the voice cloning system.
"""

import sys
sys.path.append('.')
import torch
import time
from app.services.performance_optimization_service import PerformanceOptimizationService, OptimizationLevel

def test_cuda_setup():
    """Test CUDA configuration and performance."""
    
    print('=== CUDA Configuration Test ===')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

    if torch.cuda.is_available():
        print(f'Current GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    else:
        print('❌ CUDA not available!')
        return False

    print('\n=== Performance Optimization Service Test ===')
    service = PerformanceOptimizationService(OptimizationLevel.AGGRESSIVE)
    gpu_info = service.gpu_optimizer.get_gpu_utilization()
    
    print(f'GPU detected: {gpu_info["gpu_available"]}')
    print(f'GPU name: {gpu_info.get("device_name", "N/A")}')
    print(f'GPU memory total: {gpu_info.get("memory_total_mb", 0):.0f}MB')
    print(f'Device: {service.gpu_config.device}')
    print(f'Memory fraction: {service.gpu_config.memory_fraction}')
    print(f'Mixed precision: {service.gpu_config.enable_mixed_precision}')
    print(f'Tensor cores: {service.gpu_config.enable_tensor_cores}')

    print('\n=== GPU Tensor Operations Test ===')
    # Test GPU tensor operations
    x = torch.randn(1000, 1000, device=service.gpu_config.device)
    y = torch.randn(1000, 1000, device=service.gpu_config.device)

    start_time = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # Wait for GPU operation to complete
    gpu_time = time.time() - start_time

    print(f'GPU matrix multiplication (1000x1000): {gpu_time:.4f}s')
    print(f'Result device: {z.device}')
    print(f'Result shape: {z.shape}')

    # Compare with CPU
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    start_time = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start_time

    print(f'CPU matrix multiplication (1000x1000): {cpu_time:.4f}s')
    print(f'GPU speedup: {cpu_time/gpu_time:.1f}x faster')

    print('\n=== Memory Management Test ===')
    # Test memory management
    initial_memory = torch.cuda.memory_allocated()
    print(f'Initial GPU memory: {initial_memory / 1e6:.1f}MB')
    
    # Allocate some memory
    large_tensor = torch.randn(5000, 5000, device=service.gpu_config.device)
    allocated_memory = torch.cuda.memory_allocated()
    print(f'After allocation: {allocated_memory / 1e6:.1f}MB')
    
    # Clear cache
    service.gpu_optimizer.clear_gpu_cache()
    del large_tensor
    final_memory = torch.cuda.memory_allocated()
    print(f'After cleanup: {final_memory / 1e6:.1f}MB')

    print('\n✅ CUDA is properly configured and working!')
    return True

if __name__ == "__main__":
    success = test_cuda_setup()
    if success:
        print('\n🎉 All CUDA tests passed! Your system is ready for GPU-accelerated voice cloning.')
    else:
        print('\n❌ CUDA setup failed. Please check your CUDA installation.')
        sys.exit(1)