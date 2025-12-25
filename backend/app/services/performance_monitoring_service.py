"""
Performance monitoring and optimization service for voice cloning system.
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    operation_type: str
    operation_id: str
    processing_time: float
    cpu_usage: float
    memory_usage: float
    queue_size: int
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceUsage:
    """System resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    active_tasks: int
    queue_length: int


@dataclass
class PerformanceThresholds:
    """Performance requirement thresholds from requirements."""
    # Requirements 6.1: Processing time limits
    voice_analysis_max_seconds: float = 60.0  # For ≤30s audio
    synthesis_max_seconds: float = 30.0       # For ≤100 words
    
    # System resource thresholds
    max_cpu_percent: float = 85.0
    max_memory_percent: float = 80.0
    max_queue_size: int = 50
    
    # Quality thresholds
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05


class PerformanceMonitor:
    """Main performance monitoring service."""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.resource_history: deque = deque(maxlen=1000)   # Keep last 1k resource readings
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.thresholds = PerformanceThresholds()
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Performance statistics
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'success_count': 0,
            'error_count': 0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'recent_times': deque(maxlen=100)
        })
        
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        metrics_dir = Path(settings.RESULTS_DIR) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def start_monitoring(self) -> None:
        """Start background resource monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if monitoring is currently active."""
        return self._monitoring_active
    
    def _monitor_resources(self) -> None:
        """Background thread for monitoring system resources."""
        while self._monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Get queue information (placeholder - would integrate with actual queue)
                queue_length = len(self.active_operations)
                
                resource_usage = ResourceUsage(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_mb=memory.available / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    active_tasks=len(self.active_operations),
                    queue_length=queue_length
                )
                
                with self._lock:
                    self.resource_history.append(resource_usage)
                
                # Check for resource threshold violations
                self._check_resource_thresholds(resource_usage)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(10)  # Wait longer on error
    
    def _check_resource_thresholds(self, resource_usage: ResourceUsage) -> None:
        """Check if resource usage exceeds thresholds."""
        warnings = []
        
        if resource_usage.cpu_percent > self.thresholds.max_cpu_percent:
            warnings.append(f"High CPU usage: {resource_usage.cpu_percent:.1f}%")
        
        if resource_usage.memory_percent > self.thresholds.max_memory_percent:
            warnings.append(f"High memory usage: {resource_usage.memory_percent:.1f}%")
        
        if resource_usage.queue_length > self.thresholds.max_queue_size:
            warnings.append(f"High queue size: {resource_usage.queue_length}")
        
        if warnings:
            logger.warning(f"Resource threshold violations: {'; '.join(warnings)}")
    
    def start_operation(self, operation_type: str, operation_id: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking a performance-critical operation."""
        with self._lock:
            self.active_operations[operation_id] = {
                'type': operation_type,
                'start_time': time.time(),
                'start_cpu': psutil.cpu_percent(),
                'start_memory': psutil.virtual_memory().percent,
                'metadata': metadata or {}
            }
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     error_message: Optional[str] = None,
                     additional_metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """End tracking an operation and record metrics."""
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent
        
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return None
            
            operation = self.active_operations.pop(operation_id)
            
            processing_time = end_time - operation['start_time']
            
            # Combine metadata
            metadata = operation['metadata'].copy()
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation_type=operation['type'],
                operation_id=operation_id,
                processing_time=processing_time,
                cpu_usage=end_cpu,
                memory_usage=end_memory,
                queue_size=len(self.active_operations),
                success=success,
                error_message=error_message,
                metadata=metadata
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update operation statistics
            self._update_operation_stats(metrics)
            
            # Check performance thresholds
            self._check_performance_thresholds(metrics)
            
            return metrics
    
    def _update_operation_stats(self, metrics: PerformanceMetrics) -> None:
        """Update operation statistics."""
        op_type = metrics.operation_type
        stats = self.operation_stats[op_type]
        
        stats['count'] += 1
        stats['total_time'] += metrics.processing_time
        
        if metrics.success:
            stats['success_count'] += 1
        else:
            stats['error_count'] += 1
        
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], metrics.processing_time)
        stats['max_time'] = max(stats['max_time'], metrics.processing_time)
        stats['recent_times'].append(metrics.processing_time)
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check if operation performance meets requirements."""
        warnings = []
        
        # Check processing time thresholds based on operation type
        if metrics.operation_type == 'voice_analysis':
            if metrics.processing_time > self.thresholds.voice_analysis_max_seconds:
                warnings.append(f"Voice analysis exceeded time limit: {metrics.processing_time:.1f}s > {self.thresholds.voice_analysis_max_seconds}s")
        
        elif metrics.operation_type == 'speech_synthesis':
            if metrics.processing_time > self.thresholds.synthesis_max_seconds:
                warnings.append(f"Speech synthesis exceeded time limit: {metrics.processing_time:.1f}s > {self.thresholds.synthesis_max_seconds}s")
        
        if warnings:
            logger.warning(f"Performance threshold violations for {metrics.operation_id}: {'; '.join(warnings)}")
    
    def get_operation_statistics(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for operations."""
        with self._lock:
            if operation_type:
                if operation_type in self.operation_stats:
                    stats = self.operation_stats[operation_type].copy()
                    # Convert deque to list for JSON serialization
                    stats['recent_times'] = list(stats['recent_times'])
                    return {operation_type: stats}
                else:
                    return {}
            else:
                # Return all statistics
                all_stats = {}
                for op_type, stats in self.operation_stats.items():
                    stats_copy = stats.copy()
                    stats_copy['recent_times'] = list(stats_copy['recent_times'])
                    all_stats[op_type] = stats_copy
                return all_stats
    
    def get_resource_usage(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent resource usage data."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_usage = [
                {
                    'timestamp': usage.timestamp.isoformat(),
                    'cpu_percent': usage.cpu_percent,
                    'memory_percent': usage.memory_percent,
                    'memory_available_mb': usage.memory_available_mb,
                    'disk_usage_percent': usage.disk_usage_percent,
                    'active_tasks': usage.active_tasks,
                    'queue_length': usage.queue_length
                }
                for usage in self.resource_history
                if usage.timestamp >= cutoff_time
            ]
        
        return recent_usage
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary (non-blocking version)."""
        try:
            with self._lock:
                # Calculate overall success rate
                total_operations = sum(stats['count'] for stats in self.operation_stats.values())
                total_successes = sum(stats['success_count'] for stats in self.operation_stats.values())
                overall_success_rate = total_successes / total_operations if total_operations > 0 else 1.0
                
                # Get current resource usage safely
                current_resources = {}
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
                    memory = psutil.virtual_memory()
                    current_resources = {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_mb': memory.available / (1024 * 1024),
                        'active_tasks': len(self.active_operations),
                        'queue_length': len(self.active_operations)
                    }
                except Exception:
                    current_resources = {
                        'cpu_percent': 0.0,
                        'memory_percent': 0.0,
                        'memory_available_mb': 1000.0,
                        'active_tasks': 0,
                        'queue_length': 0
                    }
                
                # Check threshold compliance
                threshold_compliance = {
                    'success_rate_ok': overall_success_rate >= self.thresholds.min_success_rate,
                    'cpu_usage_ok': current_resources['cpu_percent'] <= self.thresholds.max_cpu_percent,
                    'memory_usage_ok': current_resources['memory_percent'] <= self.thresholds.max_memory_percent,
                    'queue_size_ok': current_resources['queue_length'] <= self.thresholds.max_queue_size
                }
                
                return {
                    'total_operations': total_operations,
                    'overall_success_rate': overall_success_rate,
                    'active_operations': len(self.active_operations),
                    'current_resources': current_resources,
                    'threshold_compliance': threshold_compliance,
                    'operation_statistics': {},  # Simplified to avoid blocking
                    'monitoring_active': self._monitoring_active
                }
        except Exception as e:
            # Return safe defaults if anything fails
            return {
                'total_operations': 0,
                'overall_success_rate': 1.0,
                'active_operations': 0,
                'current_resources': {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'memory_available_mb': 1000.0,
                    'active_tasks': 0,
                    'queue_length': 0
                },
                'threshold_compliance': {
                    'success_rate_ok': True,
                    'cpu_usage_ok': True,
                    'memory_usage_ok': True,
                    'queue_size_ok': True
                },
                'operation_statistics': {},
                'monitoring_active': False
            }
    
    def estimate_completion_time(self, operation_type: str, 
                               input_size: Optional[float] = None) -> Optional[float]:
        """Estimate completion time for an operation based on historical data."""
        with self._lock:
            if operation_type not in self.operation_stats:
                return None
            
            stats = self.operation_stats[operation_type]
            if stats['count'] == 0:
                return None
            
            # Use recent times for better estimation
            recent_times = list(stats['recent_times'])
            if not recent_times:
                return stats['avg_time']
            
            # Simple estimation based on recent average
            # In production, this could be more sophisticated with input size correlation
            recent_avg = sum(recent_times) / len(recent_times)
            
            # Add some buffer for safety
            estimated_time = recent_avg * 1.2
            
            return estimated_time
    
    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """Export performance metrics to JSON file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(settings.RESULTS_DIR, "metrics", f"performance_metrics_{timestamp}.json")
        
        with self._lock:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'recent_resource_usage': self.get_resource_usage(minutes=120),
                'thresholds': {
                    'voice_analysis_max_seconds': self.thresholds.voice_analysis_max_seconds,
                    'synthesis_max_seconds': self.thresholds.synthesis_max_seconds,
                    'max_cpu_percent': self.thresholds.max_cpu_percent,
                    'max_memory_percent': self.thresholds.max_memory_percent,
                    'max_queue_size': self.thresholds.max_queue_size,
                    'min_success_rate': self.thresholds.min_success_rate
                }
            }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Performance metrics exported to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            raise


class QueueManager:
    """Request queue manager with priority handling."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.queues: Dict[str, deque] = {
            'high': deque(),
            'normal': deque(),
            'low': deque()
        }
        self._lock = threading.Lock()
        self.processing_stats = {
            'total_queued': 0,
            'total_processed': 0,
            'current_queue_size': 0
        }
    
    def add_request(self, request_id: str, operation_type: str, 
                   priority: str = 'normal', metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add request to appropriate priority queue."""
        if priority not in self.queues:
            priority = 'normal'
        
        request_data = {
            'id': request_id,
            'type': operation_type,
            'priority': priority,
            'queued_at': datetime.now(),
            'metadata': metadata or {}
        }
        
        with self._lock:
            self.queues[priority].append(request_data)
            self.processing_stats['total_queued'] += 1
            self.processing_stats['current_queue_size'] += 1
    
    def get_next_request(self) -> Optional[Dict[str, Any]]:
        """Get next request from highest priority queue."""
        with self._lock:
            # Check queues in priority order
            for priority in ['high', 'normal', 'low']:
                if self.queues[priority]:
                    request = self.queues[priority].popleft()
                    self.processing_stats['total_processed'] += 1
                    self.processing_stats['current_queue_size'] -= 1
                    return request
            
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            queue_sizes = {priority: len(queue) for priority, queue in self.queues.items()}
            
            # Calculate estimated wait times
            estimated_wait_times = {}
            for priority in ['high', 'normal', 'low']:
                queue_size = queue_sizes[priority]
                if queue_size > 0:
                    # Estimate based on average processing time
                    avg_processing_time = 30.0  # Default estimate
                    estimated_wait_times[priority] = queue_size * avg_processing_time
                else:
                    estimated_wait_times[priority] = 0
            
            return {
                'queue_sizes': queue_sizes,
                'total_queue_size': sum(queue_sizes.values()),
                'estimated_wait_times': estimated_wait_times,
                'processing_stats': self.processing_stats.copy()
            }


class LazyPerformanceMonitor:
    """Lazy loading wrapper for PerformanceMonitor to avoid blocking imports."""
    
    def __init__(self):
        self._instance = None
    
    def __getattr__(self, name):
        if self._instance is None:
            self._instance = PerformanceMonitor()
        return getattr(self._instance, name)

class LazyQueueManager:
    """Lazy loading wrapper for QueueManager to avoid blocking imports."""
    
    def __init__(self):
        self._instance = None
    
    def __getattr__(self, name):
        if self._instance is None:
            self._instance = QueueManager(performance_monitor)
        return getattr(self._instance, name)

# Global service instances with lazy loading
performance_monitor = LazyPerformanceMonitor()
queue_manager = LazyQueueManager()

# Keep the getter functions for explicit access
_performance_monitor = None
_queue_manager = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance (lazy initialization)."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance (lazy initialization)."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager(get_performance_monitor())
    return _queue_manager