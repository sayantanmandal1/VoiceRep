"""
Disabled performance monitoring service to avoid import blocking issues.
This is a temporary stub to allow the application to run.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Stub performance metrics."""
    pass


class PerformanceMonitor:
    """Stub performance monitor that does nothing."""
    
    def __init__(self):
        self._monitoring_active = False
    
    def start_monitoring(self) -> None:
        """Stub method."""
        pass
    
    def stop_monitoring(self) -> None:
        """Stub method."""
        pass
    
    @property
    def is_running(self) -> bool:
        """Always return False for stub."""
        return False
    
    def start_operation(self, operation_type: str, operation_id: str, metadata: Dict = None) -> None:
        """Stub method."""
        pass
    
    def end_operation(self, operation_type: str, operation_id: str, success: bool = True, 
                     error_message: str = None, metadata: Dict = None) -> None:
        """Stub method."""
        pass
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Return empty summary."""
        return {
            'queue_length': 0,
            'system_load': 0.0,
            'memory_usage': 0.0,
            'active_tasks': 0,
            'current_resources': {},
            'threshold_compliance': {}
        }
    
    def get_operation_statistics(self, operation_type: str = None) -> Dict[str, Any]:
        """Return empty stats."""
        return {}
    
    def get_resource_usage(self, minutes: int = 30) -> List[Dict]:
        """Return empty usage data."""
        return []
    
    def estimate_completion_time(self, operation_type: str, input_size: int = None) -> Optional[float]:
        """Return None for no estimate."""
        return None
    
    def export_metrics(self) -> str:
        """Return empty file path."""
        return "/tmp/empty_metrics.json"


class QueueManager:
    """Stub queue manager that does nothing."""
    
    def __init__(self, performance_monitor):
        pass
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Return empty queue status."""
        return {
            'queue_sizes': {'high': 0, 'normal': 0, 'low': 0},
            'total_queue_size': 0,
            'estimated_wait_times': {'high': 0, 'normal': 0, 'low': 0},
            'processing_stats': {}
        }


# Global service instances (stubs)
performance_monitor = PerformanceMonitor()
queue_manager = QueueManager(performance_monitor)

# Getter functions
def get_performance_monitor() -> PerformanceMonitor:
    return performance_monitor

def get_queue_manager() -> QueueManager:
    return queue_manager