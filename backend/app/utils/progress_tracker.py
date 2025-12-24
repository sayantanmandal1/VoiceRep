"""
Progress tracking utilities for voice cloning operations.
"""

import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.services.performance_monitoring_service import performance_monitor, queue_manager


@dataclass
class ProgressUpdate:
    """Progress update data structure."""
    task_id: str
    operation_type: str
    progress: float  # 0-100
    stage: str
    message: str
    estimated_completion: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class ProgressTracker:
    """Utility class for tracking and reporting operation progress."""
    
    def __init__(self):
        self.active_trackers: Dict[str, Dict[str, Any]] = {}
    
    def start_tracking(
        self, 
        task_id: str, 
        operation_type: str, 
        total_steps: int = 100,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start tracking progress for an operation.
        
        Args:
            task_id: Unique identifier for the task
            operation_type: Type of operation being tracked
            total_steps: Total number of steps for completion
            metadata: Additional metadata for the operation
        """
        # Start performance monitoring
        performance_monitor.start_operation(operation_type, task_id, metadata)
        
        # Initialize progress tracking
        self.active_trackers[task_id] = {
            'operation_type': operation_type,
            'start_time': time.time(),
            'total_steps': total_steps,
            'current_step': 0,
            'current_stage': 'initializing',
            'metadata': metadata or {}
        }
        
        # Add to queue if needed
        priority = metadata.get('priority', 'normal') if metadata else 'normal'
        queue_manager.add_request(task_id, operation_type, priority, metadata)
    
    def update_progress(
        self, 
        task_id: str, 
        step: int, 
        stage: str, 
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressUpdate:
        """
        Update progress for a tracked operation.
        
        Args:
            task_id: Task identifier
            step: Current step number
            stage: Current stage name
            message: Progress message
            metadata: Additional metadata
            
        Returns:
            ProgressUpdate object with current progress information
        """
        if task_id not in self.active_trackers:
            raise ValueError(f"Task {task_id} is not being tracked")
        
        tracker = self.active_trackers[task_id]
        tracker['current_step'] = step
        tracker['current_stage'] = stage
        
        # Calculate progress percentage
        progress = min(100.0, (step / tracker['total_steps']) * 100)
        
        # Estimate completion time based on current progress and historical data
        estimated_completion = self._estimate_completion(task_id, progress)
        
        # Update metadata
        if metadata:
            tracker['metadata'].update(metadata)
        
        return ProgressUpdate(
            task_id=task_id,
            operation_type=tracker['operation_type'],
            progress=progress,
            stage=stage,
            message=message,
            estimated_completion=estimated_completion,
            metadata=tracker['metadata']
        )
    
    def complete_tracking(
        self, 
        task_id: str, 
        success: bool = True, 
        error_message: Optional[str] = None,
        final_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Complete progress tracking for an operation.
        
        Args:
            task_id: Task identifier
            success: Whether the operation completed successfully
            error_message: Error message if operation failed
            final_metadata: Final metadata for the operation
        """
        if task_id not in self.active_trackers:
            return
        
        tracker = self.active_trackers[task_id]
        
        # Update final metadata
        if final_metadata:
            tracker['metadata'].update(final_metadata)
        
        # End performance monitoring
        performance_monitor.end_operation(
            task_id, 
            success=success, 
            error_message=error_message,
            additional_metadata=tracker['metadata']
        )
        
        # Remove from active trackers
        del self.active_trackers[task_id]
    
    def get_progress(self, task_id: str) -> Optional[ProgressUpdate]:
        """
        Get current progress for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Current progress information or None if task not found
        """
        if task_id not in self.active_trackers:
            return None
        
        tracker = self.active_trackers[task_id]
        progress = (tracker['current_step'] / tracker['total_steps']) * 100
        estimated_completion = self._estimate_completion(task_id, progress)
        
        return ProgressUpdate(
            task_id=task_id,
            operation_type=tracker['operation_type'],
            progress=progress,
            stage=tracker['current_stage'],
            message=f"Step {tracker['current_step']} of {tracker['total_steps']}",
            estimated_completion=estimated_completion,
            metadata=tracker['metadata']
        )
    
    def _estimate_completion(self, task_id: str, current_progress: float) -> Optional[datetime]:
        """
        Estimate completion time based on current progress and historical data.
        
        Args:
            task_id: Task identifier
            current_progress: Current progress percentage (0-100)
            
        Returns:
            Estimated completion datetime or None
        """
        if task_id not in self.active_trackers:
            return None
        
        tracker = self.active_trackers[task_id]
        elapsed_time = time.time() - tracker['start_time']
        
        if current_progress <= 0:
            # Use historical data for initial estimate
            estimated_time = performance_monitor.estimate_completion_time(
                tracker['operation_type']
            )
            if estimated_time:
                return datetime.now() + timedelta(seconds=estimated_time)
            return None
        
        # Calculate remaining time based on current progress
        estimated_total_time = elapsed_time / (current_progress / 100)
        remaining_time = estimated_total_time - elapsed_time
        
        # Add some buffer for safety
        remaining_time *= 1.1
        
        return datetime.now() + timedelta(seconds=max(0, remaining_time))
    
    def get_all_active_tasks(self) -> Dict[str, ProgressUpdate]:
        """
        Get progress information for all active tasks.
        
        Returns:
            Dictionary mapping task IDs to their progress information
        """
        result = {}
        for task_id in self.active_trackers:
            progress = self.get_progress(task_id)
            if progress:
                result[task_id] = progress
        return result
    
    def cleanup_stale_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up tasks that have been running for too long.
        
        Args:
            max_age_hours: Maximum age in hours before considering a task stale
            
        Returns:
            Number of tasks cleaned up
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        stale_tasks = []
        
        for task_id, tracker in self.active_trackers.items():
            if current_time - tracker['start_time'] > max_age_seconds:
                stale_tasks.append(task_id)
        
        # Clean up stale tasks
        for task_id in stale_tasks:
            self.complete_tracking(
                task_id, 
                success=False, 
                error_message="Task timed out - cleaned up as stale"
            )
        
        return len(stale_tasks)


# Global progress tracker instance
progress_tracker = ProgressTracker()


def create_progress_callback(task_id: str, total_steps: int = 100) -> Callable[[int, str], None]:
    """
    Create a progress callback function for use with synthesis operations.
    
    Args:
        task_id: Task identifier
        total_steps: Total number of steps for the operation
        
    Returns:
        Callback function that can be used to update progress
    """
    def progress_callback(step: int, message: str) -> None:
        """Progress callback function."""
        try:
            progress_tracker.update_progress(task_id, step, "processing", message)
        except ValueError:
            # Task not being tracked, ignore
            pass
    
    return progress_callback


def get_queue_position(task_id: str) -> Optional[int]:
    """
    Get the queue position for a task.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Queue position (1-based) or None if not in queue
    """
    # This is a simplified implementation
    # In a real system, you'd need to track queue positions more precisely
    queue_status = queue_manager.get_queue_status()
    total_queue_size = queue_status.get('total_queue_size', 0)
    
    if total_queue_size == 0:
        return None
    
    # For now, return a rough estimate based on when the task was added
    # In production, you'd maintain proper queue ordering
    return min(total_queue_size, 5)  # Simplified estimate