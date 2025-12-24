"""
Enhanced batch processing service for multiple voice synthesis requests.
"""

import time
import uuid
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque

from app.core.config import settings
from app.services.performance_monitoring_service import performance_monitor
from app.services.model_cache_service import model_cache_service
from app.schemas.synthesis import (
    SynthesisRequest, BatchSynthesisRequest, BatchSynthesisResult,
    SynthesisStatus, SynthesisResult
)

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Batch processing priority levels."""
    HIGH = 1
    NORMAL = 5
    LOW = 10


@dataclass
class BatchJob:
    """Batch processing job data structure."""
    batch_id: str
    batch_name: Optional[str]
    requests: List[SynthesisRequest]
    priority: BatchPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: SynthesisStatus = SynthesisStatus.PENDING
    results: List[SynthesisResult] = field(default_factory=list)
    progress: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_requests(self) -> int:
        return len(self.requests)
    
    @property
    def completed_requests(self) -> int:
        return len([r for r in self.results if r.status == SynthesisStatus.COMPLETED])
    
    @property
    def failed_requests(self) -> int:
        return len([r for r in self.results if r.status == SynthesisStatus.FAILED])
    
    @property
    def processing_time(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0


@dataclass
class BatchProcessingStats:
    """Batch processing performance statistics."""
    total_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    average_batch_time: float = 0.0
    average_request_time: float = 0.0
    throughput_requests_per_minute: float = 0.0
    queue_length: int = 0
    active_batches: int = 0


class BatchOptimizer:
    """Optimizer for batch processing efficiency."""
    
    def __init__(self):
        self.model_usage_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.batch_performance_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def optimize_batch_order(self, batches: List[BatchJob]) -> List[BatchJob]:
        """Optimize batch processing order for efficiency."""
        try:
            # Sort by priority first, then by optimization criteria
            def batch_score(batch: BatchJob) -> Tuple[int, float]:
                priority_score = batch.priority.value
                
                # Calculate efficiency score
                efficiency_score = 0.0
                
                # Favor batches with similar voice models (cache efficiency)
                voice_models = [req.voice_model_id for req in batch.requests]
                unique_models = len(set(voice_models))
                model_diversity_penalty = unique_models / len(voice_models) if voice_models else 1.0
                efficiency_score += (1.0 - model_diversity_penalty) * 10
                
                # Favor smaller batches for faster turnaround
                size_bonus = max(0, 10 - batch.total_requests) * 0.5
                efficiency_score += size_bonus
                
                # Consider waiting time
                wait_time_penalty = (datetime.now() - batch.created_at).total_seconds() / 3600  # Hours
                efficiency_score += wait_time_penalty * 2
                
                return (priority_score, -efficiency_score)  # Negative for descending order
            
            optimized_batches = sorted(batches, key=batch_score)
            
            logger.info(f"Optimized batch order for {len(batches)} batches")
            return optimized_batches
            
        except Exception as e:
            logger.error(f"Batch optimization failed: {str(e)}")
            return batches  # Return original order on error
    
    def suggest_batch_grouping(self, requests: List[SynthesisRequest]) -> List[List[SynthesisRequest]]:
        """Suggest optimal grouping of requests into batches."""
        try:
            # Group by voice model for cache efficiency
            model_groups: Dict[str, List[SynthesisRequest]] = defaultdict(list)
            
            for request in requests:
                model_groups[request.voice_model_id].append(request)
            
            # Create batches with optimal size
            optimal_batch_size = getattr(settings, 'OPTIMAL_BATCH_SIZE', 5)
            batches = []
            
            for model_id, model_requests in model_groups.items():
                # Split large groups into optimal-sized batches
                for i in range(0, len(model_requests), optimal_batch_size):
                    batch_requests = model_requests[i:i + optimal_batch_size]
                    batches.append(batch_requests)
            
            logger.info(f"Suggested {len(batches)} batches from {len(requests)} requests")
            return batches
            
        except Exception as e:
            logger.error(f"Batch grouping failed: {str(e)}")
            return [requests]  # Return single batch on error
    
    def record_batch_performance(self, batch: BatchJob) -> None:
        """Record batch performance for future optimization."""
        try:
            with self._lock:
                performance_record = {
                    'batch_id': batch.batch_id,
                    'total_requests': batch.total_requests,
                    'processing_time': batch.processing_time,
                    'success_rate': batch.completed_requests / batch.total_requests if batch.total_requests > 0 else 0,
                    'unique_models': len(set(req.voice_model_id for req in batch.requests)),
                    'completed_at': batch.completed_at
                }
                
                self.batch_performance_history.append(performance_record)
                
                # Update model usage patterns
                for request in batch.requests:
                    self.model_usage_patterns[request.voice_model_id].append(datetime.now())
                    
                    # Keep only recent usage (last 24 hours)
                    cutoff = datetime.now() - timedelta(hours=24)
                    self.model_usage_patterns[request.voice_model_id] = [
                        usage_time for usage_time in self.model_usage_patterns[request.voice_model_id]
                        if usage_time >= cutoff
                    ]
                
        except Exception as e:
            logger.error(f"Failed to record batch performance: {str(e)}")
    
    def get_popular_models(self, hours: int = 24) -> List[Tuple[str, int]]:
        """Get most popular voice models in recent hours."""
        try:
            cutoff = datetime.now() - timedelta(hours=hours)
            
            with self._lock:
                model_counts = {}
                for model_id, usage_times in self.model_usage_patterns.items():
                    recent_usage = [t for t in usage_times if t >= cutoff]
                    if recent_usage:
                        model_counts[model_id] = len(recent_usage)
                
                return sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
                
        except Exception as e:
            logger.error(f"Failed to get popular models: {str(e)}")
            return []


class BatchProcessingService:
    """Enhanced batch processing service with optimization."""
    
    def __init__(self):
        self.batch_queue: deque = deque()
        self.active_batches: Dict[str, BatchJob] = {}
        self.completed_batches: Dict[str, BatchJob] = {}
        self.stats = BatchProcessingStats()
        self.optimizer = BatchOptimizer()
        
        # Processing configuration
        self.max_concurrent_batches = getattr(settings, 'MAX_CONCURRENT_BATCHES', 3)
        self.max_batch_size = getattr(settings, 'MAX_BATCH_SIZE', 10)
        self.batch_timeout_minutes = getattr(settings, 'BATCH_TIMEOUT_MINUTES', 30)
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_batches)
        self._processing_active = False
        self._processing_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Progress callbacks
        self.progress_callbacks: Dict[str, Callable] = {}
    
    def start_processing(self) -> None:
        """Start batch processing service."""
        if self._processing_active:
            return
        
        self._processing_active = True
        self._processing_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True
        )
        self._processing_thread.start()
        
        # Preload popular models for efficiency
        self._preload_popular_models()
        
        logger.info("Batch processing service started")
    
    def stop_processing(self) -> None:
        """Stop batch processing service."""
        self._processing_active = False
        
        if self._processing_thread:
            self._processing_thread.join(timeout=10.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Batch processing service stopped")
    
    def submit_batch(self, batch_request: BatchSynthesisRequest, 
                    user_id: Optional[str] = None) -> str:
        """
        Submit a batch processing job.
        
        Args:
            batch_request: Batch synthesis request
            user_id: Optional user identifier
            
        Returns:
            Batch ID for tracking
        """
        try:
            batch_id = str(uuid.uuid4())
            
            # Create batch job
            batch_job = BatchJob(
                batch_id=batch_id,
                batch_name=batch_request.batch_name,
                requests=batch_request.requests,
                priority=BatchPriority(batch_request.priority),
                created_at=datetime.now(),
                metadata={
                    'user_id': user_id,
                    'submitted_at': datetime.now().isoformat(),
                    'request_count': len(batch_request.requests)
                }
            )
            
            # Validate batch
            if not self._validate_batch(batch_job):
                raise ValueError("Batch validation failed")
            
            # Add to queue
            with self._lock:
                self.batch_queue.append(batch_job)
                self.stats.total_batches += 1
                self.stats.total_requests += batch_job.total_requests
                self.stats.queue_length += 1
            
            logger.info(f"Batch {batch_id} submitted with {batch_job.total_requests} requests")
            return batch_id
            
        except Exception as e:
            logger.error(f"Failed to submit batch: {str(e)}")
            raise
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchSynthesisResult]:
        """Get status of a batch job."""
        try:
            with self._lock:
                # Check active batches
                if batch_id in self.active_batches:
                    batch = self.active_batches[batch_id]
                    return self._create_batch_result(batch)
                
                # Check completed batches
                if batch_id in self.completed_batches:
                    batch = self.completed_batches[batch_id]
                    return self._create_batch_result(batch)
                
                # Check queue
                for batch in self.batch_queue:
                    if batch.batch_id == batch_id:
                        return self._create_batch_result(batch)
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get batch status: {str(e)}")
            return None
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch job."""
        try:
            with self._lock:
                # Remove from queue if not started
                for i, batch in enumerate(self.batch_queue):
                    if batch.batch_id == batch_id:
                        del self.batch_queue[i]
                        self.stats.queue_length -= 1
                        logger.info(f"Batch {batch_id} cancelled from queue")
                        return True
                
                # Mark active batch as cancelled
                if batch_id in self.active_batches:
                    batch = self.active_batches[batch_id]
                    batch.status = SynthesisStatus.FAILED
                    batch.error_message = "Cancelled by user"
                    logger.info(f"Active batch {batch_id} marked for cancellation")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel batch: {str(e)}")
            return False
    
    def register_progress_callback(self, batch_id: str, callback: Callable) -> None:
        """Register progress callback for a batch."""
        self.progress_callbacks[batch_id] = callback
    
    def get_processing_statistics(self) -> BatchProcessingStats:
        """Get comprehensive processing statistics."""
        with self._lock:
            # Update dynamic statistics
            self.stats.queue_length = len(self.batch_queue)
            self.stats.active_batches = len(self.active_batches)
            
            # Calculate averages
            if self.stats.completed_batches > 0:
                completed_batch_times = [
                    batch.processing_time for batch in self.completed_batches.values()
                    if batch.processing_time > 0
                ]
                if completed_batch_times:
                    self.stats.average_batch_time = sum(completed_batch_times) / len(completed_batch_times)
            
            if self.stats.completed_requests > 0:
                # Estimate average request time
                total_batch_time = sum(
                    batch.processing_time for batch in self.completed_batches.values()
                    if batch.processing_time > 0
                )
                if total_batch_time > 0:
                    self.stats.average_request_time = total_batch_time / self.stats.completed_requests
            
            # Calculate throughput
            recent_batches = [
                batch for batch in self.completed_batches.values()
                if batch.completed_at and batch.completed_at >= datetime.now() - timedelta(hours=1)
            ]
            if recent_batches:
                recent_requests = sum(batch.completed_requests for batch in recent_batches)
                self.stats.throughput_requests_per_minute = recent_requests / 60.0
            
            return self.stats
    
    def _processing_worker(self) -> None:
        """Main processing worker thread."""
        while self._processing_active:
            try:
                # Get next batch to process
                batch_to_process = None
                
                with self._lock:
                    if (self.batch_queue and 
                        len(self.active_batches) < self.max_concurrent_batches):
                        
                        # Optimize batch order
                        queue_list = list(self.batch_queue)
                        optimized_batches = self.optimizer.optimize_batch_order(queue_list)
                        
                        # Get highest priority batch
                        batch_to_process = optimized_batches[0]
                        
                        # Remove from queue and add to active
                        self.batch_queue.remove(batch_to_process)
                        self.active_batches[batch_to_process.batch_id] = batch_to_process
                        self.stats.queue_length -= 1
                        self.stats.active_batches += 1
                
                if batch_to_process:
                    # Submit batch for processing
                    future = self.executor.submit(self._process_batch, batch_to_process)
                    
                    # Don't wait for completion here - let it run concurrently
                    logger.info(f"Started processing batch {batch_to_process.batch_id}")
                
                # Sleep briefly to prevent busy waiting
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in processing worker: {str(e)}")
                time.sleep(5.0)  # Wait longer on error
    
    def _process_batch(self, batch: BatchJob) -> None:
        """Process a single batch job."""
        operation_id = f"batch_processing_{batch.batch_id}"
        
        # Start performance monitoring
        performance_monitor.start_operation(
            'batch_processing',
            operation_id,
            {
                'batch_id': batch.batch_id,
                'request_count': batch.total_requests,
                'priority': batch.priority.value
            }
        )
        
        try:
            batch.started_at = datetime.now()
            batch.status = SynthesisStatus.PROCESSING
            
            logger.info(f"Processing batch {batch.batch_id} with {batch.total_requests} requests")
            
            # Preload required voice models
            self._preload_batch_models(batch)
            
            # Process requests with optimized concurrency
            batch.results = self._process_batch_requests(batch)
            
            # Update batch status
            batch.completed_at = datetime.now()
            
            if batch.failed_requests == 0:
                batch.status = SynthesisStatus.COMPLETED
            elif batch.completed_requests > 0:
                batch.status = SynthesisStatus.PARTIALLY_COMPLETED
            else:
                batch.status = SynthesisStatus.FAILED
                batch.error_message = "All requests failed"
            
            # Update statistics
            with self._lock:
                self.stats.completed_batches += 1
                self.stats.completed_requests += batch.completed_requests
                self.stats.failed_requests += batch.failed_requests
                
                # Move from active to completed
                if batch.batch_id in self.active_batches:
                    del self.active_batches[batch.batch_id]
                    self.stats.active_batches -= 1
                
                self.completed_batches[batch.batch_id] = batch
            
            # Record performance for optimization
            self.optimizer.record_batch_performance(batch)
            
            # End performance monitoring
            performance_monitor.end_operation(
                operation_id,
                success=batch.status in [SynthesisStatus.COMPLETED, SynthesisStatus.PARTIALLY_COMPLETED],
                additional_metadata={
                    'completed_requests': batch.completed_requests,
                    'failed_requests': batch.failed_requests,
                    'processing_time': batch.processing_time,
                    'success_rate': batch.completed_requests / batch.total_requests if batch.total_requests > 0 else 0
                }
            )
            
            logger.info(f"Batch {batch.batch_id} completed: {batch.completed_requests}/{batch.total_requests} successful")
            
        except Exception as e:
            batch.status = SynthesisStatus.FAILED
            batch.error_message = str(e)
            batch.completed_at = datetime.now()
            
            # Update statistics
            with self._lock:
                self.stats.failed_batches += 1
                if batch.batch_id in self.active_batches:
                    del self.active_batches[batch.batch_id]
                    self.stats.active_batches -= 1
                self.completed_batches[batch.batch_id] = batch
            
            # End performance monitoring with error
            performance_monitor.end_operation(
                operation_id,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Batch {batch.batch_id} failed: {str(e)}")
    
    def _process_batch_requests(self, batch: BatchJob) -> List[SynthesisResult]:
        """Process all requests in a batch with optimal concurrency."""
        results = []
        
        try:
            # Determine optimal concurrency level
            max_concurrent = min(
                len(batch.requests),
                getattr(settings, 'MAX_CONCURRENT_SYNTHESIS', 3)
            )
            
            # Process requests with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                # Submit all requests
                future_to_request = {
                    executor.submit(self._process_single_request, request, batch.batch_id): request
                    for request in batch.requests
                }
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_request):
                    request = future_to_request[future]
                    
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per request
                        results.append(result)
                        
                        completed_count += 1
                        batch.progress = (completed_count / batch.total_requests) * 100
                        
                        # Call progress callback if registered
                        if batch.batch_id in self.progress_callbacks:
                            try:
                                self.progress_callbacks[batch.batch_id](batch.progress, f"Completed {completed_count}/{batch.total_requests}")
                            except:
                                pass  # Ignore callback errors
                        
                    except Exception as e:
                        # Create error result for failed request
                        error_result = SynthesisResult(
                            task_id=f"error_{request.voice_model_id}_{int(time.time())}",
                            status=SynthesisStatus.FAILED,
                            error_message=str(e),
                            created_at=datetime.now(),
                            completed_at=datetime.now()
                        )
                        results.append(error_result)
                        
                        completed_count += 1
                        batch.progress = (completed_count / batch.total_requests) * 100
                        
                        logger.error(f"Request failed in batch {batch.batch_id}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch request processing failed: {str(e)}")
            # Return error results for all requests
            return [
                SynthesisResult(
                    task_id=f"error_{i}_{int(time.time())}",
                    status=SynthesisStatus.FAILED,
                    error_message=f"Batch processing error: {str(e)}",
                    created_at=datetime.now(),
                    completed_at=datetime.now()
                )
                for i in range(len(batch.requests))
            ]
    
    def _process_single_request(self, request: SynthesisRequest, batch_id: str) -> SynthesisResult:
        """Process a single synthesis request."""
        try:
            # This would integrate with the actual synthesis service
            # For now, simulate processing
            
            start_time = time.time()
            
            # Simulate processing time based on text length
            processing_time = min(5.0, len(request.text) * 0.05)  # Max 5 seconds
            time.sleep(processing_time)
            
            # Create successful result
            result = SynthesisResult(
                task_id=f"batch_{batch_id}_{request.voice_model_id}_{int(time.time() * 1000)}",
                status=SynthesisStatus.COMPLETED,
                audio_url=f"/api/v1/synthesis/results/batch_{batch_id}_{int(time.time())}.wav",
                duration=processing_time * 2,  # Simulate audio duration
                file_size=int(processing_time * 44100 * 2),  # Simulate file size
                processing_time=processing_time,
                created_at=datetime.now(),
                completed_at=datetime.now(),
                metadata={
                    'batch_id': batch_id,
                    'text_length': len(request.text),
                    'voice_model_id': request.voice_model_id
                }
            )
            
            return result
            
        except Exception as e:
            return SynthesisResult(
                task_id=f"error_{batch_id}_{int(time.time())}",
                status=SynthesisStatus.FAILED,
                error_message=str(e),
                created_at=datetime.now(),
                completed_at=datetime.now()
            )
    
    def _preload_batch_models(self, batch: BatchJob) -> None:
        """Preload voice models required for batch processing."""
        try:
            # Get unique voice models in batch
            voice_models = list(set(req.voice_model_id for req in batch.requests))
            
            # Preload models
            preload_results = model_cache_service.preload_models(voice_models)
            
            successful_preloads = sum(1 for success in preload_results.values() if success)
            logger.info(f"Preloaded {successful_preloads}/{len(voice_models)} models for batch {batch.batch_id}")
            
        except Exception as e:
            logger.error(f"Failed to preload models for batch {batch.batch_id}: {str(e)}")
    
    def _preload_popular_models(self) -> None:
        """Preload popular voice models for better performance."""
        try:
            popular_models = self.optimizer.get_popular_models(hours=24)
            
            if popular_models:
                # Preload top 5 most popular models
                top_models = [model_id for model_id, _ in popular_models[:5]]
                preload_results = model_cache_service.preload_models(top_models)
                
                successful_preloads = sum(1 for success in preload_results.values() if success)
                logger.info(f"Preloaded {successful_preloads}/{len(top_models)} popular models")
                
        except Exception as e:
            logger.error(f"Failed to preload popular models: {str(e)}")
    
    def _validate_batch(self, batch: BatchJob) -> bool:
        """Validate batch job before processing."""
        try:
            # Check batch size limits
            if batch.total_requests > self.max_batch_size:
                logger.error(f"Batch {batch.batch_id} exceeds size limit: {batch.total_requests} > {self.max_batch_size}")
                return False
            
            # Validate individual requests
            for i, request in enumerate(batch.requests):
                if not request.text or not request.text.strip():
                    logger.error(f"Batch {batch.batch_id} request {i} has empty text")
                    return False
                
                if len(request.text) > 1000:  # Character limit from requirements
                    logger.error(f"Batch {batch.batch_id} request {i} text too long: {len(request.text)} chars")
                    return False
                
                if not request.voice_model_id:
                    logger.error(f"Batch {batch.batch_id} request {i} missing voice model ID")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Batch validation failed: {str(e)}")
            return False
    
    def _create_batch_result(self, batch: BatchJob) -> BatchSynthesisResult:
        """Create batch result from batch job."""
        return BatchSynthesisResult(
            batch_id=batch.batch_id,
            status=batch.status,
            total_requests=batch.total_requests,
            completed_requests=batch.completed_requests,
            failed_requests=batch.failed_requests,
            results=batch.results,
            processing_time=batch.processing_time,
            created_at=batch.created_at,
            completed_at=batch.completed_at
        )


# Global service instance
batch_processing_service = BatchProcessingService()