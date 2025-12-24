"""
Comprehensive error handling and recovery system for Voice Style Replication.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Error categories for classification and handling."""
    FILE_PROCESSING = "file_processing"
    VOICE_ANALYSIS = "voice_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    SYSTEM = "system"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Available recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    NOTIFY_USER = "notify_user"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class ErrorInfo(BaseModel):
    """Structured error information."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: int = 1  # seconds
    recovery_actions: List[RecoveryAction] = []
    is_retryable: bool = True


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for service protection."""
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Service {self.service_name} is temporarily unavailable"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class ErrorRecoveryManager:
    """Manages error recovery strategies and circuit breakers."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorInfo] = []
        self.max_history_size = 1000
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(service_name)
        return self.circuit_breakers[service_name]
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """Handle and classify error."""
        error_info = self._classify_error(error, category, context)
        self._log_error(error_info)
        self._store_error(error_info)
        
        return error_info
    
    def _classify_error(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """Classify error and determine recovery actions."""
        error_id = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Determine severity and recovery actions based on error type and category
        severity, recovery_actions, is_retryable = self._analyze_error(error, category)
        
        return ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            details=context or {},
            timestamp=datetime.now(),
            recovery_actions=recovery_actions,
            is_retryable=is_retryable
        )
    
    def _analyze_error(
        self,
        error: Exception,
        category: ErrorCategory
    ) -> tuple[ErrorSeverity, List[RecoveryAction], bool]:
        """Analyze error to determine severity and recovery actions."""
        
        # File processing errors
        if category == ErrorCategory.FILE_PROCESSING:
            if "file size" in str(error).lower():
                return ErrorSeverity.LOW, [RecoveryAction.NOTIFY_USER], False
            elif "format" in str(error).lower():
                return ErrorSeverity.LOW, [RecoveryAction.NOTIFY_USER], False
            elif "corrupted" in str(error).lower():
                return ErrorSeverity.MEDIUM, [RecoveryAction.RETRY, RecoveryAction.NOTIFY_USER], True
            else:
                return ErrorSeverity.MEDIUM, [RecoveryAction.RETRY], True
        
        # Voice analysis errors
        elif category == ErrorCategory.VOICE_ANALYSIS:
            if "quality" in str(error).lower():
                return ErrorSeverity.MEDIUM, [RecoveryAction.NOTIFY_USER], False
            elif "timeout" in str(error).lower():
                return ErrorSeverity.MEDIUM, [RecoveryAction.RETRY], True
            else:
                return ErrorSeverity.HIGH, [RecoveryAction.RETRY, RecoveryAction.FALLBACK], True
        
        # Synthesis errors
        elif category == ErrorCategory.SYNTHESIS:
            if "text" in str(error).lower():
                return ErrorSeverity.LOW, [RecoveryAction.NOTIFY_USER], False
            elif "timeout" in str(error).lower():
                return ErrorSeverity.MEDIUM, [RecoveryAction.RETRY], True
            elif "model" in str(error).lower():
                return ErrorSeverity.HIGH, [RecoveryAction.FALLBACK], True
            else:
                return ErrorSeverity.HIGH, [RecoveryAction.RETRY], True
        
        # System errors
        elif category == ErrorCategory.SYSTEM:
            if "memory" in str(error).lower() or "resource" in str(error).lower():
                return ErrorSeverity.CRITICAL, [RecoveryAction.ESCALATE], False
            else:
                return ErrorSeverity.HIGH, [RecoveryAction.RETRY, RecoveryAction.ESCALATE], True
        
        # Default classification
        return ErrorSeverity.MEDIUM, [RecoveryAction.RETRY], True
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = f"[{error_info.error_id}] {error_info.category}: {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={"error_info": error_info.model_dump()})
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={"error_info": error_info.model_dump()})
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={"error_info": error_info.model_dump()})
        else:
            logger.info(log_message, extra={"error_info": error_info.model_dump()})
    
    def _store_error(self, error_info: ErrorInfo):
        """Store error in history."""
        self.error_history.append(error_info)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """Determine if error should be retried."""
        return (
            error_info.is_retryable and
            error_info.retry_count < error_info.max_retries and
            RecoveryAction.RETRY in error_info.recovery_actions
        )
    
    def get_retry_delay(self, error_info: ErrorInfo) -> int:
        """Calculate retry delay with exponential backoff."""
        base_delay = error_info.retry_delay
        return min(base_delay * (2 ** error_info.retry_count), 60)  # Max 60 seconds
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0}
        
        recent_errors = [
            e for e in self.error_history
            if (datetime.now() - e.timestamp).seconds < 3600  # Last hour
        ]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "circuit_breaker_states": {
                name: breaker.state for name, breaker in self.circuit_breakers.items()
            }
        }


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


# Exception handlers for FastAPI
async def validation_exception_handler(request: Request, exc: Exception):
    """Handle validation errors."""
    error_info = error_recovery_manager.handle_error(
        exc, ErrorCategory.VALIDATION, {"request_url": str(request.url)}
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error_id": error_info.error_id,
            "detail": error_info.message,
            "category": error_info.category,
            "recovery_suggestions": [
                "Check input format and try again",
                "Ensure all required fields are provided",
                "Verify data types match expected format"
            ]
        }
    )


async def file_processing_exception_handler(request: Request, exc: Exception):
    """Handle file processing errors."""
    error_info = error_recovery_manager.handle_error(
        exc, ErrorCategory.FILE_PROCESSING, {"request_url": str(request.url)}
    )
    
    status_code = 400
    if "size" in str(exc).lower():
        status_code = 413
    elif "format" in str(exc).lower():
        status_code = 415
    
    recovery_suggestions = []
    if RecoveryAction.RETRY in error_info.recovery_actions:
        recovery_suggestions.append("Try uploading the file again")
    if RecoveryAction.NOTIFY_USER in error_info.recovery_actions:
        recovery_suggestions.extend([
            "Check file format (supported: mp3, wav, flac, m4a, mp4, avi, mov, mkv)",
            "Ensure file size is under 100MB",
            "Verify file is not corrupted"
        ])
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error_id": error_info.error_id,
            "detail": error_info.message,
            "category": error_info.category,
            "is_retryable": error_info.is_retryable,
            "recovery_suggestions": recovery_suggestions
        }
    )


async def synthesis_exception_handler(request: Request, exc: Exception):
    """Handle synthesis errors."""
    error_info = error_recovery_manager.handle_error(
        exc, ErrorCategory.SYNTHESIS, {"request_url": str(request.url)}
    )
    
    status_code = 500
    if "timeout" in str(exc).lower():
        status_code = 408
    elif "model" in str(exc).lower():
        status_code = 503
    
    recovery_suggestions = []
    if RecoveryAction.RETRY in error_info.recovery_actions:
        recovery_suggestions.append("Retry synthesis operation")
    if RecoveryAction.FALLBACK in error_info.recovery_actions:
        recovery_suggestions.append("Try with different voice settings")
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error_id": error_info.error_id,
            "detail": error_info.message,
            "category": error_info.category,
            "is_retryable": error_info.is_retryable,
            "retry_after": error_recovery_manager.get_retry_delay(error_info),
            "recovery_suggestions": recovery_suggestions
        }
    )


async def system_exception_handler(request: Request, exc: Exception):
    """Handle system errors."""
    error_info = error_recovery_manager.handle_error(
        exc, ErrorCategory.SYSTEM, {"request_url": str(request.url)}
    )
    
    return JSONResponse(
        status_code=503,
        content={
            "error_id": error_info.error_id,
            "detail": "System temporarily unavailable",
            "category": error_info.category,
            "is_retryable": error_info.is_retryable,
            "retry_after": error_recovery_manager.get_retry_delay(error_info),
            "recovery_suggestions": [
                "Please try again in a few moments",
                "Contact support if the problem persists"
            ]
        }
    )


# Utility functions for error handling
def handle_with_recovery(func, category: ErrorCategory, context: Optional[Dict[str, Any]] = None):
    """Decorator for automatic error handling and recovery."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_info = error_recovery_manager.handle_error(e, category, context)
            
            if error_recovery_manager.should_retry(error_info):
                # Implement retry logic here if needed
                pass
            
            raise e
    return wrapper


def with_circuit_breaker(service_name: str):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            circuit_breaker = error_recovery_manager.get_circuit_breaker(service_name)
            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator