"""
Comprehensive logging and monitoring configuration for Voice Style Replication.
"""

import os
import sys
import json
import logging
import logging.config
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from app.core.config import settings


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def _serialize_value(self, value):
        """Recursively serialize values, handling datetime objects."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        else:
            return value
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present, serializing datetime objects
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        if hasattr(record, 'task_id'):
            log_entry['task_id'] = record.task_id
        
        if hasattr(record, 'error_info'):
            log_entry['error_info'] = self._serialize_value(record.error_info)
        
        if hasattr(record, 'performance_metrics'):
            log_entry['performance_metrics'] = self._serialize_value(record.performance_metrics)
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        try:
            return json.dumps(log_entry, ensure_ascii=False, cls=DateTimeEncoder)
        except (TypeError, ValueError) as e:
            # Fallback to string representation if JSON serialization fails
            log_entry['serialization_error'] = str(e)
            # Convert all values to strings as fallback
            safe_log_entry = {k: str(v) for k, v in log_entry.items()}
            return json.dumps(safe_log_entry, ensure_ascii=False)


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter performance logs."""
        return hasattr(record, 'performance_metrics') or 'performance' in record.getMessage().lower()


class ErrorFilter(logging.Filter):
    """Filter for error-related log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter error logs."""
        return record.levelno >= logging.ERROR or hasattr(record, 'error_info')


class SecurityFilter(logging.Filter):
    """Filter for security-related log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter security logs."""
        security_keywords = ['auth', 'login', 'permission', 'access', 'security', 'token']
        message = record.getMessage().lower()
        return any(keyword in message for keyword in security_keywords)


def setup_logging():
    """Set up comprehensive logging configuration."""
    
    # Create logs directory
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(funcName)s(): %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": JSONFormatter
            }
        },
        "filters": {
            "performance_filter": {
                "()": PerformanceFilter
            },
            "error_filter": {
                "()": ErrorFilter
            },
            "security_filter": {
                "()": SecurityFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file_all": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": log_dir / "voice_replication.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_json": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": log_dir / "voice_replication.json",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "file_errors": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": log_dir / "errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "encoding": "utf8",
                "filters": ["error_filter"]
            },
            "file_performance": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": log_dir / "performance.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
                "filters": ["performance_filter"]
            },
            "file_security": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": log_dir / "security.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10,
                "encoding": "utf8",
                "filters": ["security_filter"]
            }
        },
        "loggers": {
            "app": {
                "level": "DEBUG",
                "handlers": ["console", "file_all", "file_json"],
                "propagate": False
            },
            "app.api": {
                "level": "INFO",
                "handlers": ["file_all", "file_json"],
                "propagate": True
            },
            "app.services": {
                "level": "DEBUG",
                "handlers": ["file_all", "file_json"],
                "propagate": True
            },
            "app.tasks": {
                "level": "INFO",
                "handlers": ["file_all", "file_json"],
                "propagate": True
            },
            "app.performance": {
                "level": "INFO",
                "handlers": ["file_performance"],
                "propagate": False
            },
            "app.security": {
                "level": "INFO",
                "handlers": ["file_security"],
                "propagate": False
            },
            "app.errors": {
                "level": "ERROR",
                "handlers": ["file_errors"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file_all"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file_all"],
                "propagate": False
            },
            "celery": {
                "level": "INFO",
                "handlers": ["file_all", "file_json"],
                "propagate": False
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["file_all"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file_all"]
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set up specific loggers
    setup_specialized_loggers()


def setup_specialized_loggers():
    """Set up specialized loggers for different components."""
    
    # Performance logger
    performance_logger = logging.getLogger("app.performance")
    performance_logger.info("Performance monitoring initialized")
    
    # Security logger
    security_logger = logging.getLogger("app.security")
    security_logger.info("Security monitoring initialized")
    
    # Error logger
    error_logger = logging.getLogger("app.errors")
    error_logger.info("Error monitoring initialized")


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for the class."""
        return logging.getLogger(f"app.{self.__class__.__module__}.{self.__class__.__name__}")


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("app.performance")
    
    def log_request_performance(
        self,
        endpoint: str,
        method: str,
        duration: float,
        status_code: int,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log API request performance."""
        metrics = {
            "type": "api_request",
            "endpoint": endpoint,
            "method": method,
            "duration_ms": round(duration * 1000, 2),
            "status_code": status_code
        }
        
        extra = {"performance_metrics": metrics}
        if request_id:
            extra["request_id"] = request_id
        if user_id:
            extra["user_id"] = user_id
        
        self.logger.info(f"API {method} {endpoint} - {duration*1000:.2f}ms", extra=extra)
    
    def log_task_performance(
        self,
        task_name: str,
        duration: float,
        status: str,
        task_id: Optional[str] = None,
        **kwargs
    ):
        """Log background task performance."""
        metrics = {
            "type": "background_task",
            "task_name": task_name,
            "duration_ms": round(duration * 1000, 2),
            "status": status,
            **kwargs
        }
        
        extra = {"performance_metrics": metrics}
        if task_id:
            extra["task_id"] = task_id
        
        self.logger.info(f"Task {task_name} - {duration*1000:.2f}ms ({status})", extra=extra)
    
    def log_synthesis_performance(
        self,
        text_length: int,
        processing_time: float,
        quality_score: float,
        task_id: str,
        voice_model_id: str
    ):
        """Log synthesis performance metrics."""
        metrics = {
            "type": "synthesis",
            "text_length": text_length,
            "processing_time_ms": round(processing_time * 1000, 2),
            "quality_score": quality_score,
            "voice_model_id": voice_model_id,
            "chars_per_second": round(text_length / processing_time, 2) if processing_time > 0 else 0
        }
        
        extra = {
            "performance_metrics": metrics,
            "task_id": task_id
        }
        
        self.logger.info(
            f"Synthesis completed - {text_length} chars in {processing_time*1000:.2f}ms "
            f"(quality: {quality_score:.2f})",
            extra=extra
        )
    
    def log_voice_analysis_performance(
        self,
        audio_duration: float,
        processing_time: float,
        quality_score: float,
        voice_profile_id: str
    ):
        """Log voice analysis performance metrics."""
        metrics = {
            "type": "voice_analysis",
            "audio_duration_s": audio_duration,
            "processing_time_ms": round(processing_time * 1000, 2),
            "quality_score": quality_score,
            "voice_profile_id": voice_profile_id,
            "processing_ratio": round(processing_time / audio_duration, 2) if audio_duration > 0 else 0
        }
        
        extra = {"performance_metrics": metrics}
        
        self.logger.info(
            f"Voice analysis completed - {audio_duration:.1f}s audio in {processing_time*1000:.2f}ms "
            f"(quality: {quality_score:.2f})",
            extra=extra
        )


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self):
        self.logger = logging.getLogger("app.security")
    
    def log_authentication_attempt(
        self,
        user_id: Optional[str],
        success: bool,
        ip_address: str,
        user_agent: str
    ):
        """Log authentication attempts."""
        self.logger.info(
            f"Authentication {'successful' if success else 'failed'} for user {user_id or 'unknown'} "
            f"from {ip_address}",
            extra={
                "user_id": user_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "auth_success": success
            }
        )
    
    def log_rate_limit_exceeded(
        self,
        ip_address: str,
        endpoint: str,
        limit: int
    ):
        """Log rate limit violations."""
        self.logger.warning(
            f"Rate limit exceeded for {ip_address} on {endpoint} (limit: {limit})",
            extra={
                "ip_address": ip_address,
                "endpoint": endpoint,
                "rate_limit": limit
            }
        )
    
    def log_suspicious_activity(
        self,
        activity_type: str,
        details: Dict[str, Any],
        ip_address: str,
        user_id: Optional[str] = None
    ):
        """Log suspicious activities."""
        self.logger.warning(
            f"Suspicious activity detected: {activity_type}",
            extra={
                "activity_type": activity_type,
                "details": details,
                "ip_address": ip_address,
                "user_id": user_id
            }
        )


# Global logger instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f"app.{name}")


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f"app.{func.__module__}.{func.__name__}")
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


def log_performance(operation_name: str):
    """Decorator to log performance metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_logger.log_task_performance(
                    task_name=f"{func.__module__}.{func.__name__}",
                    duration=duration,
                    status="success"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_logger.log_task_performance(
                    task_name=f"{func.__module__}.{func.__name__}",
                    duration=duration,
                    status="error",
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator