"""
Logging Utilities for Multi-Modal Deepfake Detection
Comprehensive logging configuration with multiple handlers, formatters, and security features
Supports structured logging, performance monitoring, and audit trails
"""

import logging
import logging.handlers
import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings
warnings.filterwarnings("ignore")

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging with JSON output
    Includes contextual information and performance metrics
    """

    def __init__(self, include_extra: bool = True):
        """
        Initialize structured formatter

        Args:
            include_extra: Include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra

        # Standard fields to include in every log record
        self.standard_fields = {
            'timestamp', 'level', 'logger', 'message', 'module', 
            'function', 'line', 'thread', 'process'
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""

        # Create base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }

        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in self.standard_fields and not key.startswith('_'):
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry)


class PerformanceLogger:
    """
    Performance monitoring logger for tracking execution times and resource usage
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger

        Args:
            logger: Logger instance to use
        """
        self.logger = logger
        self.start_times = {}

    def start_timer(self, operation: str, **context) -> str:
        """
        Start timing an operation

        Args:
            operation: Name of operation being timed
            **context: Additional context information

        Returns:
            Timer ID for stopping the timer
        """
        timer_id = f"{operation}_{time.time()}"
        self.start_times[timer_id] = {
            'start_time': time.time(),
            'operation': operation,
            'context': context
        }

        self.logger.debug(
            f"Started operation: {operation}",
            extra={
                'timer_id': timer_id,
                'operation': operation,
                'event_type': 'performance_start',
                **context
            }
        )

        return timer_id

    def end_timer(self, timer_id: str, **additional_context) -> float:
        """
        End timing an operation

        Args:
            timer_id: Timer ID from start_timer
            **additional_context: Additional context information

        Returns:
            Elapsed time in seconds
        """
        if timer_id not in self.start_times:
            self.logger.warning(f"Timer ID not found: {timer_id}")
            return 0.0

        timer_info = self.start_times.pop(timer_id)
        end_time = time.time()
        elapsed_time = end_time - timer_info['start_time']

        # Determine log level based on elapsed time
        if elapsed_time > 10.0:  # Very slow
            log_level = logging.WARNING
        elif elapsed_time > 1.0:  # Slow
            log_level = logging.INFO
        else:  # Normal
            log_level = logging.DEBUG

        self.logger.log(
            log_level,
            f"Completed operation: {timer_info['operation']} in {elapsed_time:.3f}s",
            extra={
                'timer_id': timer_id,
                'operation': timer_info['operation'],
                'elapsed_time': elapsed_time,
                'event_type': 'performance_end',
                **timer_info['context'],
                **additional_context
            }
        )

        return elapsed_time

    def log_metrics(self, metrics: Dict[str, Any], operation: str = None):
        """
        Log performance metrics

        Args:
            metrics: Dictionary of metrics to log
            operation: Operation associated with metrics
        """
        self.logger.info(
            f"Performance metrics{'for ' + operation if operation else ''}",
            extra={
                'event_type': 'metrics',
                'operation': operation,
                'metrics': metrics
            }
        )


class SecurityLogger:
    """
    Security-focused logger for audit trails and security events
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize security logger

        Args:
            logger: Logger instance to use
        """
        self.logger = logger

    def log_access(self, user_id: str, resource: str, action: str, 
                   success: bool = True, ip_address: str = None, **context):
        """
        Log access attempts

        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            success: Whether access was successful
            ip_address: IP address of request
            **context: Additional context information
        """
        level = logging.INFO if success else logging.WARNING

        self.logger.log(
            level,
            f"Access {'granted' if success else 'denied'}: {user_id} -> {action} on {resource}",
            extra={
                'event_type': 'access',
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'success': success,
                'ip_address': ip_address,
                **context
            }
        )

    def log_security_event(self, event_type: str, description: str, 
                          severity: str = 'info', **context):
        """
        Log security events

        Args:
            event_type: Type of security event
            description: Description of event
            severity: Event severity ('debug', 'info', 'warning', 'error', 'critical')
            **context: Additional context information
        """
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }

        level = level_map.get(severity.lower(), logging.INFO)

        self.logger.log(
            level,
            f"Security event: {event_type} - {description}",
            extra={
                'event_type': 'security',
                'security_event_type': event_type,
                'severity': severity,
                'description': description,
                **context
            }
        )

    def log_data_access(self, data_type: str, operation: str, 
                       record_count: int = None, **context):
        """
        Log data access operations

        Args:
            data_type: Type of data accessed
            operation: Operation performed
            record_count: Number of records affected
            **context: Additional context information
        """
        self.logger.info(
            f"Data access: {operation} on {data_type}" + 
            (f" ({record_count} records)" if record_count else ""),
            extra={
                'event_type': 'data_access',
                'data_type': data_type,
                'operation': operation,
                'record_count': record_count,
                **context
            }
        )


class ContextualLogger:
    """
    Logger with contextual information for request tracing
    """

    def __init__(self, logger: logging.Logger):
        """
        Initialize contextual logger

        Args:
            logger: Logger instance to use
        """
        self.logger = logger
        self.context_stack = []

    def push_context(self, **context):
        """
        Push context onto the stack

        Args:
            **context: Context information to add
        """
        self.context_stack.append(context)

    def pop_context(self):
        """Remove the most recent context from the stack"""
        if self.context_stack:
            self.context_stack.pop()

    def get_current_context(self) -> Dict[str, Any]:
        """Get merged current context"""
        merged_context = {}
        for context in self.context_stack:
            merged_context.update(context)
        return merged_context

    def debug(self, message: str, **extra):
        """Log debug message with context"""
        self._log(logging.DEBUG, message, extra)

    def info(self, message: str, **extra):
        """Log info message with context"""
        self._log(logging.INFO, message, extra)

    def warning(self, message: str, **extra):
        """Log warning message with context"""
        self._log(logging.WARNING, message, extra)

    def error(self, message: str, **extra):
        """Log error message with context"""
        self._log(logging.ERROR, message, extra)

    def critical(self, message: str, **extra):
        """Log critical message with context"""
        self._log(logging.CRITICAL, message, extra)

    def _log(self, level: int, message: str, extra: Dict[str, Any]):
        """Log message with merged context"""
        merged_extra = self.get_current_context()
        merged_extra.update(extra)

        self.logger.log(level, message, extra=merged_extra)


def setup_logger(name: str = 'deepfake_detection',
                log_level: str = 'INFO',
                log_dir: str = 'logs',
                log_to_file: bool = True,
                log_to_console: bool = True,
                structured_logging: bool = True,
                max_file_size: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5,
                enable_performance_logging: bool = True,
                enable_security_logging: bool = True) -> logging.Logger:
    """
    Set up comprehensive logging configuration

    Args:
        name: Logger name
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_dir: Directory for log files
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        structured_logging: Use structured JSON logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        enable_performance_logging: Enable performance logging
        enable_security_logging: Enable security logging

    Returns:
        Configured logger instance
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create log directory
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)

        if structured_logging:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handlers
    if log_to_file:
        # Main log file (rotating)
        main_file_handler = logging.handlers.RotatingFileHandler(
            Path(log_dir) / f'{name}.log',
            maxBytes=max_file_size,
            backupCount=backup_count
        )

        if structured_logging:
            main_formatter = StructuredFormatter()
        else:
            main_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )

        main_file_handler.setFormatter(main_formatter)
        logger.addHandler(main_file_handler)

        # Error log file (errors and above only)
        error_file_handler = logging.handlers.RotatingFileHandler(
            Path(log_dir) / f'{name}_errors.log',
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(main_formatter)
        logger.addHandler(error_file_handler)

        # Performance log file (if enabled)
        if enable_performance_logging:
            perf_filter = lambda record: hasattr(record, 'event_type') and record.event_type.startswith('performance')

            perf_file_handler = logging.handlers.RotatingFileHandler(
                Path(log_dir) / f'{name}_performance.log',
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            perf_file_handler.addFilter(perf_filter)
            perf_file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(perf_file_handler)

        # Security log file (if enabled)
        if enable_security_logging:
            security_filter = lambda record: hasattr(record, 'event_type') and record.event_type in ['security', 'access', 'data_access']

            security_file_handler = logging.handlers.RotatingFileHandler(
                Path(log_dir) / f'{name}_security.log',
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            security_file_handler.addFilter(security_filter)
            security_file_handler.setFormatter(StructuredFormatter())
            logger.addHandler(security_file_handler)

    # Add custom attributes to logger
    logger.performance = PerformanceLogger(logger) if enable_performance_logging else None
    logger.security = SecurityLogger(logger) if enable_security_logging else None
    logger.contextual = ContextualLogger(logger)

    logger.info(f"Logger '{name}' initialized with level {log_level}")

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get existing logger or create a basic one

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Logger instance
    """
    if name is None:
        # Get name from calling module
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')

    logger = logging.getLogger(name)

    # If logger has no handlers, set up basic logging
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


class LoggerContextManager:
    """
    Context manager for automatic context management and timing
    """

    def __init__(self, logger: logging.Logger, operation: str, 
                 level: int = logging.INFO, **context):
        """
        Initialize context manager

        Args:
            logger: Logger instance
            operation: Operation name
            level: Log level
            **context: Context information
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.context = context
        self.timer_id = None
        self.start_time = None

    def __enter__(self):
        """Enter context"""
        self.start_time = time.time()

        # Start performance timer if available
        if hasattr(self.logger, 'performance') and self.logger.performance:
            self.timer_id = self.logger.performance.start_timer(self.operation, **self.context)

        # Push context if available
        if hasattr(self.logger, 'contextual') and self.logger.contextual:
            self.logger.contextual.push_context(operation=self.operation, **self.context)

        self.logger.log(
            self.level,
            f"Starting operation: {self.operation}",
            extra={'event_type': 'operation_start', 'operation': self.operation, **self.context}
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        # End performance timer if available
        if self.timer_id and hasattr(self.logger, 'performance') and self.logger.performance:
            self.logger.performance.end_timer(self.timer_id)

        # Pop context if available
        if hasattr(self.logger, 'contextual') and self.logger.contextual:
            self.logger.contextual.pop_context()

        if exc_type is not None:
            # Operation failed
            self.logger.error(
                f"Operation failed: {self.operation} after {elapsed_time:.3f}s",
                extra={
                    'event_type': 'operation_error',
                    'operation': self.operation,
                    'elapsed_time': elapsed_time,
                    'exception_type': exc_type.__name__,
                    'exception_message': str(exc_val),
                    **self.context
                },
                exc_info=True
            )
        else:
            # Operation succeeded
            self.logger.log(
                self.level,
                f"Completed operation: {self.operation} in {elapsed_time:.3f}s",
                extra={
                    'event_type': 'operation_success',
                    'operation': self.operation,
                    'elapsed_time': elapsed_time,
                    **self.context
                }
            )


def configure_logging_for_production():
    """Configure logging for production environment"""

    # Set up main application logger
    app_logger = setup_logger(
        name='deepfake_detection',
        log_level='INFO',
        log_dir='/var/log/deepfake_detection',
        log_to_file=True,
        log_to_console=False,  # Disable console in production
        structured_logging=True,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10,
        enable_performance_logging=True,
        enable_security_logging=True
    )

    # Configure third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Set up error monitoring (example with custom handler)
    class ErrorAlertHandler(logging.Handler):
        """Custom handler for critical errors"""

        def emit(self, record):
            if record.levelno >= logging.ERROR:
                # In production, this could send alerts to monitoring systems
                print(f"ALERT: {record.getMessage()}", file=sys.stderr)

    error_handler = ErrorAlertHandler()
    error_handler.setLevel(logging.ERROR)
    app_logger.addHandler(error_handler)

    return app_logger


def configure_logging_for_development():
    """Configure logging for development environment"""

    return setup_logger(
        name='deepfake_detection',
        log_level='DEBUG',
        log_dir='logs',
        log_to_file=True,
        log_to_console=True,
        structured_logging=False,  # Human-readable for development
        max_file_size=10 * 1024 * 1024,  # 10MB
        backup_count=3,
        enable_performance_logging=True,
        enable_security_logging=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Test logger setup
    test_logger = setup_logger(
        name='test_logger',
        log_level='DEBUG',
        log_dir='test_logs',
        structured_logging=True
    )

    # Test basic logging
    test_logger.info("Test message", extra={'test_field': 'test_value'})

    # Test performance logging
    if test_logger.performance:
        timer_id = test_logger.performance.start_timer("test_operation", param1="value1")
        time.sleep(0.1)  # Simulate work
        test_logger.performance.end_timer(timer_id, result="success")

    # Test security logging
    if test_logger.security:
        test_logger.security.log_access("user123", "model", "predict", True, "192.168.1.1")

    # Test context manager
    with LoggerContextManager(test_logger, "test_context_operation", test_param="test"):
        test_logger.info("Inside context operation")

    print("Logger testing completed. Check 'test_logs' directory for output files.")
