"""
Logging utilities for the Intelligent Feedback Analysis System.
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

from config.settings import LOGGING_CONFIG, LOGS_DIR


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m',      # End color
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['ENDC']}"
            )
        
        # Format timestamp
        record.asctime = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        
        return super().format(record)


def setup_logging():
    """Set up logging configuration for the entire system."""
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get logging level
    log_level = getattr(logging, LOGGING_CONFIG["level"].upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation - THIS CREATES logs/system.log
    log_file = LOGS_DIR / "system.log"
    
    # Ensure the log file and directory exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        fmt=LOGGING_CONFIG["format"],
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Create initial log entry to ensure file is created
    initial_logger = logging.getLogger("system_startup")
    initial_logger.info("=" * 60)
    initial_logger.info("INTELLIGENT FEEDBACK ANALYSIS SYSTEM - STARTUP")
    initial_logger.info("=" * 60)
    initial_logger.info(f"Log file created: {log_file}")
    initial_logger.info(f"Log level: {LOGGING_CONFIG['level']}")
    initial_logger.info(f"System startup time: {datetime.now().isoformat()}")
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually module name)
        level: Optional logging level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Ensure the logger writes to system.log
    if not logger.handlers:
        # If no handlers, ensure we get the root logger configuration
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            setup_logging()
    
    return logger


def create_system_log():
    """
    Explicitly create the system.log file with initial content.
    This ensures logs/system.log exists even before any logging occurs.
    """
    log_file = LOGS_DIR / "system.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the file if it doesn't exist
    if not log_file.exists():
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Intelligent Feedback Analysis System - Log File\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write(f"# Log Level: {LOGGING_CONFIG['level']}\n")
            f.write(f"# Format: {LOGGING_CONFIG['format']}\n")
            f.write(f"#\n")
            f.write(f"# This file contains system logs for the feedback analysis pipeline.\n")
            f.write(f"# Logs are automatically rotated when they exceed 10MB.\n")
            f.write(f"#\n\n")
    
    return log_file


class StructuredLogger:
    """Logger that outputs structured data for better analysis."""
    
    def __init__(self, name: str):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self.structured_log_file = LOGS_DIR / "structured.jsonl"
        
        # Ensure structured log file exists
        self.structured_log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.structured_log_file.exists():
            self.structured_log_file.touch()
    
    def log_structured(self, event_type: str, data: dict, level: str = "INFO"):
        """
        Log structured data in JSON Lines format.
        
        Args:
            event_type: Type of event (e.g., "processing", "error", "metric")
            data: Data to log
            level: Logging level
        """
        import json
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "level": level,
            "data": data
        }
        
        # Log to regular logger (this goes to system.log)
        getattr(self.logger, level.lower())(
            f"{event_type}: {json.dumps(data, default=str)}"
        )
        
        # Append to structured log file
        try:
            with open(self.structured_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, default=str) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write structured log: {e}")
    
    def log_processing_event(
        self, 
        agent_name: str, 
        source_id: str, 
        action: str, 
        success: bool,
        duration: float,
        details: dict = None
    ):
        """Log a processing event."""
        data = {
            "agent_name": agent_name,
            "source_id": source_id,
            "action": action,
            "success": success,
            "duration_ms": duration * 1000,
            "details": details or {}
        }
        
        level = "INFO" if success else "ERROR"
        self.log_structured("processing", data, level)
    
    def log_metric(self, metric_name: str, value: float, tags: dict = None):
        """Log a metric value."""
        data = {
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {}
        }
        
        self.log_structured("metric", data, "INFO")
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log an error event."""
        data = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self.log_structured("error", data, "ERROR")


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, name: str):
        """
        Initialize performance logger.
        
        Args:
            name: Logger name
        """
        self.logger = get_logger(f"perf.{name}")
        self.start_time = None
        self.performance_log_file = LOGS_DIR / "performance.log"
        
        # Ensure performance log file exists
        self.performance_log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.performance_log_file.exists():
            with open(self.performance_log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Performance Log - {datetime.now().isoformat()}\n")
                f.write(f"# Format: timestamp,operation,duration_ms,details\n\n")
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_time = datetime.utcnow()
        self.logger.debug(f"Started: {operation}")
    
    def end_timer(self, operation: str, details: dict = None):
        """End timing and log duration."""
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds()
            
            log_msg = f"Completed: {operation} in {duration:.3f}s"
            if details:
                log_msg += f" - {details}"
            
            self.logger.info(log_msg)
            
            # Also log to performance file
            try:
                with open(self.performance_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.utcnow().isoformat()},{operation},{duration*1000:.1f},{details or ''}\n")
            except Exception as e:
                self.logger.error(f"Failed to write performance log: {e}")
            
            return duration
        else:
            self.logger.warning(f"No start time recorded for: {operation}")
            return None
    
    def log_performance_metric(
        self, 
        metric_name: str, 
        value: float, 
        unit: str = "ms",
        threshold: float = None
    ):
        """Log a performance metric."""
        log_msg = f"{metric_name}: {value:.3f}{unit}"
        
        if threshold and value > threshold:
            self.logger.warning(f"SLOW - {log_msg} (threshold: {threshold}{unit})")
        else:
            self.logger.info(log_msg)


def log_function_call(func):
    """Decorator to log function calls with execution time."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.utcnow()
        
        try:
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Error in {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls with execution time."""
    import asyncio
    
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.utcnow()
        
        try:
            logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
            result = await func(*args, **kwargs)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"Completed async {func.__name__} in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Error in async {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return wrapper


# Initialize logging when module is imported AND create system.log
def initialize_logging_system():
    """Initialize the complete logging system and create all log files."""
    
    # Create the logs directory
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create system.log file
    system_log_file = create_system_log()
    
    # Set up logging configuration
    setup_logging()
    
    # Create additional log files
    additional_logs = [
        "structured.jsonl",
        "performance.log", 
        "errors.log",
        "agents.log"
    ]
    
    for log_name in additional_logs:
        log_file = LOGS_DIR / log_name
        if not log_file.exists():
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"# {log_name} - Created {datetime.now().isoformat()}\n")
    
    # Log initialization success
    init_logger = get_logger("logging_system")
    init_logger.info("Logging system initialized successfully")
    init_logger.info(f"Main log file: {system_log_file}")
    init_logger.info(f"Additional log files created in: {LOGS_DIR}")
    
    return system_log_file


# Automatically initialize when module is imported
initialize_logging_system()