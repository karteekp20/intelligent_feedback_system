"""
Configuration settings for the Intelligent Feedback Analysis System.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Agent Configuration
AGENT_SETTINGS = {
    "csv_reader": {
        "max_file_size_mb": 50,
        "encoding": "utf-8",
        "chunk_size": 1000
    },
    "classifier": {
        "confidence_threshold": 0.7,
        "categories": ["Bug", "Feature Request", "Praise", "Complaint", "Spam"],
        "model_temperature": 0.1
    },
    "bug_analyzer": {
        "severity_levels": ["Critical", "High", "Medium", "Low"],
        "platforms": ["iOS", "Android", "Web", "Desktop"],
        "confidence_threshold": 0.8
    },
    "feature_extractor": {
        "impact_levels": ["High", "Medium", "Low"],
        "complexity_levels": ["Simple", "Moderate", "Complex"],
        "confidence_threshold": 0.6
    },
    "ticket_creator": {
        "priority_mapping": {
            "Critical": "P0",
            "High": "P1",
            "Medium": "P2",
            "Low": "P3"
        },
        "template_format": "structured"
    },
    "quality_critic": {
        "review_threshold": 0.85,
        "required_fields": ["title", "description", "category", "priority"],
        "quality_metrics": ["completeness", "clarity", "actionability"]
    }
}

# Processing Configuration
PROCESSING_SETTINGS = {
    "max_concurrent_agents": 5,
    "batch_size": 10,
    "timeout_seconds": 300,
    "retry_attempts": 3,
    "retry_delay": 1.0
}

# Classification Thresholds
CLASSIFICATION_THRESHOLDS = {
    "bug_detection": 0.75,
    "feature_request": 0.70,
    "spam_detection": 0.85,
    "sentiment_analysis": 0.60
}

# Priority Assignment Rules
PRIORITY_RULES = {
    "critical_keywords": ["crash", "data loss", "security", "payment", "login failed"],
    "high_keywords": ["bug", "error", "broken", "not working", "issue"],
    "feature_keywords": ["request", "suggest", "would like", "please add", "enhancement"],
    "low_priority_keywords": ["cosmetic", "minor", "nice to have", "suggestion"]
}

# UI Configuration
STREAMLIT_CONFIG = {
    "port": 8501,
    "host": "localhost",
    "theme": "light",
    "auto_refresh_interval": 30,
    "max_display_items": 100
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": LOGS_DIR / "system.log",
    "max_file_size": "10MB",
    "backup_count": 5
}

# File Paths
FILE_PATHS = {
    "app_store_reviews": INPUT_DIR / "app_store_reviews.csv",
    "support_emails": INPUT_DIR / "support_emails.csv",
    "expected_classifications": INPUT_DIR / "expected_classifications.csv",
    "generated_tickets": OUTPUT_DIR / "generated_tickets.csv",
    "processing_log": OUTPUT_DIR / "processing_log.csv",
    "metrics": OUTPUT_DIR / "metrics.csv"
}

# CSV Schemas
CSV_SCHEMAS = {
    "app_store_reviews": [
        "review_id", "platform", "rating", "review_text", 
        "user_name", "date", "app_version"
    ],
    "support_emails": [
        "email_id", "subject", "body", "sender_email", 
        "timestamp", "priority"
    ],
    "expected_classifications": [
        "source_id", "source_type", "category", "priority", 
        "technical_details", "suggested_title"
    ],
    "generated_tickets": [
        "ticket_id", "source_id", "source_type", "category", 
        "priority", "title", "description", "technical_details", 
        "created_at", "agent_confidence"
    ],
    "processing_log": [
        "timestamp", "source_id", "agent_name", "action", 
        "details", "confidence_score"
    ],
    "metrics": [
        "metric_name", "value", "timestamp", "details"
    ]
}

# Validation Rules
VALIDATION_RULES = {
    "review_id": {"type": "str", "required": True, "unique": True},
    "rating": {"type": "int", "min": 1, "max": 5},
    "email_id": {"type": "str", "required": True, "unique": True},
    "priority": {"type": "str", "values": ["Critical", "High", "Medium", "Low"]},
    "category": {"type": "str", "values": ["Bug", "Feature Request", "Praise", "Complaint", "Spam"]}
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    "enable_monitoring": True,
    "metrics_collection_interval": 60,  # seconds
    "performance_thresholds": {
        "processing_time_per_item": 5.0,  # seconds
        "classification_accuracy": 0.85,
        "memory_usage_mb": 1000
    }
}

# Error Handling
ERROR_HANDLING = {
    "max_retries": 3,
    "retry_delay": 2.0,
    "fallback_strategies": {
        "classification_failure": "assign_manual_review",
        "api_failure": "use_local_model",
        "file_read_error": "skip_and_log"
    }
}

# Development Settings
DEBUG_SETTINGS = {
    "enable_debug": os.getenv("DEBUG", "False").lower() == "true",
    "verbose_logging": os.getenv("VERBOSE", "False").lower() == "true",
    "save_intermediate_results": True,
    "profile_performance": False
}

def get_setting(key: str, default=None):
    """Get a configuration setting with optional default."""
    settings_map = {
        "agent": AGENT_SETTINGS,
        "processing": PROCESSING_SETTINGS,
        "classification": CLASSIFICATION_THRESHOLDS,
        "priority": PRIORITY_RULES,
        "ui": STREAMLIT_CONFIG,
        "logging": LOGGING_CONFIG,
        "paths": FILE_PATHS,
        "schemas": CSV_SCHEMAS,
        "validation": VALIDATION_RULES,
        "performance": PERFORMANCE_CONFIG,
        "error": ERROR_HANDLING,
        "debug": DEBUG_SETTINGS
    }
    
    for category, settings in settings_map.items():
        if key in settings:
            return settings[key]
    
    return default

def validate_configuration():
    """Validate the configuration settings."""
    errors = []
    
    # Check required API key
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")
    
    # Check directory existence
    required_dirs = [DATA_DIR, INPUT_DIR, OUTPUT_DIR, LOGS_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            errors.append(f"Directory does not exist: {dir_path}")
    
    # Check threshold values
    for threshold in CLASSIFICATION_THRESHOLDS.values():
        if not (0.0 <= threshold <= 1.0):
            errors.append("Classification thresholds must be between 0.0 and 1.0")
            break
    
    # Check processing settings
    if PROCESSING_SETTINGS["batch_size"] <= 0:
        errors.append("Batch size must be positive")
    
    if PROCESSING_SETTINGS["timeout_seconds"] <= 0:
        errors.append("Timeout must be positive")
    
    return errors

# Validate configuration on import
config_errors = validate_configuration()
if config_errors:
    print("Configuration errors found:")
    for error in config_errors:
        print(f"  - {error}")
    print("Please fix these issues before running the system.")