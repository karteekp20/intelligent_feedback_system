#!/usr/bin/env python3
"""
Script to initialize the logging system and create all log files.
This ensures logs/system.log and other log files are created before running the main system.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import LOGS_DIR
from src.utils.logger import initialize_logging_system, get_logger


def main():
    """Initialize logging system and create all log files."""
    
    print("Initializing Intelligent Feedback Analysis System Logging...")
    print(f"Logs directory: {LOGS_DIR}")
    
    # Initialize the complete logging system
    system_log_file = initialize_logging_system()
    
    # Test all loggers to ensure they work
    test_loggers()
    
    print(f"Logging system initialized successfully!")
    print(f"Main log file: {system_log_file}")
    print(f"Additional logs created in: {LOGS_DIR}")
    
    # List all created log files
    log_files = list(LOGS_DIR.glob("*.log")) + list(LOGS_DIR.glob("*.jsonl"))
    print(f"\n Created log files:")
    for log_file in sorted(log_files):
        size = log_file.stat().st_size if log_file.exists() else 0
        print(f"  {log_file.name} ({size} bytes)")


def test_loggers():
    """Test all logger types to ensure they write to files correctly."""
    
    # Test main system logger
    main_logger = get_logger("system_test")
    main_logger.info("System logger test - this should appear in logs/system.log")
    main_logger.warning("Warning test message")
    main_logger.error("Error test message")
    
    # Test structured logger
    from src.utils.logger import StructuredLogger
    struct_logger = StructuredLogger("test_structured")
    struct_logger.log_structured("test_event", {
        "test_key": "test_value",
        "timestamp": datetime.now().isoformat(),
        "system": "feedback_analysis"
    })
    
    # Test performance logger
    from src.utils.logger import PerformanceLogger
    perf_logger = PerformanceLogger("test_performance")
    perf_logger.start_timer("test_operation")
    import time
    time.sleep(0.01)  # Brief pause
    perf_logger.end_timer("test_operation", {"test": True})
    
    # Test agent-specific logger
    agent_logger = get_logger("test_agent")
    agent_logger.info("Agent logger test - CSV Reader Agent simulation")
    agent_logger.info("Agent logger test - Classifier Agent simulation")
    
    print("✅ All logger types tested successfully")


def create_sample_logs():
    """Create sample log entries to demonstrate the system."""
    
    logger = get_logger("sample_system")
    
    # Simulate system startup
    logger.info("=" * 60)
    logger.info("SAMPLE SYSTEM OPERATION LOG")
    logger.info("=" * 60)
    
    # Simulate agent operations
    agents = ["csv_reader", "classifier", "bug_analyzer", "feature_extractor", "ticket_creator", "quality_critic"]
    
    for agent in agents:
        agent_logger = get_logger(f"agent.{agent}")
        agent_logger.info(f"Agent {agent} initialized successfully")
        agent_logger.info(f"Agent {agent} processing sample data...")
        agent_logger.info(f"Agent {agent} completed processing")
    
    # Simulate pipeline completion
    logger.info("Sample pipeline execution completed")
    logger.info("All log files created and tested")


if __name__ == "__main__":
    main()
    
    # Also create sample logs for demonstration
    create_sample_logs()
    
    print("\n🎉 Logging system is ready!")
    print("💡 You can now run the main system and all logs will be captured in logs/system.log")