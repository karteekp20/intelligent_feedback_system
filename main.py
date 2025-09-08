#!/usr/bin/env python3
"""
Enhanced Main Application Entry Point
Intelligent User Feedback Analysis and Action System

This module provides the main entry point for the feedback analysis system
with enhanced error handling, monitoring, real-time processing capabilities, and dashboard support.
"""

import asyncio
import argparse
import sys
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.core.pipeline import FeedbackPipeline
from src.core.data_models import PipelineResult, SystemStats
from src.utils.logger import get_logger, PerformanceLogger
from config.settings import (
    INPUT_DIR, OUTPUT_DIR, PROCESSING_SETTINGS, 
)

logger = get_logger(__name__)
performance_logger = PerformanceLogger('main_application')

class FeedbackSystemApp:
    """Enhanced main application class with monitoring, real-time capabilities, and dashboard support."""
    
    def __init__(self):
        self.pipeline = None
        self.is_running = False
        self.stats = SystemStats()
        self.shutdown_event = asyncio.Event()
        self.dashboard_process = None
        
    async def initialize(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize the feedback processing system."""
        try:
            logger.info("🚀 Initializing Intelligent Feedback Analysis System...")
            
            # Validate environment and files
            await self._validate_environment()
            
            # Initialize pipeline with enhanced configuration
            pipeline_config = PROCESSING_SETTINGS.copy()
            if config_override:
                pipeline_config.update(config_override)
                
            self.pipeline = FeedbackPipeline(pipeline_config)
            # Pipeline initialized in constructor
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            logger.info("✅ System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def _validate_environment(self):
        """Validate system environment and requirements."""
        logger.info("🔍 Validating system environment...")
        
        # Check directories
        for directory in [INPUT_DIR, OUTPUT_DIR]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Created directory: {directory}")
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        if not logs_dir.exists():
            logs_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created directory: {logs_dir}")
        
        # Validate input files
        input_files = list(INPUT_DIR.glob("*.csv"))
        if not input_files:
            logger.warning("⚠️ No CSV files found in input directory")
            
        # Create sample input files if none exist
        await self._create_sample_files_if_needed()
    
    async def _create_sample_files_if_needed(self):
        """Create sample input files if directory is empty."""
        input_files = list(INPUT_DIR.glob("*.csv"))
        if not input_files:
            logger.info("📄 Creating sample input files...")
            
            # Create sample app store reviews
            sample_reviews = INPUT_DIR / "app_store_reviews.csv"
            sample_reviews_content = """review_id,user_id,rating,review_text,date,app_version,device_type,country
1,user123,5,"Great app! Love the new features",2024-01-15,2.1.0,iPhone,US
2,user456,2,"App crashes when I try to save my work",2024-01-16,2.1.0,Android,UK
3,user789,4,"Good app but could use dark mode",2024-01-17,2.1.0,iPad,CA
4,user101,1,"Terrible update, nothing works anymore",2024-01-18,2.1.0,iPhone,AU
5,user202,5,"Perfect! Exactly what I needed",2024-01-19,2.1.0,Android,US
"""
            
            # Create sample support emails
            sample_emails = INPUT_DIR / "support_emails.csv"
            sample_emails_content = """email_id,subject,body,sender_email,received_date,priority,category,status
1,"App Crash Issue","The app keeps crashing when I try to export data. This is very frustrating.",user1@email.com,2024-01-15,high,bug,open
2,"Feature Request","Please add the ability to sync with Google Drive. This would be very helpful.",user2@email.com,2024-01-16,medium,feature,open
3,"Login Problems","I can't login to my account. The password reset doesn't work.",user3@email.com,2024-01-17,high,bug,open
4,"Thank you","Just wanted to say thank you for the great customer service!",user4@email.com,2024-01-18,low,praise,closed
5,"Performance Issue","The app is very slow when loading large files. Can this be improved?",user5@email.com,2024-01-19,medium,performance,open
"""
            
            try:
                with open(sample_reviews, 'w') as f:
                    f.write(sample_reviews_content)
                logger.info(f"✅ Created sample file: {sample_reviews}")
                
                with open(sample_emails, 'w') as f:
                    f.write(sample_emails_content)
                logger.info(f"✅ Created sample file: {sample_emails}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create sample files: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def process_batch(self, input_files: Optional[list] = None) -> PipelineResult:
        """Process a batch of feedback files."""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            logger.info("🔄 Starting batch processing...")
            start_time = datetime.now()
            
            # Get input files if not provided
            if input_files is None:
                input_files = list(INPUT_DIR.glob("*.csv"))
            
            if not input_files:
                logger.warning("⚠️ No input files found for processing")
                return PipelineResult(
                    id="batch_empty",
                    input_feedback_count=0,
                    processed_feedback_count=0,
                    generated_tickets_count=0,
                    success_rate=0.0,
                    total_processing_time=0.0
                )
            
            # Process files using pipeline
            logger.info(f"📄 Processing {len(input_files)} files...")
            
            # Processing without performance measurement
            pipeline_input = {
                    "file_paths": [str(f) for f in input_files],
                    "file_type": "all"
                }
                
            result = await self.pipeline.run()
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.stats.total_feedback_processed += result.get('input_feedback_count', 0)
            self.stats.total_tickets_generated += result.get('generated_tickets_count', 0)
            self.stats.last_updated = datetime.now()
            
            if result.get('success_rate', 0.0) > 0:
                self.stats.success_rate = (
                    self.stats.success_rate + result.get('success_rate', 0.0)
                ) / 2  # Running average
            
            # Generate summary report
            summary = self._generate_batch_summary(result, processing_time)
            logger.info(f"📊 Batch processing completed: {summary}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return PipelineResult(
                id="batch_failed",
                input_feedback_count=0,
                processed_feedback_count=0,
                generated_tickets_count=0,
                success_rate=0.0,
                total_processing_time=0.0,
                failed_items=[str(e)]
            )
    
    async def start_continuous_monitoring(self, check_interval: int = 60):
        """Start continuous monitoring for new files."""
        logger.info(f"👀 Starting continuous monitoring (check interval: {check_interval}s)")
        self.is_running = True
        
        processed_files = set()
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Check for new files
                current_files = set(INPUT_DIR.glob("*.csv"))
                new_files = current_files - processed_files
                
                if new_files:
                    logger.info(f"🆕 Found {len(new_files)} new files to process")
                    result = await self.process_batch(list(new_files))
                    
                    if result.get('success_rate', 0.0) > 0.5:  # Consider successful if > 50% success rate
                        processed_files.update(new_files)
                        logger.info("✅ New files processed successfully")
                    else:
                        logger.error("❌ Failed to process new files")
                
                # Wait for next check or shutdown signal
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(), 
                        timeout=check_interval
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue monitoring
                    
            except Exception as e:
                logger.error(f"❌ Error in continuous monitoring: {e}")
                await asyncio.sleep(check_interval)
        
        logger.info("🛑 Continuous monitoring stopped")
    
    def start_dashboard(self, port: int = 8501, host: str = "localhost"):
        """Start the Streamlit dashboard in a separate process."""
        try:
            logger.info(f"🌐 Starting dashboard on http://{host}:{port}")
            
            # Command to run dashboard
            dashboard_cmd = [
                sys.executable, "-m", "streamlit", "run", 
                "src/ui/dashboard.py",
                "--server.port", str(port),
                "--server.address", host,
                "--server.headless", "true"
            ]
            
            # Start dashboard process
            self.dashboard_process = subprocess.Popen(
                dashboard_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(__file__).parent)
            )
            
            # Monitor dashboard process in background
            def monitor_dashboard():
                try:
                    stdout, stderr = self.dashboard_process.communicate()
                    if self.dashboard_process.returncode != 0:
                        logger.error(f"❌ Dashboard process failed: {stderr.decode()}")
                    else:
                        logger.info("🌐 Dashboard process ended normally")
                except Exception as e:
                    logger.error(f"❌ Error monitoring dashboard: {e}")
            
            dashboard_thread = threading.Thread(target=monitor_dashboard, daemon=True)
            dashboard_thread.start()
            
            logger.info("✅ Dashboard started successfully")
            logger.info(f"🔗 Access dashboard at: http://{host}:{port}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def _generate_batch_summary(self, result: PipelineResult, processing_time: float) -> str:
        """Generate a summary of batch processing results."""
        return (
            f"Processed {result.get('input_feedback_count', 0)} feedback items, "
            f"generated {result.get('generated_tickets_count', 0)} tickets in {processing_time:.2f}s "
            f"(Success rate: {result.get('success_rate', 0.0):.1%})"
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        return {
            "system_stats": {
                "total_feedback_processed": self.stats.total_feedback_processed,
                "total_tickets_generated": self.stats.total_tickets_generated,
                "success_rate": self.stats.success_rate,
                "last_updated": self.stats.last_updated.isoformat() if self.stats.last_updated else None
            },
            "pipeline_status": "running" if self.is_running else "stopped",
            "dashboard_status": "running" if self.dashboard_process and self.dashboard_process.poll() is None else "stopped",
            "performance_metrics": performance_logger.get_summary(),
            "input_files_count": len(list(INPUT_DIR.glob("*.csv"))),
            "output_files_count": len(list(OUTPUT_DIR.glob("*.csv"))),
            "last_updated": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("🔄 Initiating system shutdown...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop dashboard if running
        if self.dashboard_process and self.dashboard_process.poll() is None:
            logger.info("🛑 Stopping dashboard...")
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
            logger.info("✅ Dashboard stopped")
        
        if self.pipeline:
            await self.pipeline.cleanup()
        
        # Save final statistics
        await self._save_final_stats()
        
        logger.info("✅ System shutdown completed")
    
    async def _save_final_stats(self):
        """Save final system statistics."""
        try:
            stats_file = OUTPUT_DIR / "final_system_stats.json"
            stats_data = {
                "system_stats": {
                    "total_feedback_processed": self.stats.total_feedback_processed,
                    "total_tickets_generated": self.stats.total_tickets_generated,
                    "success_rate": self.stats.success_rate,
                    "last_updated": self.stats.last_updated.isoformat() if self.stats.last_updated else None
                },
                "performance_summary": performance_logger.get_summary(),
                "shutdown_time": datetime.now().isoformat()
            }
            
            import json
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2, default=str)
            
            logger.info(f"💾 Final statistics saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final statistics: {e}")


async def main():
    """Enhanced main function with comprehensive CLI support including dashboard."""
    parser = argparse.ArgumentParser(
        description="Intelligent User Feedback Analysis and Action System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --batch                    # Process all files once
  python main.py --monitor --interval 30   # Monitor for new files every 30s
  python main.py --file data/reviews.csv   # Process specific file
  python main.py --status                  # Show system status
  python main.py --dashboard               # Start web dashboard
  python main.py --dashboard --port 8502   # Start dashboard on custom port
        """
    )
    
    parser.add_argument(
        "--batch", 
        action="store_true",
        help="Process all files in batch mode"
    )
    
    parser.add_argument(
        "--monitor", 
        action="store_true",
        help="Start continuous monitoring for new files"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true", 
        help="Start the web dashboard"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Dashboard port (default: 8501)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost", 
        help="Dashboard host (default: localhost)"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Monitoring check interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--file", 
        type=Path,
        help="Process a specific file"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Show system status and exit"
    )
    
    parser.add_argument(
        "--config", 
        type=Path,
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize application
    app = FeedbackSystemApp()
    
    # Load custom configuration if provided
    config_override = None
    if args.config and args.config.exists():
        import json
        with open(args.config) as f:
            config_override = json.load(f)
        logger.info(f"📄 Loaded custom configuration from {args.config}")
    
    # Initialize system
    if not await app.initialize(config_override):
        logger.error("❌ Failed to initialize system")
        return 1
    
    try:
        # Handle different execution modes
        if args.status:
            status = await app.get_system_status()
            print("📊 System Status:")
            print(f"   Files processed: {status['system_stats']['total_feedback_processed']}")
            print(f"   Tickets generated: {status['system_stats']['total_tickets_generated']}")
            print(f"   Success rate: {status['system_stats']['success_rate']:.1%}")
            print(f"   Pipeline status: {status['pipeline_status']}")
            print(f"   Dashboard status: {status['dashboard_status']}")
            print(f"   Input files: {status['input_files_count']}")
            print(f"   Output files: {status['output_files_count']}")
            return 0
        
        elif args.dashboard:
            logger.info("🌐 Starting dashboard mode...")
            
            # Start dashboard
            if app.start_dashboard(args.port, args.host):
                print(f"\n🎉 Dashboard started successfully!")
                print(f"🔗 Access it at: http://{args.host}:{args.port}")
                print(f"📊 Use the dashboard to:")
                print(f"   - Upload and process feedback files")
                print(f"   - View generated tickets and analytics")
                print(f"   - Monitor system performance")
                print(f"   - Configure system settings")
                print(f"\n⌨️  Press Ctrl+C to stop the dashboard")
                
                # Keep the main process running while dashboard is active
                try:
                    while app.dashboard_process and app.dashboard_process.poll() is None:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    logger.info("⌨️ Received keyboard interrupt")
                
                return 0
            else:
                logger.error("❌ Failed to start dashboard")
                return 1
        
        elif args.file:
            if not args.file.exists():
                logger.error(f"❌ File not found: {args.file}")
                return 1
            
            logger.info(f"📄 Processing single file: {args.file}")
            result = await app.process_batch([args.file])
            
            if result.get('success_rate', 0.0) > 0.5:
                logger.info("✅ File processed successfully")
                print(f"\n🎉 Processing completed!")
                print(f"📊 Results:")
                print(f"   - Processed: {result.get('processed_feedback_count', 0)} feedback items")
                print(f"   - Generated: {result.get('generated_tickets_count', 0)} tickets")
                print(f"   - Success rate: {result.get('success_rate', 0.0):.1%}")
                print(f"   - Processing time: {result.get('total_processing_time', 0.0):.2f}s")
                return 0
            else:
                logger.error("❌ File processing failed")
                return 1
        
        elif args.monitor:
            logger.info("👀 Starting continuous monitoring mode...")
            print(f"\n👀 Monitoring {INPUT_DIR} for new files...")
            print(f"🔄 Check interval: {args.interval} seconds")
            print(f"⌨️  Press Ctrl+C to stop monitoring")
            
            await app.start_continuous_monitoring(args.interval)
            return 0
        
        elif args.batch:
            logger.info("🔄 Starting batch processing mode...")
            result = await app.process_batch()
            
            if result.get('success_rate', 0.0) > 0.5:
                logger.info("✅ Batch processing completed successfully")
                print(f"\n🎉 Batch processing completed!")
                print(f"📊 Results:")
                print(f"   - Processed: {result.get('processed_feedback_count', 0)} feedback items")
                print(f"   - Generated: {result.get('generated_tickets_count', 0)} tickets")
                print(f"   - Success rate: {result.get('success_rate', 0.0):.1%}")
                print(f"   - Processing time: {result.get('total_processing_time', 0.0):.2f}s")
                print(f"📁 Check '{OUTPUT_DIR}' for generated tickets and logs")
                return 0
            else:
                logger.error("❌ Batch processing failed")
                return 1
        
        else:
            # Default: show help and suggest dashboard
            print("🤖 Intelligent User Feedback Analysis System")
            print("=" * 50)
            print("No command specified. Here are your options:")
            print()
            print("🌐 Start Web Dashboard (Recommended):")
            print("   python main.py --dashboard")
            print()
            print("🔄 Process Files:")
            print("   python main.py --batch")
            print()
            print("👀 Monitor for New Files:")
            print("   python main.py --monitor")
            print()
            print("📊 Check System Status:")
            print("   python main.py --status")
            print()
            print("❓ Show All Options:")
            print("   python main.py --help")
            print()
            print("🚀 Quick Start: python main.py --dashboard")
            return 0
    
    except KeyboardInterrupt:
        logger.info("⌨️ Received keyboard interrupt")
        await app.shutdown()
        return 0
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        await app.shutdown()
        return 1
    
    finally:
        await app.shutdown()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)