from src.core.data_models import create_pipeline_result, ProcessingStatus
"""
Main processing pipeline that orchestrates all agents.
"""

import asyncio
import time
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from .data_models import (
    FeedbackItem, PipelineResult, AgentResult, ProcessingLog, 
    Metric, SystemStats, Ticket
)
from src.core.data_models import create_pipeline_result, ProcessingStatus
from src.agents.csv_reader_agent import CSVReaderAgent
from src.agents.feedbac_classifier_agent import FeedbackClassifierAgent
from src.agents.bug_analysis_agent import BugAnalysisAgent
from src.agents.feature_extractor_agent import FeatureExtractorAgent

from src.agents.ticket_creator_agent import TicketCreatorAgent
from src.agents.quality_cretic_agent import QualityCriticAgent

from src.utils.csv_handler import CSVHandler
from src.utils.logger import get_logger, PerformanceLogger
from config.settings import OUTPUT_DIR, PROCESSING_SETTINGS


class FeedbackPipeline:
    """Main pipeline for processing user feedback through multiple agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feedback processing pipeline.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config or {}
        self.logger = get_logger("pipeline")
        self.perf_logger = PerformanceLogger("pipeline")
        self.csv_handler = CSVHandler()
        
        # Initialize agents
        self._initialize_agents()
        
        # Pipeline state
        self.stats = SystemStats()
        self.processing_logs = []
        self.metrics = []
        self.results = []
        
        # Performance tracking
        self.start_time = None
        self.end_time = None

    def cleanup(self):
        """Cleanup method for pipeline"""
        self.logger.info("Pipeline cleanup completed")
        
    def _initialize_agents(self):
        """Initialize all agents with their configurations."""
        self.logger.info("Initializing agents...")
        
        # CSV Reader Agent
        self.csv_reader = CSVReaderAgent(
            config=self.config.get("csv_reader", {})
        )
        
        # Feedback Classifier Agent
        self.classifier = FeedbackClassifierAgent(
            config=self.config.get("classifier", {})
        )
        
        # Bug Analysis Agent
        self.bug_analyzer = BugAnalysisAgent(
            config=self.config.get("bug_analyzer", {})
        )
        
        # Feature Extractor Agent
        self.feature_extractor = FeatureExtractorAgent(
            config=self.config.get("feature_extractor", {})
        )
        
        # Ticket Creator Agent
        self.ticket_creator = TicketCreatorAgent(
            config=self.config.get("ticket_creator", {})
        )
        
        # Quality Critic Agent
        self.quality_critic = QualityCriticAgent(
            config=self.config.get("quality_critic", {})
        )
        
        self.agents = [
            self.csv_reader,
            self.classifier,
            self.bug_analyzer,
            self.feature_extractor,
            self.ticket_creator,
            self.quality_critic
        ]
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the complete feedback processing pipeline.
        
        Returns:
            Dictionary containing processing results and statistics
        """
        self.start_time = time.time()
        self.logger.info("Starting feedback processing pipeline")
        
        try:
            # Step 1: Read input data
            self.logger.info("Step 1: Reading input data...")
            feedback_items = await self._read_input_data()
            
            if not feedback_items:
                self.logger.error("No feedback items found to process")
                return {"error": "No input data found"}
            
            self.logger.info(f"Found {len(feedback_items)} feedback items to process")
            
            # Step 2: Process feedback items
            self.logger.info("Step 2: Processing feedback items...")
            processed_results = await self._process_feedback_batch(feedback_items)
            
            # Step 3: Generate output files
            if not self.config.get("dry_run", False):
                self.logger.info("Step 3: Generating output files...")
                await self._generate_output_files(processed_results)
            else:
                self.logger.info("Step 3: Skipping output generation (dry run mode)")
            
            # Step 4: Generate final statistics
            final_stats = self._generate_final_statistics(processed_results)
            
            self.end_time = time.time()
            total_time = self.end_time - self.start_time
            
            self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            
            return {
                "success": True,
                "total_processed": len(feedback_items),
                "successful_results": len([r for r in processed_results if r.is_successful()]),
                "tickets_generated": len([r for r in processed_results if r.generated_ticket]),
                "total_time": total_time,
                "avg_confidence": final_stats.get("avg_confidence", 0.0),
                "stats": final_stats
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _read_input_data(self) -> List[FeedbackItem]:
        """Read input data using CSV Reader Agent."""
        input_config = {
            "file_type": self.config.get("file_type", "all")
        }
        
        # Override file paths if provided in config
        if "input_dir" in self.config:
            input_dir = Path(self.config["input_dir"])
            input_config["file_paths"] = [
                input_dir / "app_store_reviews.csv",
                input_dir / "support_emails.csv"
            ]
        
        result = await self.csv_reader.execute(input_config)
        
        if not result.success:
            raise RuntimeError(f"Failed to read input data: {result.error_message}")
        
        return result.data or []
    
    async def _process_feedback_batch(self, feedback_items: List[FeedbackItem]) -> List[PipelineResult]:
        """Process a batch of feedback items through the agent pipeline."""
        batch_size = self.config.get("batch_size", PROCESSING_SETTINGS["batch_size"])
        max_concurrent = self.config.get("max_concurrent", PROCESSING_SETTINGS["max_concurrent_agents"])
        
        self.logger.info(f"Processing {len(feedback_items)} items in batches of {batch_size}")
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(feedback_items), batch_size):
            batch = feedback_items[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(feedback_items) + batch_size - 1) // batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            # Create semaphore to limit concurrent processing
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # Process batch items concurrently
            tasks = [
                self._process_single_feedback(item, semaphore) 
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing item {batch[j].id}: {result}")
                    # Create failed result
                    failed_result = create_pipeline_result(
                        id=f"failed_{batch[j].id}",
                        input_feedback_count=1,
                        processed_feedback_count=0,
                        generated_tickets_count=0,
                        success_rate=0.0,
                        total_processing_time=0.0
                    )
                    all_results.append(failed_result)
                else:
                    all_results.append(result)
            
            # Log batch completion
            #successful_in_batch = sum(1 for r in batch_results if not isinstance(r, Exception) and r.is_successful())
            successful_in_batch = sum(1 for r in batch_results if not isinstance(r, Exception) and len(r.generated_tickets) > 0)
            self.logger.info(f"Batch {batch_num} completed: {successful_in_batch}/{len(batch)} successful")
        
        return all_results
    
    async def _process_single_feedback(self, item: FeedbackItem, semaphore: asyncio.Semaphore) -> PipelineResult:
        """Process a single feedback item through the agent pipeline."""
        async with semaphore:
            self.perf_logger.start_timer(f"process_item_{item.id}")
            
            result = create_pipeline_result(
    id=f"batch_result",
    input_feedback_count=0,
    processed_feedback_count=0,
    generated_tickets_count=0,
    success_rate=0.0,
    total_processing_time=0.0
)
            current_data = item
            result = create_pipeline_result(
                id=f"process_{item.id}",
                input_feedback_count=1,
                processed_feedback_count=0,
                generated_tickets_count=0,
                success_rate=0.0,
                total_processing_time=0.0
            )
            try:
                # Step 1: Classify feedback
                classification_result = await self.classifier.execute(current_data)
                result.agent_results["classifier"] = classification_result
                
                if classification_result.success:
                    if isinstance(classification_result.data, dict):
                        # Convert dict to Classification object
                        from src.core.data_models import Classification, FeedbackCategory
                        result.classification = Classification(
                            feedback_id=item.id,
                            predicted_category=FeedbackCategory.from_string(classification_result.data.get('category', 'other')),
                            confidence_score=classification_result.data.get('confidence', 0.0),
                            reasoning=classification_result.data.get('reasoning', ''),
                            keywords_found=classification_result.data.get('keywords', [])
                        )
                    else:
                        result.classification = classification_result.data
                    current_data = {"item": item, "classification": result.classification}
                    
                    # Step 2: Analyze based on category
                    if result.classification.predicted_category.value == "Bug":
                        bug_result = await self.bug_analyzer.execute(current_data)
                        result.agent_results["bug_analyzer"] = bug_result
                        if bug_result.success:
                            result.bug_details = bug_result.data
                    
                    elif result.classification.predicted_category.value == "Feature Request":
                        feature_result = await self.feature_extractor.execute(current_data)
                        result.agent_results["feature_extractor"] = feature_result
                        if feature_result.success:
                            result.feature_details = feature_result.data
                    
                    # Step 3: Create ticket
                    ticket_data = {
                        "item": item,
                        "classification": result.classification,
                        "bug_details": result.bug_details,
                        "feature_details": result.feature_details
                    }
                    
                    ticket_result = await self.ticket_creator.execute(ticket_data)
                    result.agent_results["ticket_creator"] = ticket_result
                    
                    if ticket_result.success:
                        result.generated_ticket = ticket_result.data
                        
                        # Step 4: Quality review
                        quality_result = await self.quality_critic.execute(result.generated_ticket)
                        # quality_result = await self.quality_critic.execute({
                        #     "ticket": result.generated_ticket,
                        #     "original_item": item
                        # })
                        result.agent_results["quality_critic"] = quality_result
                        
#                        if quality_result.success and quality_result.data:
#                            # Update ticket with quality improvements
#                            result.generated_ticket = quality_result.data
                
                # Calculate overall confidence
                confidences = [
                    r.confidence for r in result.agent_results.values() 
                    if r.confidence is not None
                ]
                
                # Update stats
                self.stats.update_stats(result)
                
                # Log processing completion
                result.processing_logs.append(ProcessingLog(
                    feedback_id=item.id,
                    agent_name="pipeline",
                    level='INFO',
                    message=f"Processed with confidence: {result.overall_confidence:.2f}",
                   # confidence_score=result.overall_confidence
                   id=item.id
                ))
                
            except Exception as e:
                self.logger.error(f"Error processing item {item.id}: {e}")
                result.processing_logs.append(ProcessingLog(id="log_" + str(item.id), level="ERROR", message=f"Processing error: {str(e)}", agent_name="pipeline", feedback_id=item.id))
            
            finally:
                result.processing_time = self.perf_logger.end_timer(f"process_item_{item.id}")
            
            return result
    
    async def _generate_output_files(self, results: List[PipelineResult]):
        """Generate output CSV files from processing results."""
        output_dir = Path(self.config.get("output_dir", OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate tickets CSV
        # tickets = [r.generated_ticket for r in results if r.generated_ticket]

        tickets = []
        for result in results:
            # Use generated_tickets list instead of generated_ticket property
            if result.generated_tickets:
                tickets.extend(result.generated_tickets)
        
        if tickets:
            # tickets_data = [ticket.to_dict() for ticket in tickets]
            tickets_data = [ticket.to_dict() if hasattr(ticket, 'to_dict') else ticket for ticket in tickets]
            tickets_df = pd.DataFrame(tickets_data)
            
            tickets_file = output_dir / "generated_tickets.csv"
            success = self.csv_handler.write_csv(tickets_df, tickets_file)
            
            if success:
                self.logger.info(f"Generated {len(tickets)} tickets in {tickets_file}")
            else:
                self.logger.error(f"Failed to write tickets file: {tickets_file}")
        
        # Generate processing logs CSV
        all_logs = []
        for result in results:
            all_logs.extend(result.processing_logs)
        
        if all_logs:
            logs_data = [log.to_dict() for log in all_logs]
            logs_df = pd.DataFrame(logs_data)
            
            logs_file = output_dir / "processing_log.csv"
            success = self.csv_handler.write_csv(logs_df, logs_file)
            
            if success:
                self.logger.info(f"Generated {len(all_logs)} log entries in {logs_file}")
            else:
                self.logger.error(f"Failed to write logs file: {logs_file}")
        
        # Generate metrics CSV
        pipeline_metrics = self._calculate_pipeline_metrics(results)
        if pipeline_metrics:
            metrics_data = [metric.to_dict() for metric in pipeline_metrics]
            metrics_df = pd.DataFrame(metrics_data)
            
            metrics_file = output_dir / "metrics.csv"
            success = self.csv_handler.write_csv(metrics_df, metrics_file)
            
            if success:
                self.logger.info(f"Generated {len(pipeline_metrics)} metrics in {metrics_file}")
            else:
                self.logger.error(f"Failed to write metrics file: {metrics_file}")
    
    def _calculate_pipeline_metrics(self, results: List[PipelineResult]) -> List[Metric]:
        """Calculate pipeline performance metrics."""
        metrics = []
        
        if not results:
            return metrics
        
        # Basic metrics
        total_items = len(results)
        successful_items = len([r for r in results if r.is_successful()])
        tickets_generated = len([r for r in results if r.generated_ticket])
        
        metrics.extend([
            Metric("total_processed", total_items),
            Metric("successful_processed", successful_items),
            Metric("tickets_generated", tickets_generated),
            Metric("success_rate", successful_items / total_items if total_items > 0 else 0),
            Metric("ticket_generation_rate", tickets_generated / total_items if total_items > 0 else 0)
        ])
        
        # Processing time metrics
        processing_times = [r.processing_time for r in results if r.processing_time]
        if processing_times:
            metrics.extend([
                Metric("avg_processing_time", sum(processing_times) / len(processing_times)),
                Metric("max_processing_time", max(processing_times)),
                Metric("min_processing_time", min(processing_times)),
                Metric("total_processing_time", sum(processing_times))
            ])
        
        # Confidence metrics
        confidences = [r.overall_confidence for r in results if r.overall_confidence > 0]
        if confidences:
            metrics.extend([
                Metric("avg_confidence", sum(confidences) / len(confidences)),
                Metric("max_confidence", max(confidences)),
                Metric("min_confidence", min(confidences))
            ])
        
        # Category distribution
        categories = {}
        for result in results:
            if result.classification:
                category = result.classification.predicted_category.value
                categories[category] = categories.get(category, 0) + 1
        
        for category, count in categories.items():
            metrics.append(Metric(f"category_{category.lower().replace(' ', '_')}_count", count))
        
        # Agent performance metrics
        for agent in self.agents:
            agent_metrics = agent.get_metrics()
            for metric_name, value in agent_metrics.items():
                if isinstance(value, (int, float)):
                    metrics.append(Metric(f"agent_{agent.name}_{metric_name}", value))
        
        return metrics
    
    def _generate_final_statistics(self, results: List[PipelineResult]) -> Dict[str, Any]:
        """Generate final processing statistics."""
        if not results:
            return {}
        
        total_items = len(results)
        successful_items = len([r for r in results if r.is_successful()])
        tickets_generated = len([r for r in results if r.generated_ticket])
        
        # Calculate average confidence
        confidences = [r.overall_confidence for r in results if r.overall_confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Category distribution
        categories = {}
        for result in results:
            if result.classification:
                category = result.classification.predicted_category.value
                categories[category] = categories.get(category, 0) + 1
        
        # Processing time statistics
        processing_times = [r.processing_time for r in results if r.processing_time]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return {
            "total_items": total_items,
            "successful_items": successful_items,
            "tickets_generated": tickets_generated,
            "success_rate": successful_items / total_items if total_items > 0 else 0,
            "avg_confidence": avg_confidence,
            "avg_processing_time": avg_processing_time,
            "category_distribution": categories,
            "agent_performance": {agent.name: agent.get_metrics() for agent in self.agents}
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "is_running": self.start_time is not None and self.end_time is None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stats": self.stats.__dict__,
            "agent_health": {agent.name: agent.is_healthy() for agent in self.agents}
        }
    
    async def process_single_item(self, item: FeedbackItem) -> PipelineResult:
        """Process a single feedback item (useful for real-time processing)."""
        semaphore = asyncio.Semaphore(1)
        return await self._process_single_feedback(item, semaphore)