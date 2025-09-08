"""
Data repository layer for the Intelligent Feedback Analysis System.
Provides abstraction layer for data storage and retrieval operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

from ..core.data_models import FeedbackItem, Ticket, Classification, ProcessingLog, Metric
from ..utils.logger import get_logger


class BaseRepository(ABC):
    """Abstract base class for data repositories."""
    
    @abstractmethod
    def save(self, entity: Any) -> bool:
        """Save an entity to the repository."""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: str) -> Optional[Any]:
        """Find an entity by ID."""
        pass
    
    @abstractmethod
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Find all entities matching filters."""
        pass
    
    @abstractmethod
    def update(self, entity: Any) -> bool:
        """Update an entity in the repository."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete an entity from the repository."""
        pass


class DataRepository:
    """
    Main data repository providing unified access to all data operations.
    
    INPUT SOURCES:
    - CSV files (app_store_reviews.csv, support_emails.csv)
    - JSON data streams
    - API responses
    - Database records
    
    OUTPUT DESTINATIONS:
    - SQLite database
    - CSV files (generated_tickets.csv, processing_log.csv, metrics.csv)
    - JSON exports
    - Excel reports
    """
    
    def __init__(self, db_path: str = "data/feedback_system.db"):
        """Initialize data repository with database connection."""
        self.db_path = db_path
        self.logger = get_logger("data_repository")
        
        # Initialize sub-repositories
        self.feedback_repo = FeedbackRepository(db_path)
        self.ticket_repo = TicketRepository(db_path)
        self.classification_repo = ClassificationRepository(db_path)
        self.log_repo = ProcessingLogRepository(db_path)
        self.metric_repo = MetricRepository(db_path)
    
    # INPUT OPERATIONS
    def import_feedback_data(self, data_source: Union[str, List[Dict]], source_type: str) -> Dict[str, Any]:
        """
        Import feedback data from various sources.
        
        INPUT FORMATS:
        - CSV file path
        - List of dictionaries
        - JSON string
        - API response data
        
        Returns:
            Import results summary
        """
        try:
            if isinstance(data_source, str) and data_source.endswith('.csv'):
                # CSV file input
                df = pd.read_csv(data_source)
                feedback_items = self._csv_to_feedback_items(df, source_type)
            elif isinstance(data_source, list):
                # List of dictionaries input
                feedback_items = self._dicts_to_feedback_items(data_source, source_type)
            else:
                raise ValueError(f"Unsupported data source type: {type(data_source)}")
            
            # Save to repository
            saved_count = 0
            for item in feedback_items:
                if self.feedback_repo.save(item):
                    saved_count += 1
            
            return {
                "success": True,
                "total_items": len(feedback_items),
                "saved_items": saved_count,
                "source_type": source_type
            }
            
        except Exception as e:
            self.logger.error(f"Error importing feedback data: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_items": 0,
                "saved_items": 0
            }
    
    def import_expected_results(self, csv_path: str) -> Dict[str, Any]:
        """
        Import expected classification results for evaluation.
        
        INPUT: expected_classifications.csv
        """
        try:
            df = pd.read_csv(csv_path)
            imported_count = 0
            
            for _, row in df.iterrows():
                classification = Classification(
                    category=row["category"],
                    confidence=1.0,  # Expected results are 100% confident
                    reasoning=f"Expected result for {row['source_id']}"
                )
                
                if self.classification_repo.save_expected(row["source_id"], classification):
                    imported_count += 1
            
            return {
                "success": True,
                "imported_count": imported_count,
                "total_rows": len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error importing expected results: {e}")
            return {"success": False, "error": str(e)}
    
    # OUTPUT OPERATIONS  
    def export_tickets(self, output_path: str, format: str = "csv", filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Export tickets to various formats.
        
        OUTPUT FORMATS:
        - CSV: generated_tickets.csv
        - JSON: generated_tickets.json  
        - Excel: generated_tickets.xlsx
        """
        try:
            tickets = self.ticket_repo.find_all(filters)
            
            if format == "csv":
                ticket_dicts = [self._ticket_to_dict(ticket) for ticket in tickets]
                df = pd.DataFrame(ticket_dicts)
                df.to_csv(output_path, index=False)
            
            elif format == "json":
                import json
                ticket_dicts = [self._ticket_to_dict(ticket) for ticket in tickets]
                with open(output_path, 'w') as f:
                    json.dump(ticket_dicts, f, indent=2, default=str)
            
            elif format == "excel":
                ticket_dicts = [self._ticket_to_dict(ticket) for ticket in tickets]
                df = pd.DataFrame(ticket_dicts)
                df.to_excel(output_path, index=False)
            
            return {
                "success": True,
                "exported_count": len(tickets),
                "output_path": output_path,
                "format": format
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting tickets: {e}")
            return {"success": False, "error": str(e)}
    
    def export_processing_logs(self, output_path: str, days: int = 7) -> Dict[str, Any]:
        """
        Export processing logs to CSV.
        
        OUTPUT: processing_log.csv
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            logs = self.log_repo.find_since(cutoff_date)
            
            log_dicts = [self._log_to_dict(log) for log in logs]
            df = pd.DataFrame(log_dicts)
            df.to_csv(output_path, index=False)
            
            return {
                "success": True,
                "exported_count": len(logs),
                "output_path": output_path,
                "days_included": days
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting logs: {e}")
            return {"success": False, "error": str(e)}
    
    def export_metrics(self, output_path: str, days: int = 30) -> Dict[str, Any]:
        """
        Export metrics to CSV.
        
        OUTPUT: metrics.csv
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            metrics = self.metric_repo.find_since(cutoff_date)
            
            metric_dicts = [self._metric_to_dict(metric) for metric in metrics]
            df = pd.DataFrame(metric_dicts)
            df.to_csv(output_path, index=False)
            
            return {
                "success": True,
                "exported_count": len(metrics),
                "output_path": output_path,
                "days_included": days
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return {"success": False, "error": str(e)}
    
    def export_analytics_summary(self, output_path: str) -> Dict[str, Any]:
        """
        Export comprehensive analytics summary.
        
        OUTPUT: analytics_summary.json
        """
        try:
            summary = {
                "generated_at": datetime.now().isoformat(),
                "total_feedback": len(self.feedback_repo.find_all()),
                "total_tickets": len(self.ticket_repo.find_all()),
                "category_distribution": self._get_category_distribution(),
                "priority_distribution": self._get_priority_distribution(),
                "processing_stats": self._get_processing_stats(),
                "quality_metrics": self._get_quality_metrics()
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return {
                "success": True,
                "output_path": output_path,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting analytics: {e}")
            return {"success": False, "error": str(e)}
    
    # QUERY OPERATIONS
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of all feedback data."""
        return {
            "total_feedback": len(self.feedback_repo.find_all()),
            "app_store_reviews": len(self.feedback_repo.find_by_source_type("app_store_review")),
            "support_emails": len(self.feedback_repo.find_by_source_type("support_email")),
            "latest_feedback": self.feedback_repo.find_latest(5)
        }
    
    def get_ticket_summary(self) -> Dict[str, Any]:
        """Get summary of all ticket data."""
        return {
            "total_tickets": len(self.ticket_repo.find_all()),
            "category_distribution": self._get_category_distribution(),
            "priority_distribution": self._get_priority_distribution(),
            "recent_tickets": self.ticket_repo.find_latest(10)
        }
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing activity."""
        return {
            "total_logs": len(self.log_repo.find_all()),
            "success_rate": self.log_repo.get_success_rate(),
            "agent_performance": self.log_repo.get_agent_performance(),
            "recent_activity": self.log_repo.find_latest(20)
        }
    
    # HELPER METHODS
    def _csv_to_feedback_items(self, df: pd.DataFrame, source_type: str) -> List[FeedbackItem]:
        """Convert CSV DataFrame to FeedbackItem objects."""
        items = []
        
        for _, row in df.iterrows():
            if source_type == "app_store_reviews":
                item = FeedbackItem(
                    id=row["review_id"],
                    source_type="app_store_review",
                    content=row["review_text"],
                    rating=int(row["rating"]) if pd.notna(row["rating"]) else None,
                    platform=row.get("platform"),
                    metadata={
                        "user_name": row.get("user_name", ""),
                        "app_version": row.get("app_version", ""),
                        "date": row.get("date", "")
                    }
                )
            elif source_type == "support_emails":
                content = f"Subject: {row.get('subject', '')}\n\n{row.get('body', '')}"
                item = FeedbackItem(
                    id=row["email_id"],
                    source_type="support_email",
                    content=content,
                    metadata={
                        "subject": row.get("subject", ""),
                        "sender_email": row.get("sender_email", ""),
                        "priority": row.get("priority", ""),
                        "timestamp": row.get("timestamp", "")
                    }
                )
            else:
                continue
            
            items.append(item)
        
        return items
    
    def _dicts_to_feedback_items(self, data: List[Dict], source_type: str) -> List[FeedbackItem]:
        """Convert list of dictionaries to FeedbackItem objects."""
        items = []
        
        for item_data in data:
            item = FeedbackItem(
                id=item_data["id"],
                source_type=source_type,
                content=item_data["content"],
                rating=item_data.get("rating"),
                platform=item_data.get("platform"),
                metadata=item_data.get("metadata", {})
            )
            items.append(item)
        
        return items
    
    def _ticket_to_dict(self, ticket: Ticket) -> Dict[str, Any]:
        """Convert Ticket object to dictionary for export."""
        return ticket.to_dict()
    
    def _log_to_dict(self, log: ProcessingLog) -> Dict[str, Any]:
        """Convert ProcessingLog object to dictionary for export."""
        return log.to_dict()
    
    def _metric_to_dict(self, metric: Metric) -> Dict[str, Any]:
        """Convert Metric object to dictionary for export."""
        return metric.to_dict()
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of tickets by category."""
        return self.ticket_repo.get_category_distribution()
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of tickets by priority."""
        return self.ticket_repo.get_priority_distribution()
    
    def _get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.log_repo.get_processing_stats()
    
    def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for tickets."""
        return self.ticket_repo.get_quality_metrics()


class FeedbackRepository(BaseRepository):
    """Repository for FeedbackItem entities."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = get_logger("feedback_repository")
    
    def save(self, feedback_item: FeedbackItem) -> bool:
        """Save feedback item to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO feedback_items 
                    (id, source_type, content, rating, platform, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_item.id,
                    feedback_item.source_type,
                    feedback_item.content,
                    feedback_item.rating,
                    feedback_item.platform,
                    str(feedback_item.metadata),
                    feedback_item.timestamp
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving feedback item: {e}")
            return False
    
    def find_by_id(self, feedback_id: str) -> Optional[FeedbackItem]:
        """Find feedback item by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM feedback_items WHERE id = ?", (feedback_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_feedback_item(row)
                return None
        except Exception as e:
            self.logger.error(f"Error finding feedback item: {e}")
            return None
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[FeedbackItem]:
        """Find all feedback items matching filters."""
        try:
            query = "SELECT * FROM feedback_items"
            params = []
            
            if filters:
                conditions = []
                if "source_type" in filters:
                    conditions.append("source_type = ?")
                    params.append(filters["source_type"])
                if "rating_min" in filters:
                    conditions.append("rating >= ?")
                    params.append(filters["rating_min"])
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_feedback_item(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error finding feedback items: {e}")
            return []
    
    def find_by_source_type(self, source_type: str) -> List[FeedbackItem]:
        """Find feedback items by source type."""
        return self.find_all({"source_type": source_type})
    
    def find_latest(self, limit: int = 10) -> List[FeedbackItem]:
        """Find latest feedback items."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM feedback_items 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                
                return [self._row_to_feedback_item(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error finding latest feedback: {e}")
            return []
    
    def update(self, feedback_item: FeedbackItem) -> bool:
        """Update feedback item."""
        return self.save(feedback_item)  # INSERT OR REPLACE handles updates
    
    def delete(self, feedback_id: str) -> bool:
        """Delete feedback item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM feedback_items WHERE id = ?", (feedback_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Error deleting feedback item: {e}")
            return False
    
    def _row_to_feedback_item(self, row) -> FeedbackItem:
        """Convert database row to FeedbackItem."""
        import json
        
        return FeedbackItem(
            id=row[0],
            source_type=row[1],
            content=row[2],
            rating=row[3],
            platform=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
            timestamp=pd.to_datetime(row[6]) if row[6] else None
        )


class TicketRepository(BaseRepository):
    """Repository for Ticket entities."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = get_logger("ticket_repository")
    
    def save(self, ticket: Ticket) -> bool:
        """Save ticket to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO tickets 
                    (ticket_id, source_id, source_type, category, priority, title, description,
                     technical_details, agent_confidence, tags, estimated_effort, assigned_team, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticket.ticket_id,
                    ticket.source_id,
                    ticket.source_type,
                    ticket.category.value,
                    ticket.priority.value,
                    ticket.title,
                    ticket.description,
                    str(ticket.technical_details),
                    ticket.agent_confidence,
                    ",".join(ticket.tags),
                    ticket.estimated_effort,
                    ticket.assigned_team,
                    ticket.created_at
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving ticket: {e}")
            return False
    
    def find_by_id(self, ticket_id: str) -> Optional[Ticket]:
        """Find ticket by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_ticket(row)
                return None
        except Exception as e:
            self.logger.error(f"Error finding ticket: {e}")
            return None
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Ticket]:
        """Find all tickets matching filters."""
        try:
            query = "SELECT * FROM tickets"
            params = []
            
            if filters:
                conditions = []
                if "category" in filters:
                    conditions.append("category = ?")
                    params.append(filters["category"])
                if "priority" in filters:
                    conditions.append("priority = ?")
                    params.append(filters["priority"])
                if "assigned_team" in filters:
                    conditions.append("assigned_team = ?")
                    params.append(filters["assigned_team"])
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_ticket(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error finding tickets: {e}")
            return []
    
    def find_latest(self, limit: int = 10) -> List[Ticket]:
        """Find latest tickets."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tickets 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                
                return [self._row_to_ticket(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error finding latest tickets: {e}")
            return []
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of tickets by category."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM tickets 
                    GROUP BY category
                """)
                rows = cursor.fetchall()
                
                return {row[0]: row[1] for row in rows}
        except Exception as e:
            self.logger.error(f"Error getting category distribution: {e}")
            return {}
    
    def get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of tickets by priority."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT priority, COUNT(*) as count 
                    FROM tickets 
                    GROUP BY priority
                """)
                rows = cursor.fetchall()
                
                return {row[0]: row[1] for row in rows}
        except Exception as e:
            self.logger.error(f"Error getting priority distribution: {e}")
            return {}
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics for tickets."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Average confidence
                cursor.execute("SELECT AVG(agent_confidence) FROM tickets WHERE agent_confidence IS NOT NULL")
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Tickets with high confidence
                cursor.execute("SELECT COUNT(*) FROM tickets WHERE agent_confidence > 0.8")
                high_confidence_count = cursor.fetchone()[0]
                
                # Total tickets
                cursor.execute("SELECT COUNT(*) FROM tickets")
                total_tickets = cursor.fetchone()[0]
                
                return {
                    "avg_confidence": avg_confidence,
                    "high_confidence_rate": high_confidence_count / total_tickets if total_tickets > 0 else 0,
                    "total_tickets": total_tickets
                }
        except Exception as e:
            self.logger.error(f"Error getting quality metrics: {e}")
            return {}
    
    def update(self, ticket: Ticket) -> bool:
        """Update ticket."""
        return self.save(ticket)  # INSERT OR REPLACE handles updates
    
    def delete(self, ticket_id: str) -> bool:
        """Delete ticket."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tickets WHERE ticket_id = ?", (ticket_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Error deleting ticket: {e}")
            return False
    
    def _row_to_ticket(self, row) -> Ticket:
        """Convert database row to Ticket."""
        from ..core.data_models import FeedbackCategory, Priority
        import json
        
        return Ticket(
            ticket_id=row[0],
            source_id=row[1],
            source_type=row[2],
            category=FeedbackCategory(row[3]),
            priority=Priority(row[4]),
            title=row[5],
            description=row[6],
            technical_details=json.loads(row[7]) if row[7] else {},
            agent_confidence=row[8],
            tags=row[9].split(",") if row[9] else [],
            estimated_effort=row[10],
            assigned_team=row[11],
            created_at=pd.to_datetime(row[14]) if row[14] else datetime.utcnow()
        )


class ClassificationRepository(BaseRepository):
    """Repository for Classification entities."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = get_logger("classification_repository")
    
    def save(self, classification: Classification, feedback_id: str) -> bool:
        """Save classification linked to feedback item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO classifications 
                    (feedback_id, category, confidence, sentiment, keywords, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    feedback_id,
                    classification.category.value,
                    classification.confidence,
                    classification.sentiment,
                    ",".join(classification.keywords),
                    classification.reasoning
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving classification: {e}")
            return False
    
    def save_expected(self, feedback_id: str, classification: Classification) -> bool:
        """Save expected classification for evaluation."""
        # Could be extended to separate table for expected vs actual
        return self.save(classification, feedback_id)
    
    def find_by_id(self, classification_id: int) -> Optional[Classification]:
        """Find classification by ID."""
        # Implementation would depend on specific needs
        pass
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Classification]:
        """Find all classifications."""
        # Implementation would depend on specific needs
        pass
    
    def update(self, classification: Classification) -> bool:
        """Update classification."""
        pass
    
    def delete(self, classification_id: str) -> bool:
        """Delete classification."""
        pass


class ProcessingLogRepository(BaseRepository):
    """Repository for ProcessingLog entities."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = get_logger("log_repository")
    
    def save(self, log: ProcessingLog) -> bool:
        """Save processing log entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_logs 
                    (timestamp, source_id, agent_name, action, details, confidence_score, duration_ms, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log.timestamp,
                    log.source_id,
                    log.agent_name,
                    log.action,
                    log.details,
                    log.confidence_score,
                    log.duration_ms,
                    log.status
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving log: {e}")
            return False
    
    def find_since(self, since_date: datetime) -> List[ProcessingLog]:
        """Find logs since a specific date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM processing_logs 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                """, (since_date,))
                rows = cursor.fetchall()
                
                return [self._row_to_log(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error finding logs: {e}")
            return []
    
    def get_success_rate(self) -> float:
        """Get overall success rate from logs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful
                    FROM processing_logs
                """)
                row = cursor.fetchone()
                
                total, successful = row[0], row[1]
                return successful / total if total > 0 else 0
        except Exception as e:
            self.logger.error(f"Error getting success rate: {e}")
            return 0
    
    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics by agent."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        agent_name,
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_runs,
                        AVG(duration_ms) as avg_duration,
                        AVG(confidence_score) as avg_confidence
                    FROM processing_logs 
                    WHERE agent_name IS NOT NULL
                    GROUP BY agent_name
                """)
                rows = cursor.fetchall()
                
                performance = {}
                for row in rows:
                    agent_name = row[0]
                    performance[agent_name] = {
                        "total_runs": row[1],
                        "successful_runs": row[2],
                        "success_rate": row[2] / row[1] if row[1] > 0 else 0,
                        "avg_duration_ms": row[3] or 0,
                        "avg_confidence": row[4] or 0
                    }
                
                return performance
        except Exception as e:
            self.logger.error(f"Error getting agent performance: {e}")
            return {}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        return {
            "success_rate": self.get_success_rate(),
            "agent_performance": self.get_agent_performance(),
            "total_logs": len(self.find_all())
        }
    
    def find_by_id(self, log_id: str) -> Optional[ProcessingLog]:
        """Find log by ID."""
        pass
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[ProcessingLog]:
        """Find all logs."""
        return self.find_since(datetime.min)
    
    def find_latest(self, limit: int = 20) -> List[ProcessingLog]:
        """Find latest log entries."""
        return self.find_since(datetime.now() - timedelta(days=1))[:limit]
    
    def update(self, log: ProcessingLog) -> bool:
        """Update log entry."""
        pass
    
    def delete(self, log_id: str) -> bool:
        """Delete log entry."""
        pass
    
    def _row_to_log(self, row) -> ProcessingLog:
        """Convert database row to ProcessingLog."""
        return ProcessingLog(
            timestamp=pd.to_datetime(row[1]) if row[1] else datetime.utcnow(),
            source_id=row[2],
            agent_name=row[3],
            action=row[4],
            details=row[5],
            confidence_score=row[6],
            duration_ms=row[7],
            status=row[8]
        )


class MetricRepository(BaseRepository):
    """Repository for Metric entities."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = get_logger("metric_repository")
    
    def save(self, metric: Metric) -> bool:
        """Save metric to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO metrics 
                    (metric_name, value, timestamp, details, category)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    metric.value,
                    metric.timestamp,
                    metric.details,
                    metric.category
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving metric: {e}")
            return False
    
    def find_since(self, since_date: datetime) -> List[Metric]:
        """Find metrics since a specific date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM metrics 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp DESC
                """, (since_date,))
                rows = cursor.fetchall()
                
                return [self._row_to_metric(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error finding metrics: {e}")
            return []
    
    def find_by_id(self, metric_id: str) -> Optional[Metric]:
        """Find metric by ID."""
        pass
    
    def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Metric]:
        """Find all metrics."""
        return self.find_since(datetime.min)
    
    def update(self, metric: Metric) -> bool:
        """Update metric."""
        pass
    
    def delete(self, metric_id: str) -> bool:
        """Delete metric."""
        pass
    
    def _row_to_metric(self, row) -> Metric:
        """Convert database row to Metric."""
        return Metric(
            name=row[1],
            value=row[2],
            timestamp=pd.to_datetime(row[3]) if row[3] else datetime.utcnow(),
            details=row[4],
            category=row[5]
        )