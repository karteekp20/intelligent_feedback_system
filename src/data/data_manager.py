"""
Data management utilities for the feedback analysis system.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import sqlite3
from contextlib import contextmanager

from ..utils.logger import get_logger
from ..utils.csv_handler import CSVHandler
from .data_validator import DataValidator
from config.settings import INPUT_DIR, OUTPUT_DIR, CSV_SCHEMAS


class DataManager:
    """Centralized data management for the feedback analysis system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("data_manager")
        self.csv_handler = CSVHandler()
        self.validator = DataValidator()
        
        # Database setup (SQLite for local storage)
        self.db_path = Path(self.config.get("database_path", "data/feedback_system.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Feedback items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_items (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    rating INTEGER,
                    platform TEXT,
                    metadata TEXT,
                    timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Classifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sentiment TEXT,
                    keywords TEXT,
                    reasoning TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feedback_id) REFERENCES feedback_items (id)
                )
            """)
            
            # Tickets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    technical_details TEXT,
                    agent_confidence REAL,
                    tags TEXT,
                    estimated_effort TEXT,
                    assigned_team TEXT,
                    status TEXT DEFAULT 'Open',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES feedback_items (id)
                )
            """)
            
            # Processing logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source_id TEXT,
                    agent_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,
                    confidence_score REAL,
                    duration_ms REAL,
                    status TEXT DEFAULT 'success'
                )
            """)
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    category TEXT
                )
            """)
            
            # System configuration table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    @contextmanager
    def get_db_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def import_csv_data(self, file_path: Union[str, Path], file_type: str) -> Dict[str, Any]:
        """
        Import data from CSV file into database.
        
        Args:
            file_path: Path to CSV file
            file_type: Type of CSV file (app_store_reviews, support_emails, etc.)
            
        Returns:
            Import results dictionary
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists and schema
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Validate schema
            expected_schema = CSV_SCHEMAS.get(file_type)
            if not expected_schema:
                raise ValueError(f"Unknown file type: {file_type}")
            
            validation_result = self.csv_handler.validate_csv_schema(file_path, expected_schema)
            if not validation_result["valid"]:
                raise ValueError(f"Schema validation failed: {validation_result['error']}")
            
            # Read CSV data
            df = self.csv_handler.read_csv(file_path)
            df = self.csv_handler.clean_csv_data(df)
            
            # Import based on file type
            if file_type == "app_store_reviews":
                imported_count = self._import_feedback_from_reviews(df)
            elif file_type == "support_emails":
                imported_count = self._import_feedback_from_emails(df)
            elif file_type == "generated_tickets":
                imported_count = self._import_tickets(df)
            elif file_type == "processing_log":
                imported_count = self._import_processing_logs(df)
            elif file_type == "metrics":
                imported_count = self._import_metrics(df)
            else:
                raise ValueError(f"Import not supported for file type: {file_type}")
            
            self.logger.info(f"Successfully imported {imported_count} records from {file_path}")
            
            return {
                "success": True,
                "imported_count": imported_count,
                "total_rows": len(df),
                "file_type": file_type,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error importing CSV data: {e}")
            return {
                "success": False,
                "error": str(e),
                "imported_count": 0
            }
    
    def _import_feedback_from_reviews(self, df: pd.DataFrame) -> int:
        """Import feedback items from app store reviews DataFrame."""
        imported_count = 0
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    # Prepare metadata
                    metadata = {
                        "user_name": row.get("user_name", ""),
                        "app_version": row.get("app_version", ""),
                        "date": row.get("date", "")
                    }
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO feedback_items 
                        (id, source_type, content, rating, platform, metadata, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["review_id"],
                        "app_store_review",
                        row["review_text"],
                        row.get("rating"),
                        row.get("platform"),
                        json.dumps(metadata),
                        pd.to_datetime(row.get("date", ""), errors='coerce')
                    ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error importing review {row.get('review_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
        
    def export_data(self, table_name: str, output_path: Optional[Path] = None, format: str = "csv") -> Dict[str, Any]:
        """
        Export data from database to file.
        
        Args:
            table_name: Name of table to export
            output_path: Optional output file path
            format: Export format (csv, json, excel)
            
        Returns:
            Export results dictionary
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"{table_name}_export_{timestamp}.{format}"
            
            # Query data
            with self.get_db_connection() as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            
            if df.empty:
                return {
                    "success": False,
                    "error": f"No data found in table {table_name}",
                    "exported_count": 0
                }
            
            # Export based on format
            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "json":
                df.to_json(output_path, orient="records", indent=2)
            elif format == "excel":
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(df)} records to {output_path}")
            
            return {
                "success": True,
                "exported_count": len(df),
                "output_path": str(output_path),
                "format": format
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {
                "success": False,
                "error": str(e),
                "exported_count": 0
            }
    
    def get_feedback_items(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve feedback items from database.
        
        Args:
            filters: Optional filters to apply
            limit: Optional limit on number of results
            
        Returns:
            List of feedback item dictionaries
        """
        query = "SELECT * FROM feedback_items"
        params = []
        
        # Build WHERE clause from filters
        if filters:
            conditions = []
            
            if "source_type" in filters:
                conditions.append("source_type = ?")
                params.append(filters["source_type"])
            
            if "rating_min" in filters:
                conditions.append("rating >= ?")
                params.append(filters["rating_min"])
            
            if "rating_max" in filters:
                conditions.append("rating <= ?")
                params.append(filters["rating_max"])
            
            if "date_from" in filters:
                conditions.append("timestamp >= ?")
                params.append(filters["date_from"])
            
            if "date_to" in filters:
                conditions.append("timestamp <= ?")
                params.append(filters["date_to"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        # Add ORDER BY and LIMIT
        query += " ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_tickets(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve tickets from database.
        
        Args:
            filters: Optional filters to apply
            limit: Optional limit on number of results
            
        Returns:
            List of ticket dictionaries
        """
        query = "SELECT * FROM tickets"
        params = []
        
        # Build WHERE clause from filters
        if filters:
            conditions = []
            
            if "category" in filters:
                if isinstance(filters["category"], list):
                    placeholders = ",".join(["?" for _ in filters["category"]])
                    conditions.append(f"category IN ({placeholders})")
                    params.extend(filters["category"])
                else:
                    conditions.append("category = ?")
                    params.append(filters["category"])
            
            if "priority" in filters:
                if isinstance(filters["priority"], list):
                    placeholders = ",".join(["?" for _ in filters["priority"]])
                    conditions.append(f"priority IN ({placeholders})")
                    params.extend(filters["priority"])
                else:
                    conditions.append("priority = ?")
                    params.append(filters["priority"])
            
            if "status" in filters:
                conditions.append("status = ?")
                params.append(filters["status"])
            
            if "assigned_team" in filters:
                conditions.append("assigned_team = ?")
                params.append(filters["assigned_team"])
            
            if "confidence_min" in filters:
                conditions.append("agent_confidence >= ?")
                params.append(filters["confidence_min"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        # Add ORDER BY and LIMIT
        query += " ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_processing_logs(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve processing logs from database.
        
        Args:
            filters: Optional filters to apply
            limit: Optional limit on number of results
            
        Returns:
            List of log dictionaries
        """
        query = "SELECT * FROM processing_logs"
        params = []
        
        # Build WHERE clause from filters
        if filters:
            conditions = []
            
            if "agent_name" in filters:
                conditions.append("agent_name = ?")
                params.append(filters["agent_name"])
            
            if "status" in filters:
                conditions.append("status = ?")
                params.append(filters["status"])
            
            if "date_from" in filters:
                conditions.append("timestamp >= ?")
                params.append(filters["date_from"])
            
            if "date_to" in filters:
                conditions.append("timestamp <= ?")
                params.append(filters["date_to"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        # Add ORDER BY and LIMIT
        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_metrics(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve metrics from database.
        
        Args:
            filters: Optional filters to apply
            limit: Optional limit on number of results
            
        Returns:
            List of metric dictionaries
        """
        query = "SELECT * FROM metrics"
        params = []
        
        # Build WHERE clause from filters
        if filters:
            conditions = []
            
            if "metric_name" in filters:
                conditions.append("metric_name = ?")
                params.append(filters["metric_name"])
            
            if "category" in filters:
                conditions.append("category = ?")
                params.append(filters["category"])
            
            if "date_from" in filters:
                conditions.append("timestamp >= ?")
                params.append(filters["date_from"])
            
            if "date_to" in filters:
                conditions.append("timestamp <= ?")
                params.append(filters["date_to"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        # Add ORDER BY and LIMIT
        query += " ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get analytics summary from database.
        
        Returns:
            Analytics summary dictionary
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total counts
            cursor.execute("SELECT COUNT(*) as count FROM feedback_items")
            total_feedback = cursor.fetchone()["count"]
            
            cursor.execute("SELECT COUNT(*) as count FROM tickets")
            total_tickets = cursor.fetchone()["count"]
            
            # Category distribution
            cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM tickets 
                GROUP BY category
            """)
            category_dist = {row["category"]: row["count"] for row in cursor.fetchall()}
            
            # Priority distribution
            cursor.execute("""
                SELECT priority, COUNT(*) as count 
                FROM tickets 
                GROUP BY priority
            """)
            priority_dist = {row["priority"]: row["count"] for row in cursor.fetchall()}
            
            # Average confidence
            cursor.execute("SELECT AVG(agent_confidence) as avg_confidence FROM tickets WHERE agent_confidence IS NOT NULL")
            avg_confidence_result = cursor.fetchone()
            avg_confidence = avg_confidence_result["avg_confidence"] if avg_confidence_result["avg_confidence"] else 0
            
            # Success rate (from processing logs)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful
                FROM processing_logs
            """)
            log_stats = cursor.fetchone()
            success_rate = (log_stats["successful"] / log_stats["total"]) if log_stats["total"] > 0 else 0
            
            # Recent activity
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM processing_logs
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """)
            timeline_data = [{"date": row["date"], "count": row["count"]} for row in cursor.fetchall()]
            
            return {
                "total_feedback": total_feedback,
                "total_tickets": total_tickets,
                "category_distribution": category_dist,
                "priority_distribution": priority_dist,
                "avg_confidence": avg_confidence,
                "success_rate": success_rate,
                "timeline_data": timeline_data
            }
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old data from database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Dictionary with cleanup results
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Clean up old processing logs
            cursor.execute(
                "DELETE FROM processing_logs WHERE timestamp < ?",
                (cutoff_date,)
            )
            logs_deleted = cursor.rowcount
            
            # Clean up old metrics
            cursor.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (cutoff_date,)
            )
            metrics_deleted = cursor.rowcount
            
            conn.commit()
            
            self.logger.info(f"Cleaned up {logs_deleted} old log entries and {metrics_deleted} old metrics")
            
            return {
                "logs_deleted": logs_deleted,
                "metrics_deleted": metrics_deleted,
                "cutoff_date": cutoff_date.isoformat()
            }
    
    def backup_database(self, backup_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Optional path for backup file
            
        Returns:
            Backup results dictionary
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = Path(f"data/backup_feedback_system_{timestamp}.db")
            
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            # Get backup file size
            backup_size = backup_path.stat().st_size
            
            self.logger.info(f"Database backed up to {backup_path}")
            
            return {
                "success": True,
                "backup_path": str(backup_path),
                "backup_size_bytes": backup_size,
                "backup_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.
        
        Returns:
            Database information dictionary
        """
        try:
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get table information
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                tables = [row["name"] for row in cursor.fetchall()]
                
                # Get row counts for each table
                table_counts = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    table_counts[table] = cursor.fetchone()["count"]
                
                return {
                    "database_path": str(self.db_path),
                    "database_size_bytes": db_size,
                    "database_size_mb": round(db_size / (1024 * 1024), 2),
                    "tables": tables,
                    "table_counts": table_counts,
                    "last_modified": datetime.fromtimestamp(self.db_path.stat().st_mtime).isoformat() if self.db_path.exists() else None
                }
                
        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            return {
                "error": str(e)
            }
    
    def _import_feedback_from_emails(self, df: pd.DataFrame) -> int:
        """Import feedback items from support emails DataFrame."""
        imported_count = 0
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    # Combine subject and body
                    content = f"Subject: {row.get('subject', '')}\n\nBody: {row.get('body', '')}"
                    
                    # Prepare metadata
                    metadata = {
                        "subject": row.get("subject", ""),
                        "sender_email": row.get("sender_email", ""),
                        "priority": row.get("priority", "")
                    }
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO feedback_items 
                        (id, source_type, content, metadata, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        row["email_id"],
                        "support_email",
                        content,
                        json.dumps(metadata),
                        pd.to_datetime(row.get("timestamp", ""), errors='coerce')
                    ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error importing email {row.get('email_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
        
        return imported_count
    
    def _import_tickets(self, df: pd.DataFrame) -> int:
        """Import tickets from DataFrame."""
        imported_count = 0
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO tickets 
                        (ticket_id, source_id, source_type, category, priority, title, description,
                         technical_details, agent_confidence, tags, estimated_effort, assigned_team, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["ticket_id"],
                        row["source_id"],
                        row["source_type"],
                        row["category"],
                        row["priority"],
                        row["title"],
                        row["description"],
                        row.get("technical_details", ""),
                        row.get("agent_confidence"),
                        row.get("tags", ""),
                        row.get("estimated_effort", ""),
                        row.get("assigned_team", ""),
                        pd.to_datetime(row.get("created_at", ""), errors='coerce')
                    ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error importing ticket {row.get('ticket_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
        
        return imported_count
    
    def _import_processing_logs(self, df: pd.DataFrame) -> int:
        """Import processing logs from DataFrame."""
        imported_count = 0
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO processing_logs 
                        (timestamp, source_id, agent_name, action, details, confidence_score, duration_ms, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pd.to_datetime(row.get("timestamp", ""), errors='coerce'),
                        row.get("source_id"),
                        row["agent_name"],
                        row["action"],
                        row.get("details", ""),
                        row.get("confidence_score"),
                        row.get("duration_ms"),
                        row.get("status", "success")
                    ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error importing log entry: {e}")
                    continue
            
            conn.commit()
        
        return imported_count
    
    def _import_metrics(self, df: pd.DataFrame) -> int:
        """Import metrics from DataFrame."""
        imported_count = 0
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT INTO metrics 
                        (metric_name, value, timestamp, details, category)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        row["metric_name"],
                        row["value"],
                        pd.to_datetime(row.get("timestamp", ""), errors='coerce'),
                        row.get("details", ""),
                        row.get("category", "")
                    ))
                    
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error importing metric: {e}")
                    continue
            
            conn.commit()
        
        return imported_count