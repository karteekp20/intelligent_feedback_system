"""
Data transformation utilities for the Intelligent Feedback Analysis System.
Handles input/output data transformation and format conversion.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path

from ..core.data_models import FeedbackItem, Ticket, Classification, BugDetails, FeatureDetails
from ..utils.logger import get_logger
from config.settings import CSV_SCHEMAS


class DataTransformer:
    """Handles transformation between different data formats and structures."""
    
    def __init__(self):
        """Initialize data transformer."""
        self.logger = get_logger("data_transformer")
    
    # INPUT TRANSFORMATIONS
    def csv_to_feedback_items(self, csv_path: str, source_type: str) -> List[FeedbackItem]:
        """
        Transform CSV file to FeedbackItem objects.
        
        INPUT: CSV file (app_store_reviews.csv or support_emails.csv)
        OUTPUT: List of FeedbackItem objects
        
        Args:
            csv_path: Path to CSV file
            source_type: Type of source data
            
        Returns:
            List of FeedbackItem objects
        """
        try:
            df = pd.read_csv(csv_path)
            feedback_items = []
            
            if source_type == "app_store_reviews":
                for _, row in df.iterrows():
                    item = FeedbackItem(
                        id=row["review_id"],
                        source_type="app_store_review",
                        content=row["review_text"],
                        rating=int(row["rating"]) if pd.notna(row["rating"]) else None,
                        platform=row.get("platform"),
                        timestamp=pd.to_datetime(row.get("date"), errors='coerce'),
                        metadata={
                            "user_name": row.get("user_name", ""),
                            "app_version": row.get("app_version", ""),
                            "platform": row.get("platform", "")
                        }
                    )
                    feedback_items.append(item)
            
            elif source_type == "support_emails":
                for _, row in df.iterrows():
                    content = f"Subject: {row.get('subject', '')}\n\n{row.get('body', '')}"
                    item = FeedbackItem(
                        id=row["email_id"],
                        source_type="support_email",
                        content=content,
                        timestamp=pd.to_datetime(row.get("timestamp"), errors='coerce'),
                        metadata={
                            "subject": row.get("subject", ""),
                            "sender_email": row.get("sender_email", ""),
                            "priority": row.get("priority", ""),
                            "body": row.get("body", "")
                        }
                    )
                    feedback_items.append(item)
            
            self.logger.info(f"Transformed {len(feedback_items)} items from {csv_path}")
            return feedback_items
            
        except Exception as e:
            self.logger.error(f"Error transforming CSV to feedback items: {e}")
            return []
    
    def json_to_feedback_items(self, json_data: Union[str, Dict, List]) -> List[FeedbackItem]:
        """
        Transform JSON data to FeedbackItem objects.
        
        INPUT: JSON string/dict/list with feedback data
        OUTPUT: List of FeedbackItem objects
        """
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            
            if isinstance(data, dict):
                data = [data]  # Single item
            
            feedback_items = []
            for item_data in data:
                item = FeedbackItem(
                    id=item_data["id"],
                    source_type=item_data["source_type"],
                    content=item_data["content"],
                    rating=item_data.get("rating"),
                    platform=item_data.get("platform"),
                    timestamp=pd.to_datetime(item_data.get("timestamp"), errors='coerce'),
                    metadata=item_data.get("metadata", {})
                )
                feedback_items.append(item)
            
            return feedback_items
            
        except Exception as e:
            self.logger.error(f"Error transforming JSON to feedback items: {e}")
            return []
    
    def api_response_to_feedback_items(self, api_response: Dict[str, Any]) -> List[FeedbackItem]:
        """
        Transform API response to FeedbackItem objects.
        
        INPUT: API response dictionary
        OUTPUT: List of FeedbackItem objects
        """
        try:
            items_data = api_response.get("items", [])
            feedback_items = []
            
            for item_data in items_data:
                item = FeedbackItem(
                    id=item_data["id"],
                    source_type=item_data.get("source", "api"),
                    content=item_data["content"],
                    rating=item_data.get("rating"),
                    platform=item_data.get("platform"),
                    timestamp=datetime.utcnow(),
                    metadata=item_data.get("metadata", {})
                )
                feedback_items.append(item)
            
            return feedback_items
            
        except Exception as e:
            self.logger.error(f"Error transforming API response: {e}")
            return []
    
    # OUTPUT TRANSFORMATIONS
    def tickets_to_csv(self, tickets: List[Ticket], output_path: str) -> bool:
        """
        Transform Ticket objects to CSV file.
        
        INPUT: List of Ticket objects
        OUTPUT: CSV file (generated_tickets.csv)
        """
        try:
            ticket_dicts = [ticket.to_dict() for ticket in tickets]
            df = pd.DataFrame(ticket_dicts)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Exported {len(tickets)} tickets to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting tickets to CSV: {e}")
            return False
    
    def tickets_to_json(self, tickets: List[Ticket], output_path: str) -> bool:
        """
        Transform Ticket objects to JSON file.
        
        INPUT: List of Ticket objects  
        OUTPUT: JSON file
        """
        try:
            ticket_dicts = [ticket.to_dict() for ticket in tickets]
            
            with open(output_path, 'w') as f:
                json.dump(ticket_dicts, f, indent=2, default=str)
            
            self.logger.info(f"Exported {len(tickets)} tickets to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting tickets to JSON: {e}")
            return False
    
    def tickets_to_excel(self, tickets: List[Ticket], output_path: str) -> bool:
        """
        Transform Ticket objects to Excel file.
        
        INPUT: List of Ticket objects
        OUTPUT: Excel file (.xlsx)
        """
        try:
            ticket_dicts = [ticket.to_dict() for ticket in tickets]
            df = pd.DataFrame(ticket_dicts)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Tickets', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Tickets', 'Bugs', 'Features', 'Complaints', 'Praise', 'Spam'],
                    'Count': [
                        len(tickets),
                        len([t for t in tickets if t.category.value == 'Bug']),
                        len([t for t in tickets if t.category.value == 'Feature Request']),
                        len([t for t in tickets if t.category.value == 'Complaint']),
                        len([t for t in tickets if t.category.value == 'Praise']),
                        len([t for t in tickets if t.category.value == 'Spam'])
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            self.logger.info(f"Exported {len(tickets)} tickets to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting tickets to Excel: {e}")
            return False
    
    def analytics_to_json(self, analytics_data: Dict[str, Any], output_path: str) -> bool:
        """
        Transform analytics data to JSON file.
        
        INPUT: Analytics dictionary
        OUTPUT: JSON file with analytics
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported analytics to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting analytics: {e}")
            return False
    
    def metrics_to_csv(self, metrics: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Transform metrics to CSV file.
        
        INPUT: List of metric dictionaries
        OUTPUT: CSV file (metrics.csv)
        """
        try:
            df = pd.DataFrame(metrics)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Exported {len(metrics)} metrics to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False
    
    # BATCH TRANSFORMATIONS
    def batch_transform_csv_inputs(self, input_dir: str) -> Dict[str, List[FeedbackItem]]:
        """
        Batch transform all CSV files in input directory.
        
        INPUT: Directory containing CSV files
        OUTPUT: Dictionary mapping file types to FeedbackItem lists
        """
        input_path = Path(input_dir)
        results = {}
        
        # Process app store reviews
        reviews_file = input_path / "app_store_reviews.csv"
        if reviews_file.exists():
            results["app_store_reviews"] = self.csv_to_feedback_items(
                str(reviews_file), "app_store_reviews"
            )
        
        # Process support emails
        emails_file = input_path / "support_emails.csv"
        if emails_file.exists():
            results["support_emails"] = self.csv_to_feedback_items(
                str(emails_file), "support_emails"
            )
        
        return results
    
    def batch_export_outputs(self, tickets: List[Ticket], analytics: Dict[str, Any], 
                           metrics: List[Dict[str, Any]], output_dir: str) -> Dict[str, bool]:
        """
        Batch export all outputs to specified directory.
        
        INPUT: Tickets, analytics, metrics
        OUTPUT: Multiple files in output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Export tickets in multiple formats
        results["tickets_csv"] = self.tickets_to_csv(
            tickets, str(output_path / "generated_tickets.csv")
        )
        results["tickets_json"] = self.tickets_to_json(
            tickets, str(output_path / "generated_tickets.json")
        )
        results["tickets_excel"] = self.tickets_to_excel(
            tickets, str(output_path / "generated_tickets.xlsx")
        )
        
        # Export analytics
        results["analytics_json"] = self.analytics_to_json(
            analytics, str(output_path / "analytics_summary.json")
        )
        
        # Export metrics
        results["metrics_csv"] = self.metrics_to_csv(
            metrics, str(output_path / "metrics.csv")
        )
        
        return results
    
    # DATA FORMAT CONVERSIONS
    def feedback_item_to_dict(self, item: FeedbackItem) -> Dict[str, Any]:
        """Convert FeedbackItem to dictionary."""
        return {
            "id": item.id,
            "source_type": item.source_type,
            "content": item.content,
            "rating": item.rating,
            "platform": item.platform,
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
            "metadata": item.metadata
        }
    
    def dict_to_feedback_item(self, data: Dict[str, Any]) -> FeedbackItem:
        """Convert dictionary to FeedbackItem."""
        return FeedbackItem(
            id=data["id"],
            source_type=data["source_type"],
            content=data["content"],
            rating=data.get("rating"),
            platform=data.get("platform"),
            timestamp=pd.to_datetime(data.get("timestamp"), errors='coerce'),
            metadata=data.get("metadata", {})
        )
    
    def normalize_data_formats(self, data: Any) -> Dict[str, Any]:
        """
        Normalize various input data formats to standard dictionary.
        
        INPUT: Any data format (CSV row, JSON, API response)
        OUTPUT: Standardized dictionary
        """
        if isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, dict):
            return data
        elif hasattr(data, '__dict__'):
            return data.__dict__
        else:
            return {"raw_data": str(data)}
    
    def validate_input_format(self, data: Any, expected_format: str) -> bool:
        """
        Validate that input data matches expected format.
        
        Args:
            data: Input data to validate
            expected_format: Expected format (csv, json, api_response)
            
        Returns:
            True if format is valid
        """
        try:
            if expected_format == "csv":
                return isinstance(data, (pd.DataFrame, str, Path))
            elif expected_format == "json":
                return isinstance(data, (str, dict, list))
            elif expected_format == "api_response":
                return isinstance(data, dict) and "items" in data
            else:
                return False
        except Exception:
            return False
    
    def get_supported_input_formats(self) -> List[str]:
        """Get list of supported input formats."""
        return ["csv", "json", "api_response", "dataframe"]
    
    def get_supported_output_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return ["csv", "json", "excel", "xml", "pdf_report"]


# Data Schema Definitions for Input/Output
class DataSchemas:
    """Defines input and output data schemas."""
    
    INPUT_SCHEMAS = {
        "app_store_reviews": {
            "required_fields": ["review_id", "platform", "rating", "review_text"],
            "optional_fields": ["user_name", "date", "app_version"],
            "field_types": {
                "review_id": "string",
                "platform": "string", 
                "rating": "integer",
                "review_text": "string",
                "user_name": "string",
                "date": "string",
                "app_version": "string"
            }
        },
        "support_emails": {
            "required_fields": ["email_id", "subject", "body", "sender_email"],
            "optional_fields": ["timestamp", "priority"],
            "field_types": {
                "email_id": "string",
                "subject": "string",
                "body": "string", 
                "sender_email": "string",
                "timestamp": "string",
                "priority": "string"
            }
        }
    }
    
    OUTPUT_SCHEMAS = {
        "generated_tickets": {
            "fields": [
                "ticket_id", "source_id", "source_type", "category", "priority",
                "title", "description", "technical_details", "created_at",
                "agent_confidence", "tags", "estimated_effort", "assigned_team"
            ],
            "field_types": {
                "ticket_id": "string",
                "source_id": "string",
                "source_type": "string",
                "category": "string",
                "priority": "string",
                "title": "string",
                "description": "string",
                "technical_details": "string",
                "created_at": "datetime",
                "agent_confidence": "float",
                "tags": "string",
                "estimated_effort": "string",
                "assigned_team": "string"
            }
        },
        "processing_log": {
            "fields": [
                "timestamp", "source_id", "agent_name", "action", "details",
                "confidence_score", "duration_ms", "status"
            ]
        },
        "metrics": {
            "fields": [
                "metric_name", "value", "timestamp", "details", "category"
            ]
        }
    }