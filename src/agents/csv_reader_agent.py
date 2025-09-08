"""
CSV Reader Agent for reading and parsing feedback data from CSV files.
Handles multiple file formats and converts raw data into structured FeedbackItem objects.
"""

import pandas as pd
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from .base_agent import BaseAgent
from src.core.data_models import AgentResult, FeedbackItem, FeedbackCategory, FeedbackSource
from src.utils.csv_handler import CSVHandler
from src.utils.validators import DataValidator
from src.utils.logger import get_logger

# File path configurations
FILE_PATHS = {
    "app_store_reviews": "data/input/app_store_reviews.csv",
    "support_emails": "data/input/support_emails.csv",
    "expected_classifications": "data/input/expected_classifications.csv"
}

# CSV Schema definitions
CSV_SCHEMAS = {
    "app_store_reviews": {
        "required_columns": ["review_id", "user_id", "rating", "review_text", "date"],
        "optional_columns": ["app_version", "device_type", "country"]
    },
    "support_emails": {
        "required_columns": ["email_id", "subject", "body", "sender_email", "received_date"],
        "optional_columns": ["priority", "category", "status"]
    }
}


class CSVReaderAgent(BaseAgent):
    """Agent responsible for reading and parsing CSV files containing user feedback."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CSV Reader Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("csv_reader", config)
        self.csv_handler = CSVHandler()
        self.validator = DataValidator()
        self.supported_file_types = ["app_store_reviews", "support_emails"]
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.batch_size = config.get("batch_size", 1000) if config else 1000
        self.validate_data = config.get("validate_data", True) if config else True
        self.handle_encoding_errors = config.get("handle_encoding_errors", True) if config else True
        
    async def process(self, data: Any) -> AgentResult:
        """
        Process CSV files and return parsed feedback items.
        
        Args:
            data: Dictionary containing file paths or file type to read
                  Can be:
                  - str: file type ("app_store_reviews", "support_emails", "all")
                  - dict: {"file_paths": [...], "file_type": "..."}
                  - dict: {"file_type": "..."}
                  - None: read all supported files
                  
        Returns:
            AgentResult containing list of FeedbackItem objects
        """
        try:
            self.logger.info("Starting CSV reading process")
            start_time = datetime.now()
            
            # Determine files to process
            file_paths = self._determine_file_paths(data)
            
            if not file_paths:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="No valid CSV files found to process"
                )
            
            # Process files
            all_feedback_items = []
            processing_stats = {
                "total_files": len(file_paths), 
                "successful_files": 0, 
                "total_items": 0,
                "valid_items": 0,
                "validation_errors": 0
            }
            
            for file_info in file_paths:
                try:
                    feedback_items = await self._read_csv_file(file_info)
                    all_feedback_items.extend(feedback_items)
                    processing_stats["successful_files"] += 1
                    processing_stats["total_items"] += len(feedback_items)
                    
                    self.logger.info(f"Successfully read {len(feedback_items)} items from {file_info['path']}")
                    
                except Exception as e:
                    self.logger.error(f"Error reading file {file_info['path']}: {str(e)}")
                    continue
            
            if not all_feedback_items:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="No feedback items could be parsed from CSV files"
                )
            
            # Validate parsed items if enabled
            if self.validate_data:
                valid_items, validation_errors = await self._validate_feedback_items(all_feedback_items)
                processing_stats["valid_items"] = len(valid_items)
                processing_stats["validation_errors"] = len(validation_errors)
                
                if validation_errors:
                    self.logger.warning(f"Found {len(validation_errors)} validation errors")
                    for error in validation_errors[:5]:  # Log first 5 errors
                        self.logger.warning(error)
            else:
                valid_items = all_feedback_items
                processing_stats["valid_items"] = len(valid_items)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create detailed result
            result_details = {
                "processing_stats": processing_stats,
                "processing_time_seconds": processing_time,
                "items_per_second": len(valid_items) / processing_time if processing_time > 0 else 0
            }
            
            self.logger.info(f"CSV reading completed successfully. Processed {processing_stats['total_items']} items "
                           f"from {processing_stats['successful_files']} files in {processing_time:.2f} seconds")
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=valid_items,
                confidence=1.0,  # CSV reading is deterministic
                details=json.dumps(result_details),
                metadata={
                    "processing_stats": processing_stats,
                    "file_paths": [fp["path"] for fp in file_paths]
                }
            )
            
        except Exception as e:
            self.logger.error(f"CSV reading failed: {str(e)}", exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"CSV reading failed: {str(e)}"
            )
    
    def _determine_file_paths(self, data: Any) -> List[Dict[str, str]]:
        """
        Determine which files to process based on input data.
        
        Args:
            data: Input data specification
            
        Returns:
            List of file info dictionaries with 'path' and 'type' keys
        """
        file_paths = []
        
        if isinstance(data, str):
            # Single file type specified
            if data == "all":
                for file_type in self.supported_file_types:
                    path = FILE_PATHS.get(file_type)
                    if path and Path(path).exists():
                        file_paths.append({"path": path, "type": file_type})
            else:
                path = FILE_PATHS.get(data)
                if path and Path(path).exists():
                    file_paths.append({"path": path, "type": data})
                    
        elif isinstance(data, dict):
            if "file_paths" in data:
                # Explicit file paths provided
                for path in data["file_paths"]:
                    if Path(path).exists():
                        # Try to determine file type from path
                        file_type = self._determine_file_type_from_path(path)
                        file_paths.append({"path": path, "type": file_type})
                        
            elif "file_type" in data:
                # File type specified
                file_type = data["file_type"]
                if file_type == "all":
                    for ft in self.supported_file_types:
                        path = FILE_PATHS.get(ft)
                        if path and Path(path).exists():
                            file_paths.append({"path": path, "type": ft})
                else:
                    path = FILE_PATHS.get(file_type)
                    if path and Path(path).exists():
                        file_paths.append({"path": path, "type": file_type})
        else:
            # Default: read all supported files
            for file_type in self.supported_file_types:
                path = FILE_PATHS.get(file_type)
                if path and Path(path).exists():
                    file_paths.append({"path": path, "type": file_type})
        
        return file_paths
    
    def _determine_file_type_from_path(self, file_path: str) -> str:
        """Determine file type from file path."""
        path_lower = file_path.lower()
        if "review" in path_lower or "app_store" in path_lower:
            return "app_store_reviews"
        elif "email" in path_lower or "support" in path_lower:
            return "support_emails"
        else:
            return "unknown"
    
    async def _read_csv_file(self, file_info: Dict[str, str]) -> List[FeedbackItem]:
        """
        Read a single CSV file and convert to FeedbackItem objects.
        
        Args:
            file_info: Dictionary with 'path' and 'type' keys
            
        Returns:
            List of FeedbackItem objects
        """
        file_path = file_info["path"]
        file_type = file_info["type"]
        
        try:
            # Read CSV with proper encoding handling
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read file {file_path} with any encoding")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Validate schema
            self._validate_csv_schema(df, file_type)
            
            # Convert to FeedbackItem objects
            feedback_items = []
            
            for _, row in df.iterrows():
                try:
                    if file_type == "app_store_reviews":
                        feedback_item = self._create_feedback_from_review(row)
                    elif file_type == "support_emails":
                        feedback_item = self._create_feedback_from_email(row)
                    else:
                        feedback_item = self._create_generic_feedback(row)
                    
                    if feedback_item:
                        feedback_items.append(feedback_item)
                        
                except Exception as e:
                    self.logger.warning(f"Error parsing row: {str(e)}")
                    continue
            
            return feedback_items
            
        except Exception as e:
            self.logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            raise
    
    def _validate_csv_schema(self, df: pd.DataFrame, file_type: str) -> None:
        """
        Validate CSV schema against expected format.
        
        Args:
            df: DataFrame to validate
            file_type: Type of file being validated
        """
        if file_type not in CSV_SCHEMAS:
            self.logger.warning(f"No schema defined for file type: {file_type}")
            return
        
        schema = CSV_SCHEMAS[file_type]
        required_columns = schema["required_columns"]
        
        # Check for required columns
        missing_columns = []
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Log optional columns that are missing
        optional_columns = schema.get("optional_columns", [])
        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            self.logger.info(f"Optional columns not found: {missing_optional}")
    
    def _create_feedback_from_review(self, row: pd.Series) -> Optional[FeedbackItem]:
        """Create FeedbackItem from app store review row."""
        try:
            return FeedbackItem(
                id=str(row.get("review_id", "")),
                content=str(row.get("review_text", "")),
                source=FeedbackSource.APP_STORE,
                timestamp=pd.to_datetime(row.get("date", datetime.now())),
                metadata={
                    "user_id": str(row.get("user_id", "")),
                    "rating": int(row.get("rating", 0)) if pd.notna(row.get("rating")) else None,
                    "app_version": str(row.get("app_version", "")) if pd.notna(row.get("app_version")) else None,
                    "device_type": str(row.get("device_type", "")) if pd.notna(row.get("device_type")) else None,
                    "country": str(row.get("country", "")) if pd.notna(row.get("country")) else None
                }
            )
        except Exception as e:
            self.logger.warning(f"Error creating feedback from review: {str(e)}")
            return None
    
    def _create_feedback_from_email(self, row: pd.Series) -> Optional[FeedbackItem]:
        """Create FeedbackItem from support email row."""
        try:
            # Combine subject and body for content
            subject = str(row.get("subject", ""))
            body = str(row.get("body", ""))
            content = f"{subject}\n\n{body}" if subject and body else (subject or body)
            
            return FeedbackItem(
                id=str(row.get("email_id", "")),
                content=content,
                source=FeedbackSource.SUPPORT_EMAIL,
                timestamp=pd.to_datetime(row.get("received_date", datetime.now())),
                metadata={
                    "sender_email": str(row.get("sender_email", "")),
                    "subject": subject,
                    "priority": str(row.get("priority", "")) if pd.notna(row.get("priority")) else None,
                    "category": str(row.get("category", "")) if pd.notna(row.get("category")) else None,
                    "status": str(row.get("status", "")) if pd.notna(row.get("status")) else None
                }
            )
        except Exception as e:
            self.logger.warning(f"Error creating feedback from email: {str(e)}")
            return None
    
    def _create_generic_feedback(self, row: pd.Series) -> Optional[FeedbackItem]:
        """Create FeedbackItem from generic row."""
        try:
            # Try to find content column
            content_columns = ["content", "text", "message", "feedback", "body"]
            content = ""
            
            for col in content_columns:
                if col in row and pd.notna(row[col]):
                    content = str(row[col])
                    break
            
            if not content:
                self.logger.warning("No content found in row")
                return None
            
            # Try to find ID column
            id_columns = ["id", "feedback_id", "message_id"]
            feedback_id = ""
            
            for col in id_columns:
                if col in row and pd.notna(row[col]):
                    feedback_id = str(row[col])
                    break
            
            if not feedback_id:
                feedback_id = str(hash(content))  # Generate ID from content
            
            return FeedbackItem(
                id=feedback_id,
                content=content,
                source=FeedbackSource.OTHER,
                timestamp=datetime.now(),
                metadata={key: str(value) for key, value in row.items() if pd.notna(value)}
            )
        except Exception as e:
            self.logger.warning(f"Error creating generic feedback: {str(e)}")
            return None
    
    async def _validate_feedback_items(self, feedback_items: List[FeedbackItem]) -> Tuple[List[FeedbackItem], List[str]]:
        """
        Validate feedback items and return valid items and error messages.
        
        Args:
            feedback_items: List of feedback items to validate
            
        Returns:
            Tuple of (valid_items, error_messages)
        """
        valid_items = []
        errors = []
        
        for i, item in enumerate(feedback_items):
            try:
                # Basic validation
                if not item.id:
                    errors.append(f"Item {i}: Missing ID")
                    continue
                
                if not item.content or len(item.content.strip()) == 0:
                    errors.append(f"Item {i}: Empty content")
                    continue
                
                if len(item.content) > 10000:  # Reasonable content length limit
                    errors.append(f"Item {i}: Content too long ({len(item.content)} characters)")
                    continue
                
                # Additional validation using validator
                if self.validator.validate_feedback_item(item):
                    valid_items.append(item)
                else:
                    errors.append(f"Item {i}: Failed validation")
                    
            except Exception as e:
                errors.append(f"Item {i}: Validation error - {str(e)}")
        
        return valid_items, errors
    
    async def get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """
        Get statistics about a CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary with file statistics
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return {"error": "File not found"}
            
            # Basic file info
            stats = {
                "file_size_bytes": file_path_obj.stat().st_size,
                "file_size_mb": file_path_obj.stat().st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(file_path_obj.stat().st_mtime)
            }
            
            # Read file for content stats
            df = pd.read_csv(file_path)
            stats.update({
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": df.columns.tolist(),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
            })
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
