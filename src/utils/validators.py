"""
Data validation utilities for the Intelligent Feedback Analysis System.
"""

import re
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import pandas as pd
from email.utils import parseaddr

from .logger import get_logger


class DataValidator:
    """Comprehensive data validation utilities."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = get_logger("data_validator")
        
        # Common validation patterns
        self.patterns = {
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            "version": re.compile(r'^\d+\.\d+(?:\.\d+)?(?:\.\d+)?$'),
            "ticket_id": re.compile(r'^[A-Z]+-\d{8}-[A-Z0-9]+$'),
            "review_id": re.compile(r'^review_\d+$'),
            "email_id": re.compile(r'^email_\d+$'),
            "url": re.compile(r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$')
        }
        
        # Validation rules for different data types
        self.validation_rules = {
            "feedback_item": {
                "required_fields": ["id", "source_type", "content"],
                "field_types": {
                    "id": str,
                    "source_type": str,
                    "content": str,
                    "rating": int,
                    "platform": str,
                    "timestamp": (str, datetime)
                },
                "field_constraints": {
                    "rating": {"min": 1, "max": 5},
                    "source_type": {"values": ["app_store_review", "support_email"]},
                    "content": {"min_length": 1, "max_length": 10000}
                }
            },
            "ticket": {
                "required_fields": ["ticket_id", "source_id", "category", "priority", "title", "description"],
                "field_types": {
                    "ticket_id": str,
                    "source_id": str,
                    "category": str,
                    "priority": str,
                    "title": str,
                    "description": str,
                    "agent_confidence": float
                },
                "field_constraints": {
                    "category": {"values": ["Bug", "Feature Request", "Praise", "Complaint", "Spam"]},
                    "priority": {"values": ["Critical", "High", "Medium", "Low"]},
                    "agent_confidence": {"min": 0.0, "max": 1.0},
                    "title": {"min_length": 5, "max_length": 200},
                    "description": {"min_length": 10, "max_length": 5000}
                }
            },
            "app_store_review": {
                "required_fields": ["review_id", "platform", "rating", "review_text"],
                "field_types": {
                    "review_id": str,
                    "platform": str,
                    "rating": int,
                    "review_text": str,
                    "user_name": str,
                    "date": str,
                    "app_version": str
                },
                "field_constraints": {
                    "platform": {"values": ["Google Play", "App Store"]},
                    "rating": {"min": 1, "max": 5},
                    "review_text": {"min_length": 1, "max_length": 2000}
                }
            },
            "support_email": {
                "required_fields": ["email_id", "subject", "body", "sender_email"],
                "field_types": {
                    "email_id": str,
                    "subject": str,
                    "body": str,
                    "sender_email": str,
                    "timestamp": str,
                    "priority": str
                },
                "field_constraints": {
                    "priority": {"values": ["Critical", "High", "Medium", "Low"]},
                    "subject": {"min_length": 1, "max_length": 500},
                    "body": {"min_length": 1, "max_length": 10000}
                }
            }
        }
    
    def validate_data(self, data: Any, data_type: str) -> Dict[str, Any]:
        """
        Validate data against specified type rules.
        
        Args:
            data: Data to validate (dict, DataFrame, or object)
            data_type: Type of data to validate against
            
        Returns:
            Validation results dictionary
        """
        if data_type not in self.validation_rules:
            return {
                "valid": False,
                "errors": [f"Unknown data type: {data_type}"],
                "warnings": []
            }
        
        rules = self.validation_rules[data_type]
        
        if isinstance(data, pd.DataFrame):
            return self._validate_dataframe(data, rules)
        elif isinstance(data, dict):
            return self._validate_dict(data, rules)
        elif hasattr(data, '__dict__'):
            return self._validate_object(data, rules)
        else:
            return {
                "valid": False,
                "errors": [f"Unsupported data format: {type(data)}"],
                "warnings": []
            }
    
    def _validate_dataframe(self, df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DataFrame against rules."""
        errors = []
        warnings = []
        
        # Check required columns
        required_fields = rules.get("required_fields", [])
        missing_columns = [field for field in required_fields if field not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check extra columns
        extra_columns = [col for col in df.columns if col not in rules.get("field_types", {})]
        if extra_columns:
            warnings.append(f"Extra columns found: {extra_columns}")
        
        # Validate each row
        row_errors = []
        for idx, row in df.iterrows():
            row_validation = self._validate_dict(row.to_dict(), rules)
            if not row_validation["valid"]:
                row_errors.append(f"Row {idx}: {'; '.join(row_validation['errors'])}")
        
        if row_errors:
            errors.extend(row_errors[:10])  # Limit to first 10 row errors
            if len(row_errors) > 10:
                warnings.append(f"Additional {len(row_errors) - 10} row validation errors not shown")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "rows_validated": len(df),
            "rows_with_errors": len(row_errors)
        }
    
    def _validate_dict(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dictionary against rules."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
                errors.append(f"Required field is empty: {field}")
        
        # Check field types and constraints
        field_types = rules.get("field_types", {})
        field_constraints = rules.get("field_constraints", {})
        
        for field, value in data.items():
            if field in field_types:
                # Type validation
                expected_type = field_types[field]
                if not self._check_type(value, expected_type):
                    errors.append(f"Field {field} has wrong type: expected {expected_type}, got {type(value)}")
                
                # Constraint validation
                if field in field_constraints:
                    constraint_errors = self._validate_constraints(field, value, field_constraints[field])
                    errors.extend(constraint_errors)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_object(self, obj: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate object against rules."""
        # Convert object to dict and validate
        if hasattr(obj, '__dict__'):
            data_dict = obj.__dict__
        else:
            data_dict = {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('_')}
        
        return self._validate_dict(data_dict, rules)
    
    def _check_type(self, value: Any, expected_type: Union[type, tuple]) -> bool:
        """Check if value matches expected type."""
        if value is None:
            return True  # Allow None values
        
        if isinstance(expected_type, tuple):
            return any(isinstance(value, t) for t in expected_type)
        else:
            return isinstance(value, expected_type)
    
    def _validate_constraints(self, field: str, value: Any, constraints: Dict[str, Any]) -> List[str]:
        """Validate field constraints."""
        errors = []
        
        if value is None:
            return errors  # Skip constraint validation for None values
        
        # Numeric constraints
        if "min" in constraints and isinstance(value, (int, float)):
            if value < constraints["min"]:
                errors.append(f"Field {field} value {value} is below minimum {constraints['min']}")
        
        if "max" in constraints and isinstance(value, (int, float)):
            if value > constraints["max"]:
                errors.append(f"Field {field} value {value} is above maximum {constraints['max']}")
        
        # String constraints
        if "min_length" in constraints and isinstance(value, str):
            if len(value) < constraints["min_length"]:
                errors.append(f"Field {field} is too short: {len(value)} < {constraints['min_length']}")
        
        if "max_length" in constraints and isinstance(value, str):
            if len(value) > constraints["max_length"]:
                errors.append(f"Field {field} is too long: {len(value)} > {constraints['max_length']}")
        
        # Value constraints
        if "values" in constraints:
            allowed_values = constraints["values"]
            if value not in allowed_values:
                errors.append(f"Field {field} has invalid value: {value} not in {allowed_values}")
        
        # Pattern constraints
        if "pattern" in constraints and isinstance(value, str):
            pattern = constraints["pattern"]
            if isinstance(pattern, str):
                pattern = re.compile(pattern)
            if not pattern.match(value):
                errors.append(f"Field {field} doesn't match required pattern")
        
        return errors
    
    def validate_email(self, email: str) -> bool:
        """Validate email address format."""
        if not email or not isinstance(email, str):
            return False
        
        # Use email.utils.parseaddr for better validation
        parsed = parseaddr(email)
        return bool(parsed[1]) and self.patterns["email"].match(parsed[1])
    
    def validate_version(self, version: str) -> bool:
        """Validate version string format."""
        if not version or not isinstance(version, str):
            return False
        
        return bool(self.patterns["version"].match(version))
    
    def validate_rating(self, rating: Any) -> bool:
        """Validate rating value."""
        try:
            rating_int = int(rating)
            return 1 <= rating_int <= 5
        except (ValueError, TypeError):
            return False
    
    def validate_date(self, date_str: str, formats: Optional[List[str]] = None) -> bool:
        """Validate date string format."""
        if not date_str or not isinstance(date_str, str):
            return False
        
        if formats is None:
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%m/%d/%Y",
                "%d/%m/%Y"
            ]
        
        for fmt in formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        
        return False
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format."""
        if not url or not isinstance(url, str):
            return False
        
        return bool(self.patterns["url"].match(url))
    
    def validate_csv_file(self, file_path: str, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate CSV file structure and content.
        
        Args:
            file_path: Path to CSV file
            expected_schema: Expected schema definition
            
        Returns:
            Validation results
        """
        try:
            # Read CSV file
            df = pd.read_csv(file_path, nrows=100)  # Sample first 100 rows for validation
            
            errors = []
            warnings = []
            
            # Check required columns
            required_columns = expected_schema.get("required_columns", [])
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            column_types = expected_schema.get("column_types", {})
            for column, expected_type in column_types.items():
                if column in df.columns:
                    actual_type = df[column].dtype
                    if not self._is_compatible_dtype(actual_type, expected_type):
                        warnings.append(f"Column {column} type mismatch: expected {expected_type}, got {actual_type}")
            
            # Check for empty values in required columns
            for column in required_columns:
                if column in df.columns:
                    null_count = df[column].isnull().sum()
                    empty_count = (df[column] == "").sum()
                    total_empty = null_count + empty_count
                    
                    if total_empty > 0:
                        warnings.append(f"Column {column} has {total_empty} empty values")
            
            # Check unique constraints
            unique_columns = expected_schema.get("unique_columns", [])
            for column in unique_columns:
                if column in df.columns:
                    duplicate_count = df[column].duplicated().sum()
                    if duplicate_count > 0:
                        warnings.append(f"Column {column} has {duplicate_count} duplicate values")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "rows_checked": len(df),
                "columns_found": list(df.columns)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Error reading CSV file: {str(e)}"],
                "warnings": []
            }
    
    def _is_compatible_dtype(self, actual_dtype: str, expected_type: str) -> bool:
        """Check if pandas dtype is compatible with expected type."""
        dtype_str = str(actual_dtype)
        
        compatibility_map = {
            "string": ["object", "string"],
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32", "int64", "int32"],  # int can be converted to float
            "datetime": ["datetime64", "object"],  # object might contain datetime strings
            "bool": ["bool", "object"]
        }
        
        compatible_types = compatibility_map.get(expected_type, [expected_type])
        return any(compatible in dtype_str for compatible in compatible_types)
    
    def generate_validation_report(self, validation_results: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: List of validation result dictionaries
            
        Returns:
            Formatted validation report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DATA VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_validations = len(validation_results)
        valid_count = sum(1 for r in validation_results if r.get("valid", False))
        
        report_lines.append(f"Total validations: {total_validations}")
        report_lines.append(f"Valid: {valid_count}")
        report_lines.append(f"Invalid: {total_validations - valid_count}")
        report_lines.append(f"Success rate: {valid_count/total_validations:.1%}" if total_validations > 0 else "Success rate: N/A")
        report_lines.append("")
        
        # Detail each validation
        for i, result in enumerate(validation_results, 1):
            report_lines.append(f"Validation {i}:")
            report_lines.append(f"  Status: {'✓ VALID' if result.get('valid', False) else '✗ INVALID'}")
            
            if result.get("errors"):
                report_lines.append("  Errors:")
                for error in result["errors"]:
                    report_lines.append(f"    - {error}")
            
            if result.get("warnings"):
                report_lines.append("  Warnings:")
                for warning in result["warnings"]:
                    report_lines.append(f"    - {warning}")
            
            # Add extra info if available
            extra_info = []
            if "rows_validated" in result:
                extra_info.append(f"Rows: {result['rows_validated']}")
            if "rows_with_errors" in result:
                extra_info.append(f"Errors: {result['rows_with_errors']}")
            if "columns_found" in result:
                extra_info.append(f"Columns: {len(result['columns_found'])}")
            
            if extra_info:
                report_lines.append(f"  Info: {', '.join(extra_info)}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def create_validation_schema(self, sample_data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        Create a validation schema from sample data.
        
        Args:
            sample_data: Sample DataFrame to analyze
            data_type: Type of data
            
        Returns:
            Generated validation schema
        """
        schema = {
            "required_columns": list(sample_data.columns),
            "column_types": {},
            "unique_columns": [],
            "constraints": {}
        }
        
        for column in sample_data.columns:
            # Determine column type
            dtype = sample_data[column].dtype
            if "int" in str(dtype):
                schema["column_types"][column] = "int"
            elif "float" in str(dtype):
                schema["column_types"][column] = "float"
            elif "datetime" in str(dtype):
                schema["column_types"][column] = "datetime"
            elif "bool" in str(dtype):
                schema["column_types"][column] = "bool"
            else:
                schema["column_types"][column] = "string"
            
            # Check for unique values
            if sample_data[column].nunique() == len(sample_data):
                schema["unique_columns"].append(column)
            
            # Generate constraints
            if column not in schema["constraints"]:
                schema["constraints"][column] = {}
            
            # Numeric constraints
            if schema["column_types"][column] in ["int", "float"]:
                min_val = sample_data[column].min()
                max_val = sample_data[column].max()
                schema["constraints"][column]["min"] = min_val
                schema["constraints"][column]["max"] = max_val
            
            # String constraints
            elif schema["column_types"][column] == "string":
                string_lengths = sample_data[column].astype(str).str.len()
                min_length = string_lengths.min()
                max_length = string_lengths.max()
                schema["constraints"][column]["min_length"] = min_length
                schema["constraints"][column]["max_length"] = max_length
                
                # Check for common patterns
                if column.lower() == "email" or "email" in column.lower():
                    schema["constraints"][column]["pattern"] = self.patterns["email"]
                elif "version" in column.lower():
                    schema["constraints"][column]["pattern"] = self.patterns["version"]
        
        return schema
    
    def validate_business_rules(self, data: Dict[str, Any], rules: List[Callable]) -> Dict[str, Any]:
        """
        Validate custom business rules.
        
        Args:
            data: Data to validate
            rules: List of validation functions
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        for rule in rules:
            try:
                result = rule(data)
                if isinstance(result, dict):
                    if result.get("valid", True) is False:
                        errors.append(result.get("message", "Business rule validation failed"))
                    if result.get("warnings"):
                        warnings.extend(result["warnings"])
                elif result is False:
                    errors.append(f"Business rule failed: {rule.__name__}")
            except Exception as e:
                errors.append(f"Error in business rule {rule.__name__}: {str(e)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def sanitize_data(self, data: Dict[str, Any], sanitization_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data according to specified rules.
        
        Args:
            data: Data to sanitize
            sanitization_rules: Rules for sanitization
            
        Returns:
            Sanitized data dictionary
        """
        sanitized = data.copy()
        
        for field, rules in sanitization_rules.items():
            if field in sanitized:
                value = sanitized[field]
                
                # Strip whitespace
                if rules.get("strip", False) and isinstance(value, str):
                    value = value.strip()
                
                # Convert case
                case_rule = rules.get("case")
                if case_rule and isinstance(value, str):
                    if case_rule == "lower":
                        value = value.lower()
                    elif case_rule == "upper":
                        value = value.upper()
                    elif case_rule == "title":
                        value = value.title()
                
                # Remove special characters
                if rules.get("remove_special_chars", False) and isinstance(value, str):
                    value = re.sub(r'[^\w\s-]', '', value)
                
                # Truncate length
                max_length = rules.get("max_length")
                if max_length and isinstance(value, str) and len(value) > max_length:
                    value = value[:max_length]
                
                # Default values
                if value is None or (isinstance(value, str) and not value.strip()):
                    default = rules.get("default")
                    if default is not None:
                        value = default
                
                sanitized[field] = value
        
        return sanitized


# Predefined business rule examples

    def validate_feedback_item(self, item) -> bool:
        """
        Validate a feedback item.
        
        Args:
            item: FeedbackItem object to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required attributes
            if not hasattr(item, "id") or not item.id:
                return False
                
            if not hasattr(item, "content") or not item.content:
                return False
                
            # Content should have minimum length
            if len(str(item.content).strip()) < 3:
                return False
                
            # Check if source exists
            if not hasattr(item, "source"):
                return False
            
            return True
            
        except Exception as e:
            return False


def validate_feedback_coherence(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that feedback content matches rating (if provided)."""
    warnings = []
    
    content = data.get("content", "").lower()
    rating = data.get("rating")
    
    if rating is not None:
        # Check for mismatch between rating and sentiment
        positive_words = ["good", "great", "excellent", "amazing", "love", "perfect"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst"]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if rating >= 4 and negative_count > positive_count:
            warnings.append("High rating but negative sentiment detected")
        elif rating <= 2 and positive_count > negative_count:
            warnings.append("Low rating but positive sentiment detected")
    
    return {
        "valid": True,
        "warnings": warnings
    }


def validate_ticket_completeness(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that ticket has sufficient information."""
    errors = []
    warnings = []
    
    category = data.get("category", "")
    description = data.get("description", "")
    technical_details = data.get("technical_details", {})
    
    # Bug tickets should have technical details
    if category == "Bug":
        if not technical_details or not isinstance(technical_details, dict):
            warnings.append("Bug ticket missing technical details")
        else:
            required_bug_fields = ["severity", "platform"]
            missing_fields = [field for field in required_bug_fields if field not in technical_details]
            if missing_fields:
                warnings.append(f"Bug ticket missing technical fields: {missing_fields}")
    
    # Feature requests should have use case
    elif category == "Feature Request":
        if "use case" not in description.lower() and "because" not in description.lower():
            warnings.append("Feature request might benefit from use case description")
    
    # Check description length
    if len(description) < 50:
        warnings.append("Ticket description is quite short")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
