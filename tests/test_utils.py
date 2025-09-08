"""
Test suite for utility modules.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.csv_handler import CSVHandler
from src.utils.validators import DataValidator, validate_feedback_coherence, validate_ticket_completeness
from src.utils.logger import get_logger, StructuredLogger, PerformanceLogger


class TestCSVHandler:
    """Test CSV handling utilities."""
    
    @pytest.fixture
    def csv_handler(self):
        """Create CSV handler for testing."""
        return CSVHandler()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            "id": ["test_001", "test_002", "test_003"],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85, 92, 78],
            "active": [True, False, True]
        })
    
    def test_write_and_read_csv(self, csv_handler, sample_data, tmp_path):
        """Test writing and reading CSV files."""
        csv_file = tmp_path / "test_data.csv"
        
        # Write CSV
        success = csv_handler.write_csv(sample_data, csv_file)
        assert success is True
        assert csv_file.exists()
        
        # Read CSV
        loaded_data = csv_handler.read_csv(csv_file)
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_append_to_csv(self, csv_handler, sample_data, tmp_path):
        """Test appending data to CSV files."""
        csv_file = tmp_path / "append_test.csv"
        
        # Write initial data
        csv_handler.write_csv(sample_data, csv_file)
        
        # Append new data
        new_data = [
            {"id": "test_004", "name": "David", "score": 88, "active": True}
        ]
        success = csv_handler.append_to_csv(new_data, csv_file)
        assert success is True
        
        # Verify appended data
        loaded_data = csv_handler.read_csv(csv_file)
        assert len(loaded_data) == len(sample_data) + 1
        assert loaded_data.iloc[-1]["id"] == "test_004"
    
    def test_validate_csv_schema(self, csv_handler, sample_data, tmp_path):
        """Test CSV schema validation."""
        csv_file = tmp_path / "schema_test.csv"
        csv_handler.write_csv(sample_data, csv_file)
        
        # Valid schema
        expected_columns = ["id", "name", "score", "active"]
        result = csv_handler.validate_csv_schema(csv_file, expected_columns)
        assert result["valid"] is True
        
        # Invalid schema - missing column
        expected_columns_invalid = ["id", "name", "score", "active", "missing_column"]
        result = csv_handler.validate_csv_schema(csv_file, expected_columns_invalid)
        assert result["valid"] is False
        assert "missing_column" in str(result["missing_columns"])
    
    def test_get_csv_info(self, csv_handler, sample_data, tmp_path):
        """Test getting CSV file information."""
        csv_file = tmp_path / "info_test.csv"
        csv_handler.write_csv(sample_data, csv_file)
        
        info = csv_handler.get_csv_info(csv_file)
        
        assert "row_count" in info
        assert "column_count" in info
        assert "columns" in info
        assert info["row_count"] == len(sample_data)
        assert info["column_count"] == len(sample_data.columns)
    
    def test_clean_csv_data(self, csv_handler):
        """Test CSV data cleaning."""
        # Create dirty data
        dirty_data = pd.DataFrame({
            "id": ["  test_001  ", "test_002", ""],
            "name": ["Alice", "  Bob  ", "nan"],
            "score": [85, "", "null"]
        })
        
        cleaned_data = csv_handler.clean_csv_data(dirty_data)
        
        # Check that whitespace is stripped
        assert cleaned_data.iloc[0]["id"] == "test_001"
        assert cleaned_data.iloc[1]["name"] == "Bob"
        
        # Check that null representations are handled
        assert pd.isna(cleaned_data.iloc[2]["id"])
    
    def test_merge_csv_files(self, csv_handler, tmp_path):
        """Test merging multiple CSV files."""
        # Create multiple CSV files
        data1 = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        data2 = pd.DataFrame({"id": [3, 4], "value": ["c", "d"]})
        
        file1 = tmp_path / "merge1.csv"
        file2 = tmp_path / "merge2.csv"
        output_file = tmp_path / "merged.csv"
        
        csv_handler.write_csv(data1, file1)
        csv_handler.write_csv(data2, file2)
        
        # Merge files
        success = csv_handler.merge_csv_files([file1, file2], output_file)
        assert success is True
        
        # Verify merged data
        merged_data = csv_handler.read_csv(output_file)
        assert len(merged_data) == 4
        assert set(merged_data["id"]) == {1, 2, 3, 4}


class TestDataValidator:
    """Test data validation utilities."""
    
    @pytest.fixture
    def validator(self):
        """Create data validator for testing."""
        return DataValidator()
    
    def test_validate_email(self, validator):
        """Test email validation."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "first+last@test-domain.org"
        ]
        
        invalid_emails = [
            "invalid.email",
            "@domain.com",
            "user@",
            "user name@domain.com",
            ""
        ]
        
        for email in valid_emails:
            assert validator.validate_email(email) is True, f"Should be valid: {email}"
        
        for email in invalid_emails:
            assert validator.validate_email(email) is False, f"Should be invalid: {email}"
    
    def test_validate_version(self, validator):
        """Test version string validation."""
        valid_versions = [
            "1.0",
            "2.1.3",
            "10.15.7.2"
        ]
        
        invalid_versions = [
            "v1.0",
            "1.0.0beta",
            "invalid",
            ""
        ]
        
        for version in valid_versions:
            assert validator.validate_version(version) is True, f"Should be valid: {version}"
        
        for version in invalid_versions:
            assert validator.validate_version(version) is False, f"Should be invalid: {version}"
    
    def test_validate_rating(self, validator):
        """Test rating validation."""
        valid_ratings = [1, 2, 3, 4, 5, "1", "5"]
        invalid_ratings = [0, 6, -1, "invalid", None, ""]
        
        for rating in valid_ratings:
            assert validator.validate_rating(rating) is True, f"Should be valid: {rating}"
        
        for rating in invalid_ratings:
            assert validator.validate_rating(rating) is False, f"Should be invalid: {rating}"
    
    def test_validate_date(self, validator):
        """Test date validation."""
        valid_dates = [
            "2024-01-01",
            "2024-01-01 10:30:00",
            "2024-01-01T10:30:00",
            "01/15/2024",
            "15/01/2024"
        ]
        
        invalid_dates = [
            "2024-13-01",  # Invalid month
            "invalid-date",
            "2024/01/01",  # Not in expected format
            ""
        ]
        
        for date in valid_dates:
            assert validator.validate_date(date) is True, f"Should be valid: {date}"
        
        for date in invalid_dates:
            assert validator.validate_date(date) is False, f"Should be invalid: {date}"
    
    def test_validate_data_dict(self, validator):
        """Test validating dictionary data."""
        valid_feedback = {
            "id": "test_001",
            "source_type": "app_store_review",
            "content": "This is a test review",
            "rating": 4
        }
        
        result = validator.validate_data(valid_feedback, "feedback_item")
        assert result["valid"] is True
        
        # Invalid data - missing required field
        invalid_feedback = {
            "source_type": "app_store_review",
            "content": "This is a test review"
            # Missing 'id' field
        }
        
        result = validator.validate_data(invalid_feedback, "feedback_item")
        assert result["valid"] is False
        assert any("Missing required field: id" in error for error in result["errors"])
    
    def test_validate_dataframe(self, validator):
        """Test validating DataFrame data."""
        # Valid DataFrame
        valid_df = pd.DataFrame({
            "review_id": ["rev_001", "rev_002"],
            "platform": ["Google Play", "App Store"],
            "rating": [4, 5],
            "review_text": ["Good app", "Excellent app"]
        })
        
        result = validator.validate_data(valid_df, "app_store_review")
        assert result["valid"] is True
        
        # Invalid DataFrame - missing required column
        invalid_df = pd.DataFrame({
            "platform": ["Google Play", "App Store"],
            "rating": [4, 5],
            "review_text": ["Good app", "Excellent app"]
            # Missing 'review_id' column
        })
        
        result = validator.validate_data(invalid_df, "app_store_review")
        assert result["valid"] is False
    
    def test_validate_csv_file(self, validator, tmp_path):
        """Test CSV file validation."""
        # Create test CSV
        test_data = pd.DataFrame({
            "review_id": ["rev_001", "rev_002"],
            "platform": ["Google Play", "App Store"],
            "rating": [4, 5],
            "review_text": ["Good app", "Excellent app"]
        })
        
        csv_file = tmp_path / "test_reviews.csv"
        test_data.to_csv(csv_file, index=False)
        
        # Define schema
        schema = {
            "required_columns": ["review_id", "platform", "rating", "review_text"],
            "column_types": {
                "review_id": "string",
                "platform": "string",
                "rating": "int",
                "review_text": "string"
            }
        }
        
        result = validator.validate_csv_file(str(csv_file), schema)
        assert result["valid"] is True
    
    def test_generate_validation_report(self, validator):
        """Test validation report generation."""
        validation_results = [
            {
                "valid": True,
                "errors": [],
                "warnings": ["Minor warning"]
            },
            {
                "valid": False,
                "errors": ["Critical error"],
                "warnings": []
            }
        ]
        
        report = validator.generate_validation_report(validation_results)
        
        assert "DATA VALIDATION REPORT" in report
        assert "Total validations: 2" in report
        assert "Valid: 1" in report
        assert "Invalid: 1" in report
        assert "Critical error" in report
        assert "Minor warning" in report
    
    def test_business_rules_validation(self, validator):
        """Test custom business rules validation."""
        # Test feedback coherence rule
        coherent_feedback = {
            "content": "This app is amazing and works perfectly!",
            "rating": 5
        }
        
        result = validate_feedback_coherence(coherent_feedback)
        assert result["valid"] is True
        assert len(result.get("warnings", [])) == 0
        
        # Test incoherent feedback
        incoherent_feedback = {
            "content": "This app is terrible and awful!",
            "rating": 5  # High rating but negative sentiment
        }
        
        result = validate_feedback_coherence(incoherent_feedback)
        assert result["valid"] is True  # Still valid, just has warnings
        assert len(result.get("warnings", [])) > 0
    
    def test_ticket_completeness_validation(self, validator):
        """Test ticket completeness validation."""
        # Complete bug ticket
        complete_bug_ticket = {
            "category": "Bug",
            "description": "App crashes when user tries to save data. This happens consistently on iPhone 14.",
            "technical_details": {
                "severity": "High",
                "platform": "iOS"
            }
        }
        
        result = validate_ticket_completeness(complete_bug_ticket)
        assert result["valid"] is True
        
        # Incomplete bug ticket
        incomplete_bug_ticket = {
            "category": "Bug",
            "description": "App crashes",
            "technical_details": {}
        }
        
        result = validate_ticket_completeness(incomplete_bug_ticket)
        assert result["valid"] is True  # Still valid, just has warnings
        assert len(result.get("warnings", [])) > 0


class TestLogger:
    """Test logging utilities."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        
        assert logger.name == "test_module"
        assert logger.level >= 0  # Has some logging level set
    
    def test_structured_logger(self, tmp_path):
        """Test structured logging."""
        # Temporarily override logs directory
        import src.utils.logger as logger_module
        original_logs_dir = logger_module.LOGS_DIR
        logger_module.LOGS_DIR = tmp_path
        
        try:
            struct_logger = StructuredLogger("test_structured")
            
            # Log some structured data
            struct_logger.log_structured("test_event", {"key": "value"})
            
            # Check if structured log file was created
            struct_log_file = tmp_path / "structured.jsonl"
            assert struct_log_file.exists()
            
            # Check content
            content = struct_log_file.read_text()
            assert "test_event" in content
            assert "key" in content
            assert "value" in content
        
        finally:
            # Restore original logs directory
            logger_module.LOGS_DIR = original_logs_dir
    
    def test_performance_logger(self):
        """Test performance logging."""
        perf_logger = PerformanceLogger("test_performance")
        
        # Test timer functionality
        perf_logger.start_timer("test_operation")
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        duration = perf_logger.end_timer("test_operation")
        
        assert duration is not None
        assert duration > 0
        assert duration < 1  # Should be less than 1 second
    
    def test_log_function_decorator(self):
        """Test function logging decorator."""
        from src.utils.logger import log_function_call
        
        @log_function_call
        def test_function(x, y):
            return x + y
        
        # This should not raise an exception and should log the call
        result = test_function(1, 2)
        assert result == 3


class TestIntegrationUtils:
    """Test integration between utility modules."""
    
    def test_csv_to_validation_pipeline(self, tmp_path):
        """Test complete pipeline from CSV to validation."""
        # Create test data
        test_data = pd.DataFrame({
            "review_id": ["rev_001", "rev_002", "rev_003"],
            "platform": ["Google Play", "App Store", "Google Play"],
            "rating": [4, 5, 2],
            "review_text": [
                "Good app, works well",
                "Excellent app, love it!",
                "App crashes frequently, terrible"
            ],
            "user_name": ["User1", "User2", "User3"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "app_version": ["1.0.0", "1.0.1", "1.0.1"]
        })
        
        csv_file = tmp_path / "test_reviews.csv"
        
        # Use CSV handler to write data
        csv_handler = CSVHandler()
        success = csv_handler.write_csv(test_data, csv_file)
        assert success is True
        
        # Validate the CSV file
        validator = DataValidator()
        schema = {
            "required_columns": ["review_id", "platform", "rating", "review_text"],
            "column_types": {
                "review_id": "string",
                "platform": "string", 
                "rating": "int",
                "review_text": "string"
            }
        }
        
        validation_result = validator.validate_csv_file(str(csv_file), schema)
        assert validation_result["valid"] is True
        
        # Read back and validate data structure
        loaded_data = csv_handler.read_csv(csv_file)
        data_validation = validator.validate_data(loaded_data, "app_store_review")
        assert data_validation["valid"] is True
    
    def test_error_handling_across_modules(self, tmp_path):
        """Test error handling across utility modules."""
        csv_handler = CSVHandler()
        validator = DataValidator()
        
        # Test with non-existent file
        non_existent_file = tmp_path / "does_not_exist.csv"
        
        # CSV handler should handle missing file gracefully
        with pytest.raises(FileNotFoundError):
            csv_handler.read_csv(non_existent_file)
        
        # Validator should handle missing file gracefully
        validation_result = validator.validate_csv_file(str(non_existent_file), {})
        assert validation_result["valid"] is False
        assert "Error reading CSV file" in validation_result["errors"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])