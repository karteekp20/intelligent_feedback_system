"""
Test suite for the main processing pipeline.
"""

import pytest
import asyncio
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.pipeline import FeedbackPipeline
from src.core.data_models import FeedbackItem, PipelineResult, SystemStats
from src.utils.csv_handler import CSVHandler


class TestFeedbackPipeline:
    """Test the main feedback processing pipeline."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample pipeline configuration."""
        return {
            "batch_size": 5,
            "max_concurrent": 2,
            "dry_run": True,
            "file_type": "all"
        }
    
    @pytest.fixture
    def sample_data_files(self, tmp_path):
        """Create sample data files for testing."""
        # Create reviews file
        reviews_data = {
            "review_id": ["rev_001", "rev_002", "rev_003"],
            "platform": ["Google Play", "App Store", "Google Play"],
            "rating": [1, 5, 3],
            "review_text": [
                "App crashes constantly",
                "Amazing app, love it!",
                "Please add dark mode"
            ],
            "user_name": ["User1", "User2", "User3"],
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "app_version": ["2.1.0", "2.1.0", "2.1.1"]
        }
        
        reviews_df = pd.DataFrame(reviews_data)
        reviews_file = tmp_path / "app_store_reviews.csv"
        reviews_df.to_csv(reviews_file, index=False)
        
        # Create emails file
        emails_data = {
            "email_id": ["email_001", "email_002"],
            "subject": ["Bug Report", "Feature Request"],
            "body": [
                "App crashes when I login",
                "Please add export functionality"
            ],
            "sender_email": ["user1@test.com", "user2@test.com"],
            "timestamp": ["2024-01-01T10:00:00", "2024-01-02T11:00:00"],
            "priority": ["High", "Medium"]
        }
        
        emails_df = pd.DataFrame(emails_data)
        emails_file = tmp_path / "support_emails.csv"
        emails_df.to_csv(emails_file, index=False)
        
        return {
            "input_dir": tmp_path,
            "reviews_file": reviews_file,
            "emails_file": emails_file
        }
    
    def test_pipeline_initialization(self, sample_config):
        """Test pipeline initialization."""
        pipeline = FeedbackPipeline(sample_config)
        
        assert pipeline.config == sample_config
        assert len(pipeline.agents) == 6
        assert pipeline.stats is not None
        assert isinstance(pipeline.stats, SystemStats)
    
    @pytest.mark.asyncio
    async def test_pipeline_run_success(self, sample_config, sample_data_files):
        """Test successful pipeline execution."""
        # Update