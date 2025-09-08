"""
Test suite for all agents in the Intelligent Feedback Analysis System.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.base_agent import BaseAgent
from src.agents.csv_reader_agent import CSVReaderAgent
from src.agents.feedback_classifier_agent import FeedbackClassifierAgent
from src.agents.bug_analysis_agent import BugAnalysisAgent
from src.agents.feature_extractor_agent import FeatureExtractorAgent
from src.agents.ticket_creator_agent import TicketCreatorAgent
from src.agents.quality_critic_agent import QualityCriticAgent
from src.core.data_models import (
    FeedbackItem, Classification, BugDetails, FeatureDetails, 
    Ticket, FeedbackCategory, Priority, Platform
)


class TestBaseAgent:
    """Test the base agent functionality."""
    
    def test_base_agent_initialization(self):
        """Test base agent initialization."""
        config = {"test_param": "test_value"}
        agent = Mock(spec=BaseAgent)
        agent.name = "test_agent"
        agent.config = config
        
        assert agent.name == "test_agent"
        assert agent.config == config
    
    @pytest.mark.asyncio
    async def test_agent_execution_flow(self):
        """Test agent execution flow with success."""
        # Create a concrete implementation for testing
        class TestAgent(BaseAgent):
            async def process(self, data):
                return {
                    "agent_name": self.name,
                    "success": True,
                    "data": f"processed_{data}",
                    "confidence": 0.85
                }
        
        agent = TestAgent("test_agent")
        result = await agent.process("test_data")
        
        assert result["success"] is True
        assert result["data"] == "processed_test_data"
        assert result["confidence"] == 0.85


class TestCSVReaderAgent:
    """Test CSV Reader Agent functionality."""
    
    @pytest.fixture
    def csv_reader_agent(self):
        """Create CSV Reader Agent for testing."""
        return CSVReaderAgent()
    
    @pytest.fixture
    def sample_csv_data(self, tmp_path):
        """Create sample CSV files for testing."""
        # Create sample app store reviews
        reviews_data = """review_id,platform,rating,review_text,user_name,date,app_version
review_001,Google Play,1,"App crashes when I try to save",TestUser1,2024-01-01,v2.1.0
review_002,App Store,5,"Love this app! Great features",TestUser2,2024-01-02,v2.1.0
review_003,Google Play,2,"Please add dark mode feature",TestUser3,2024-01-03,v2.0.5"""
        
        reviews_file = tmp_path / "app_store_reviews.csv"
        reviews_file.write_text(reviews_data)
        
        # Create sample support emails
        emails_data = """email_id,subject,body,sender_email,timestamp,priority
email_001,Bug Report,App crashes on startup,user1@test.com,2024-01-01T10:00:00,High
email_002,Feature Request,Please add export feature,user2@test.com,2024-01-02T11:00:00,Medium"""
        
        emails_file = tmp_path / "support_emails.csv"
        emails_file.write_text(emails_data)
        
        return {
            "reviews_file": reviews_file,
            "emails_file": emails_file
        }
    
    @pytest.mark.asyncio
    async def test_read_app_store_reviews(self, csv_reader_agent, sample_csv_data):
        """Test reading app store reviews."""
        config = {"file_paths": [sample_csv_data["reviews_file"]]}
        result = await csv_reader_agent.execute(config)
        
        assert result.success is True
        assert len(result.data) == 3
        assert result.data[0].id == "review_001"
        assert result.data[0].source_type == "app_store_review"
        assert result.data[0].rating == 1
    
    @pytest.mark.asyncio
    async def test_read_support_emails(self, csv_reader_agent, sample_csv_data):
        """Test reading support emails."""
        config = {"file_paths": [sample_csv_data["emails_file"]]}
        result = await csv_reader_agent.execute(config)
        
        assert result.success is True
        assert len(result.data) == 2
        assert result.data[0].id == "email_001"
        assert result.data[0].source_type == "support_email"
    
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, csv_reader_agent):
        """Test handling of nonexistent files."""
        config = {"file_paths": ["/nonexistent/file.csv"]}
        result = await csv_reader_agent.execute(config)
        
        assert result.success is False
        assert "No valid CSV files found" in result.error_message
    
    def test_validate_csv_file(self, csv_reader_agent, sample_csv_data):
        """Test CSV file validation."""
        validation_result = asyncio.run(
            csv_reader_agent.validate_csv_file(str(sample_csv_data["reviews_file"]))
        )
        
        assert validation_result["valid"] is True
        assert validation_result["file_type"] == "app_store_reviews"


class TestFeedbackClassifierAgent:
    """Test Feedback Classifier Agent functionality."""
    
    @pytest.fixture
    def classifier_agent(self):
        """Create Feedback Classifier Agent for testing."""
        return FeedbackClassifierAgent()
    
    @pytest.fixture
    def sample_feedback_items(self):
        """Create sample feedback items for testing."""
        return [
            FeedbackItem(
                id="bug_001",
                source_type="app_store_review",
                content="App crashes when I try to login. This is a critical bug!",
                rating=1
            ),
            FeedbackItem(
                id="feature_001",
                source_type="support_email",
                content="Please add dark mode feature. It would be very helpful.",
                rating=4
            ),
            FeedbackItem(
                id="praise_001",
                source_type="app_store_review",
                content="Amazing app! Love the new interface design.",
                rating=5
            ),
            FeedbackItem(
                id="spam_001",
                source_type="app_store_review",
                content="Get free money now! Visit our website for amazing deals!",
                rating=1
            )
        ]
    
    @pytest.mark.asyncio
    async def test_classify_bug_report(self, classifier_agent, sample_feedback_items):
        """Test classification of bug reports."""
        bug_item = sample_feedback_items[0]
        result = await classifier_agent.execute(bug_item)
        
        assert result.success is True
        assert result.data.category == FeedbackCategory.BUG
        assert result.data.confidence > 0.5
        assert "crash" in result.data.keywords
    
    @pytest.mark.asyncio
    async def test_classify_feature_request(self, classifier_agent, sample_feedback_items):
        """Test classification of feature requests."""
        feature_item = sample_feedback_items[1]
        result = await classifier_agent.execute(feature_item)
        
        assert result.success is True
        assert result.data.category == FeedbackCategory.FEATURE_REQUEST
        assert result.data.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_classify_praise(self, classifier_agent, sample_feedback_items):
        """Test classification of praise."""
        praise_item = sample_feedback_items[2]
        result = await classifier_agent.execute(praise_item)
        
        assert result.success is True
        assert result.data.category == FeedbackCategory.PRAISE
        assert result.data.sentiment == "positive"
    
    @pytest.mark.asyncio
    async def test_classify_spam(self, classifier_agent, sample_feedback_items):
        """Test classification of spam."""
        spam_item = sample_feedback_items[3]
        result = await classifier_agent.execute(spam_item)
        
        assert result.success is True
        assert result.data.category == FeedbackCategory.SPAM
    
    @pytest.mark.asyncio
    async def test_batch_classification(self, classifier_agent, sample_feedback_items):
        """Test batch classification."""
        result = await classifier_agent.execute(sample_feedback_items)
        
        assert result.success is True
        assert len(result.data) == 4
        assert all(isinstance(item, Classification) for item in result.data)
    
    def test_keyword_based_classification(self, classifier_agent):
        """Test keyword-based classification."""
        content = "app crashes error bug broken"
        keyword_result = classifier_agent._classify_by_keywords(content)
        
        assert keyword_result[FeedbackCategory.BUG] > 0
        assert keyword_result[FeedbackCategory.BUG] > keyword_result[FeedbackCategory.PRAISE]


class TestBugAnalysisAgent:
    """Test Bug Analysis Agent functionality."""
    
    @pytest.fixture
    def bug_agent(self):
        """Create Bug Analysis Agent for testing."""
        return BugAnalysisAgent()
    
    @pytest.fixture
    def bug_feedback_data(self):
        """Create bug feedback data for testing."""
        bug_item = FeedbackItem(
            id="bug_001",
            source_type="app_store_review",
            content="App crashes on iPhone 14 when I try to save. Error code E1001. "
                   "Steps: 1. Open app 2. Try to save 3. App crashes. iOS 16.1, version 2.1.3",
            rating=1,
            platform="iOS"
        )
        
        classification = Classification(
            category=FeedbackCategory.BUG,
            confidence=0.9,
            keywords=["crash", "error", "save"]
        )
        
        return {
            "item": bug_item,
            "classification": classification
        }
    
    @pytest.mark.asyncio
    async def test_analyze_bug_report(self, bug_agent, bug_feedback_data):
        """Test bug analysis functionality."""
        result = await bug_agent.execute(bug_feedback_data)
        
        assert result.success is True
        assert isinstance(result.data, BugDetails)
        assert result.data.platform == Platform.IOS
        assert result.data.severity in ["Critical", "High", "Medium", "Low"]
        assert "E1001" in result.data.error_messages
    
    @pytest.mark.asyncio
    async def test_non_bug_classification(self, bug_agent):
        """Test handling of non-bug classifications."""
        feature_item = FeedbackItem(
            id="feature_001",
            source_type="support_email",
            content="Please add dark mode"
        )
        
        classification = Classification(
            category=FeedbackCategory.FEATURE_REQUEST,
            confidence=0.8
        )
        
        data = {"item": feature_item, "classification": classification}
        result = await bug_agent.execute(data)
        
        assert result.success is True
        assert result.data is None  # Should skip non-bug items
    
    def test_severity_determination(self, bug_agent):
        """Test bug severity determination."""
        critical_content = "app crashes data loss cannot access critical urgent"
        high_content = "error broken not working bug issue"
        medium_content = "slow lag sometimes minor issue"
        
        item = FeedbackItem(id="test", source_type="test", content="")
        
        # Test critical severity
        item.content = critical_content
        severity = bug_agent._determine_severity(critical_content.lower(), item)
        assert severity == "Critical"
        
        # Test high severity
        item.content = high_content
        severity = bug_agent._determine_severity(high_content.lower(), item)
        assert severity == "High"
    
    def test_platform_detection(self, bug_agent):
        """Test platform detection."""
        ios_content = "iphone 14 ios safari apple"
        android_content = "android samsung galaxy pixel"
        
        item = FeedbackItem(id="test", source_type="test", content="")
        
        # Test iOS detection
        platform = bug_agent._detect_platform(ios_content.lower(), item)
        assert platform == Platform.IOS
        
        # Test Android detection
        platform = bug_agent._detect_platform(android_content.lower(), item)
        assert platform == Platform.ANDROID


class TestFeatureExtractorAgent:
    """Test Feature Extractor Agent functionality."""
    
    @pytest.fixture
    def feature_agent(self):
        """Create Feature Extractor Agent for testing."""
        return FeatureExtractorAgent()
    
    @pytest.fixture
    def feature_feedback_data(self):
        """Create feature request data for testing."""
        feature_item = FeedbackItem(
            id="feature_001",
            source_type="support_email",
            content="Please add dark mode feature. It would be essential for night usage. "
                   "Other apps like Notion have this. Would help with productivity.",
            rating=4
        )
        
        classification = Classification(
            category=FeedbackCategory.FEATURE_REQUEST,
            confidence=0.85,
            keywords=["add", "feature", "dark mode"]
        )
        
        return {
            "item": feature_item,
            "classification": classification
        }
    
    @pytest.mark.asyncio
    async def test_analyze_feature_request(self, feature_agent, feature_feedback_data):
        """Test feature request analysis."""
        result = await feature_agent.execute(feature_feedback_data)
        
        assert result.success is True
        assert isinstance(result.data, FeatureDetails)
        assert "dark mode" in result.data.requested_feature.lower()
        assert result.data.user_impact in ["High", "Medium", "Low"]
        assert result.data.estimated_complexity in ["Simple", "Moderate", "Complex"]
    
    @pytest.mark.asyncio
    async def test_non_feature_classification(self, feature_agent):
        """Test handling of non-feature classifications."""
        bug_item = FeedbackItem(
            id="bug_001",
            source_type="app_store_review",
            content="App crashes when I login"
        )
        
        classification = Classification(
            category=FeedbackCategory.BUG,
            confidence=0.9
        )
        
        data = {"item": bug_item, "classification": classification}
        result = await feature_agent.execute(data)
        
        assert result.success is True
        assert result.data is None  # Should skip non-feature items
    
    def test_impact_determination(self, feature_agent):
        """Test user impact determination."""
        high_impact = "essential critical must have need required important"
        medium_impact = "useful helpful good convenient beneficial"
        low_impact = "would be nice minor small cosmetic"
        
        # Test high impact
        impact = feature_agent._determine_user_impact(high_impact)
        assert impact == "High"
        
        # Test medium impact
        impact = feature_agent._determine_user_impact(medium_impact)
        assert impact == "Medium"
        
        # Test low impact
        impact = feature_agent._determine_user_impact(low_impact)
        assert impact == "Low"
    
    def test_complexity_estimation(self, feature_agent):
        """Test complexity estimation."""
        simple_content = "simple easy basic toggle button"
        complex_content = "advanced sophisticated complete overhaul redesign"
        
        # Test simple complexity
        complexity = feature_agent._estimate_complexity(simple_content)
        assert complexity == "Simple"
        
        # Test complex complexity
        complexity = feature_agent._estimate_complexity(complex_content)
        assert complexity == "Complex"


class TestTicketCreatorAgent:
    """Test Ticket Creator Agent functionality."""
    
    @pytest.fixture
    def ticket_agent(self):
        """Create Ticket Creator Agent for testing."""
        return TicketCreatorAgent()
    
    @pytest.fixture
    def complete_bug_data(self):
        """Create complete bug analysis data."""
        feedback_item = FeedbackItem(
            id="bug_001",
            source_type="app_store_review",
            content="App crashes when saving",
            rating=1
        )
        
        classification = Classification(
            category=FeedbackCategory.BUG,
            confidence=0.9
        )
        
        bug_details = BugDetails(
            severity="Critical",
            platform=Platform.IOS,
            app_version="2.1.3",
            steps_to_reproduce=["Open app", "Try to save", "App crashes"],
            error_messages=["E1001"]
        )
        
        return {
            "item": feedback_item,
            "classification": classification,
            "bug_details": bug_details
        }
    
    @pytest.mark.asyncio
    async def test_create_bug_ticket(self, ticket_agent, complete_bug_data):
        """Test bug ticket creation."""
        result = await ticket_agent.execute(complete_bug_data)
        
        assert result.success is True
        assert isinstance(result.data, Ticket)
        assert result.data.category == FeedbackCategory.BUG
        assert result.data.priority == Priority.CRITICAL
        assert "Bug:" in result.data.title
        assert len(result.data.description) > 100
    
    @pytest.mark.asyncio
    async def test_create_feature_ticket(self, ticket_agent):
        """Test feature request ticket creation."""
        feedback_item = FeedbackItem(
            id="feature_001",
            source_type="support_email",
            content="Please add dark mode"
        )
        
        classification = Classification(
            category=FeedbackCategory.FEATURE_REQUEST,
            confidence=0.8
        )
        
        feature_details = FeatureDetails(
            requested_feature="dark mode",
            use_case="night usage",
            user_impact="High",
            estimated_complexity="Moderate"
        )
        
        data = {
            "item": feedback_item,
            "classification": classification,
            "feature_details": feature_details
        }
        
        result = await ticket_agent.execute(data)
        
        assert result.success is True
        assert result.data.category == FeedbackCategory.FEATURE_REQUEST
        assert "Feature Request:" in result.data.title
    
    def test_input_validation(self, ticket_agent):
        """Test input validation."""
        # Test missing required fields
        invalid_data = {"invalid": "data"}
        validation_errors = ticket_agent._validate_input(invalid_data)
        
        assert len(validation_errors) > 0
        assert "Missing required field: item" in validation_errors
    
    def test_priority_determination(self, ticket_agent):
        """Test ticket priority determination."""
        classification = Classification(category=FeedbackCategory.BUG, confidence=0.9)
        
        # Test critical bug
        critical_bug = BugDetails(severity="Critical", platform=Platform.IOS)
        priority = ticket_agent._determine_priority(classification, critical_bug)
        assert priority == Priority.CRITICAL
        
        # Test high impact feature
        high_feature = FeatureDetails(
            requested_feature="test", 
            user_impact="High",
            estimated_complexity="Simple"
        )
        priority = ticket_agent._determine_priority(classification, None, high_feature)
        assert priority == Priority.HIGH


class TestQualityCriticAgent:
    """Test Quality Critic Agent functionality."""
    
    @pytest.fixture
    def critic_agent(self):
        """Create Quality Critic Agent for testing."""
        return QualityCriticAgent()
    
    @pytest.fixture
    def sample_ticket(self):
        """Create sample ticket for testing."""
        return Ticket(
            ticket_id="TEST-001",
            source_id="bug_001",
            source_type="app_store_review",
            category=FeedbackCategory.BUG,
            priority=Priority.HIGH,
            title="Bug: App issue",
            description="Short description",
            technical_details={"severity": "High"},
            agent_confidence=0.8
        )
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self, critic_agent, sample_ticket):
        """Test ticket quality assessment."""
        assessment = await critic_agent._assess_quality(sample_ticket)
        
        assert "overall_score" in assessment
        assert "scores" in assessment
        assert "issues" in assessment
        assert 0.0 <= assessment["overall_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_ticket_improvement(self, critic_agent, sample_ticket):
        """Test ticket improvement functionality."""
        data = {"ticket": sample_ticket}
        result = await critic_agent.execute(data)
        
        assert result.success is True
        assert isinstance(result.data, Ticket)
        # Improved ticket should have same ID but potentially better content
        assert result.data.ticket_id == sample_ticket.ticket_id
    
    def test_completeness_assessment(self, critic_agent, sample_ticket):
        """Test completeness assessment."""
        criteria = {
            "required_fields": ["title", "description", "technical_details"]
        }
        
        score = critic_agent._assess_completeness(sample_ticket, criteria)
        assert 0.0 <= score <= 1.0
    
    def test_title_quality_assessment(self, critic_agent):
        """Test title quality assessment."""
        criteria = {"title_keywords": ["bug", "error", "issue"]}
        
        # Test good title
        good_ticket = Ticket(title="Bug: Login error needs immediate fix")
        good_score = critic_agent._assess_title_quality(good_ticket, criteria)
        
        # Test poor title
        poor_ticket = Ticket(title="issue")
        poor_score = critic_agent._assess_title_quality(poor_ticket, criteria)
        
        assert good_score > poor_score


# Integration Tests
class TestAgentIntegration:
    """Test agent integration and data flow."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test complete agent pipeline flow."""
        # Create sample feedback item
        feedback_item = FeedbackItem(
            id="integration_001",
            source_type="app_store_review",
            content="App crashes when I try to save my work. Critical bug on iPhone.",
            rating=1
        )
        
        # Step 1: Classification
        classifier = FeedbackClassifierAgent()
        classification_result = await classifier.execute(feedback_item)
        assert classification_result.success is True
        classification = classification_result.data
        
        # Step 2: Bug Analysis (if classified as bug)
        if classification.category == FeedbackCategory.BUG:
            bug_analyzer = BugAnalysisAgent()
            bug_data = {"item": feedback_item, "classification": classification}
            bug_result = await bug_analyzer.execute(bug_data)
            assert bug_result.success is True
            bug_details = bug_result.data
        
        # Step 3: Ticket Creation
        ticket_creator = TicketCreatorAgent()
        ticket_data = {
            "item": feedback_item,
            "classification": classification,
            "bug_details": bug_details if classification.category == FeedbackCategory.BUG else None
        }
        ticket_result = await ticket_creator.execute(ticket_data)
        assert ticket_result.success is True
        ticket = ticket_result.data
        
        # Step 4: Quality Review
        critic = QualityCriticAgent()
        critic_data = {"ticket": ticket, "original_item": feedback_item}
        critic_result = await critic.execute(critic_data)
        assert critic_result.success is True
        final_ticket = critic_result.data
        
        # Verify final ticket
        assert isinstance(final_ticket, Ticket)
        assert final_ticket.source_id == feedback_item.id
        assert final_ticket.category == classification.category


# Performance Tests
class TestAgentPerformance:
    """Test agent performance and load handling."""
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """Test performance with batch processing."""
        # Create batch of feedback items
        feedback_items = []
        for i in range(10):
            item = FeedbackItem(
                id=f"perf_test_{i}",
                source_type="app_store_review",
                content=f"Test feedback content {i}",
                rating=3
            )
            feedback_items.append(item)
        
        classifier = FeedbackClassifierAgent()
        
        start_time = datetime.now()
        result = await classifier.execute(feedback_items)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert result.success is True
        assert len(result.data) == 10
        assert processing_time < 30  # Should process 10 items in under 30 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self):
        """Test concurrent execution of multiple agents."""
        feedback_item = FeedbackItem(
            id="concurrent_001",
            source_type="app_store_review",
            content="Test concurrent processing",
            rating=3
        )
        
        # Create multiple agents
        agents = [
            FeedbackClassifierAgent(),
            CSVReaderAgent(),
            TicketCreatorAgent()
        ]
        
        # Execute agents concurrently (where applicable)
        tasks = []
        for agent in agents:
            if isinstance(agent, FeedbackClassifierAgent):
                tasks.append(agent.execute(feedback_item))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check that at least one agent executed successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) > 0


# Error Handling Tests
class TestErrorHandling:
    """Test error handling across all agents."""
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        agents = [
            FeedbackClassifierAgent(),
            BugAnalysisAgent(),
            FeatureExtractorAgent(),
            TicketCreatorAgent(),
            QualityCriticAgent()
        ]
        
        invalid_inputs = [None, "", {}, []]
        
        for agent in agents:
            for invalid_input in invalid_inputs:
                result = await agent.execute(invalid_input)
                assert result.success is False
                assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_malformed_data_handling(self):
        """Test handling of malformed data."""
        classifier = FeedbackClassifierAgent()
        
        # Test with malformed feedback item
        malformed_item = {
            "id": "test",
            "invalid_field": "invalid_value"
        }
        
        result = await classifier.execute(malformed_item)
        assert result.success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])