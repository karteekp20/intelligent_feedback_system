"""
Feature Extractor Agent for analyzing feature requests and extracting detailed requirements.
Transforms user feedback into structured feature specifications with business analysis.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import openai
from collections import Counter

from .base_agent import BaseAgent
from ..core.data_models import (
    AgentResult, FeedbackItem, FeatureDetails, FeedbackCategory, 
    Priority, Platform, Classification
)
from ..core.nlp_utils import NLPUtils
from ..utils.logger import get_logger


class FeatureExtractorAgent(BaseAgent):
    """Agent responsible for extracting and analyzing feature requests from feedback."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Feature Extractor Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("feature_extractor", config)
        self.nlp_utils = NLPUtils()
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.extract_use_cases = config.get("extract_use_cases", True) if config else True
        self.estimate_complexity = config.get("estimate_complexity", True) if config else True
        self.identify_dependencies = config.get("identify_dependencies", False) if config else False
        self.batch_size = config.get("batch_size", 5) if config else 5
        self.confidence_threshold = config.get("confidence_threshold", 0.7) if config else 0.7
        
        # OpenAI configuration
        self.openai_model = config.get("openai_model", "gpt-3.5-turbo") if config else "gpt-3.5-turbo"
        self.max_tokens = config.get("max_tokens", 800) if config else 800
        self.temperature = config.get("temperature", 0.4) if config else 0.4
        
        # Feature analysis patterns
        self.feature_patterns = self._load_feature_patterns()
        self.use_case_indicators = self._load_use_case_indicators()
        self.complexity_indicators = self._load_complexity_indicators()
        self.platform_indicators = self._load_platform_indicators()
        self.business_value_keywords = self._load_business_value_keywords()
        
        # Template libraries
        self.user_story_templates = self._load_user_story_templates()
        self.acceptance_criteria_templates = self._load_acceptance_criteria_templates()
        
        # Statistics tracking
        self.extraction_stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "feature_categories": Counter(),
            "complexity_distribution": Counter(),
            "average_confidence": 0.0
        }
    
    def _load_feature_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying different types of feature requests."""
        return {
            "new_feature": [
                "add", "new", "create", "implement", "build", "develop", "introduce",
                "feature", "functionality", "capability", "option", "ability"
            ],
            "enhancement": [
                "improve", "enhance", "better", "upgrade", "optimize", "refine",
                "extend", "expand", "increase", "boost", "strengthen"
            ],
            "improvement": [
                "fix", "update", "modify", "change", "adjust", "tweak", "revise",
                "polish", "streamline", "simplify", "reorganize"
            ],
            "integration": [
                "integrate", "connect", "sync", "link", "api", "webhook", "import",
                "export", "third party", "external", "plugin", "addon"
            ],
            "ui_ux": [
                "interface", "design", "layout", "theme", "style", "appearance",
                "navigation", "menu", "button", "icon", "visual", "user experience"
            ],
            "automation": [
                "automatic", "auto", "automated", "schedule", "batch", "bulk",
                "trigger", "workflow", "process", "streamline"
            ],
            "analytics": [
                "report", "analytics", "dashboard", "chart", "graph", "metrics",
                "statistics", "insights", "tracking", "monitoring"
            ],
            "security": [
                "security", "privacy", "permission", "access control", "authentication",
                "authorization", "encryption", "secure", "protection"
            ]
        }
    
    def _load_use_case_indicators(self) -> List[str]:
        """Load indicators that suggest use cases in text."""
        return [
            "when", "if", "so that", "in order to", "because", "since",
            "use case", "scenario", "situation", "example", "instance",
            "need to", "want to", "would like to", "should be able to"
        ]
    
    def _load_complexity_indicators(self) -> Dict[str, List[str]]:
        """Load indicators for estimating feature complexity."""
        return {
            "simple": [
                "button", "link", "color", "text", "label", "icon", "tooltip",
                "simple", "basic", "easy", "quick", "minor", "small"
            ],
            "medium": [
                "form", "page", "screen", "filter", "sort", "search", "validation",
                "notification", "email", "moderate", "standard", "typical"
            ],
            "complex": [
                "integration", "api", "database", "algorithm", "workflow", "automation",
                "report", "dashboard", "analytics", "complex", "advanced", "sophisticated"
            ],
            "very_complex": [
                "architecture", "system", "platform", "framework", "migration", "overhaul",
                "redesign", "machine learning", "ai", "real-time", "scalability"
            ]
        }
    
    def _load_platform_indicators(self) -> Dict[Platform, List[str]]:
        """Load platform detection patterns."""
        return {
            Platform.IOS: ["ios", "iphone", "ipad", "apple", "app store"],
            Platform.ANDROID: ["android", "google play", "play store"],
            Platform.WEB_CHROME: ["chrome", "browser", "web"],
            Platform.WEB_FIREFOX: ["firefox", "browser", "web"],
            Platform.WEB_SAFARI: ["safari", "browser", "web"],
            Platform.WEB_EDGE: ["edge", "browser", "web"],
            Platform.WINDOWS: ["windows", "desktop", "pc"],
            Platform.MACOS: ["mac", "macos", "desktop"],
            Platform.LINUX: ["linux", "desktop"],
            Platform.REST_API: ["api", "rest", "endpoint", "integration"],
            Platform.MULTIPLE: ["all platforms", "cross-platform", "everywhere"]
        }
    
    def _load_business_value_keywords(self) -> Dict[str, List[str]]:
        """Load keywords indicating business value."""
        return {
            "revenue": [
                "revenue", "money", "profit", "sales", "income", "earnings",
                "monetize", "pricing", "subscription", "payment"
            ],
            "efficiency": [
                "efficiency", "productivity", "time", "speed", "faster", "quick",
                "streamline", "automate", "simplify", "optimize"
            ],
            "user_satisfaction": [
                "satisfaction", "happy", "pleased", "delight", "experience",
                "usability", "convenience", "user-friendly", "intuitive"
            ],
            "retention": [
                "retention", "engage", "engagement", "loyalty", "return",
                "sticky", "addictive", "compelling", "valuable"
            ],
            "competitive": [
                "competitive", "competitor", "advantage", "differentiate",
                "unique", "innovative", "cutting-edge", "market"
            ],
            "cost_reduction": [
                "cost", "save", "reduce", "cheaper", "efficient", "budget",
                "resource", "minimize", "cut", "economical"
            ]
        }
    
    def _load_user_story_templates(self) -> List[str]:
        """Load user story templates."""
        return [
            "As a {user_type}, I want {functionality} so that {benefit}",
            "As a {user_type}, I need {functionality} to {goal}",
            "As a {user_type}, I should be able to {action} in order to {outcome}",
            "When I {context}, I want to {action} so that {result}",
            "Given that I am {user_type}, I want {functionality} because {reason}"
        ]
    
    def _load_acceptance_criteria_templates(self) -> List[str]:
        """Load acceptance criteria templates."""
        return [
            "Given {context}, when {action}, then {expected_result}",
            "The system should {requirement}",
            "Users must be able to {capability}",
            "The feature should {behavior}",
            "When {condition}, the system must {response}"
        ]
    
    async def process(self, data: Any) -> AgentResult:
        """
        Process feedback items and extract feature details.
        
        Args:
            data: List of FeedbackItem objects, single FeedbackItem, or list of Classifications
            
        Returns:
            AgentResult containing extracted feature details
        """
        try:
            self.logger.info("Starting feature extraction process")
            start_time = datetime.now()
            
            # Handle different input types
            feedback_items = self._extract_feedback_items(data)
            
            if not feedback_items:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="No valid feedback items found for feature extraction"
                )
            
            # Filter for feature requests and enhancements
            feature_items = self._filter_feature_requests(feedback_items)
            
            if not feature_items:
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    data={"feature_details": [], "processing_stats": {"no_features_found": len(feedback_items)}},
                    details="No feature requests found in input"
                )
            
            # Process in batches
            all_feature_details = []
            processing_stats = {
                "total_items": len(feature_items),
                "successful_extractions": 0,
                "failed_extractions": 0,
                "average_confidence": 0.0
            }
            
            for i in range(0, len(feature_items), self.batch_size):
                batch = feature_items[i:i + self.batch_size]
                
                try:
                    batch_features = await self._extract_features_batch(batch)
                    all_feature_details.extend(batch_features)
                    processing_stats["successful_extractions"] += len(batch_features)
                    
                except Exception as e:
                    self.logger.error(f"Error processing feature batch {i//self.batch_size + 1}: {str(e)}")
                    processing_stats["failed_extractions"] += len(batch)
                    continue
            
            # Calculate statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if all_feature_details:
                processing_stats["average_confidence"] = sum(
                    f.confidence_score for f in all_feature_details
                ) / len(all_feature_details)
                
                # Update global statistics
                self._update_statistics(all_feature_details)
            
            result_data = {
                "feature_details": all_feature_details,
                "processing_stats": processing_stats
            }
            
            self.logger.info(f"Feature extraction completed. Extracted {len(all_feature_details)} features "
                           f"from {len(feature_items)} feature requests in {processing_time:.2f} seconds")
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result_data,
                confidence=processing_stats.get("average_confidence", 0),
                details=f"Extracted {len(all_feature_details)} feature specifications",
                processing_time=processing_time,
                metadata={
                    "processing_stats": processing_stats,
                    "feature_count": len(all_feature_details)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Feature extraction process failed: {str(e)}", exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"Feature extraction failed: {str(e)}"
            )
    
    def _extract_feedback_items(self, data: Any) -> List[FeedbackItem]:
        """Extract FeedbackItem objects from various input types."""
        feedback_items = []
        
        if isinstance(data, FeedbackItem):
            feedback_items = [data]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, FeedbackItem):
                    feedback_items.append(item)
                elif isinstance(item, Classification):
                    # Extract feedback item from classification if available
                    # This would require the classification to store the original feedback
                    # For now, we'll skip classifications
                    continue
                elif hasattr(item, 'feedback_item'):
                    feedback_items.append(item.feedback_item)
        elif isinstance(data, dict) and 'feedback_items' in data:
            feedback_items = data['feedback_items']
        
        return feedback_items
    
    def _filter_feature_requests(self, feedback_items: List[FeedbackItem]) -> List[FeedbackItem]:
        """Filter feedback items to only include feature requests and enhancements."""
        feature_items = []
        
        for item in feedback_items:
            # Check if already classified as feature request
            if (hasattr(item, 'category') and 
                item.category == FeedbackCategory.FEATURE_REQUEST):
                feature_items.append(item)
                continue
            
            # Use pattern matching to identify potential feature requests
            content_lower = item.content.lower()
            
            # Check for feature request indicators
            feature_indicators = self.feature_patterns.get("new_feature", []) + \
                               self.feature_patterns.get("enhancement", []) + \
                               self.feature_patterns.get("improvement", [])
            
            if any(indicator in content_lower for indicator in feature_indicators):
                feature_items.append(item)
                continue
            
            # Check for request patterns
            request_patterns = [
                "please add", "would like", "could you", "wish", "hope",
                "suggestion", "request", "feature", "enhancement"
            ]
            
            if any(pattern in content_lower for pattern in request_patterns):
                feature_items.append(item)
        
        return feature_items
    
    async def _extract_features_batch(self, feedback_items: List[FeedbackItem]) -> List[FeatureDetails]:
        """Extract features from a batch of feedback items."""
        feature_details = []
        
        # Process items concurrently
        extraction_tasks = [
            self._extract_single_feature(item) for item in feedback_items
        ]
        
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        for i, result in enumerate(extraction_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error extracting feature from {feedback_items[i].id}: {str(result)}")
                continue
            else:
                feature_details.append(result)
        
        return feature_details
    
    async def _extract_single_feature(self, feedback_item: FeedbackItem) -> FeatureDetails:
        """
        Extract detailed feature information from a single feedback item.
        
        Args:
            feedback_item: Feedback item to analyze
            
        Returns:
            FeatureDetails object
        """
        try:
            # Initialize feature details
            feature_details = FeatureDetails(
                feedback_id=feedback_item.id,
                feature_title="",
                description=feedback_item.content
            )
            
            # Step 1: Basic feature analysis
            await self._analyze_feature_type(feedback_item, feature_details)
            
            # Step 2: Extract use cases and user stories
            if self.extract_use_cases:
                await self._extract_use_cases(feedback_item, feature_details)
                await self._generate_user_stories(feedback_item, feature_details)
            
            # Step 3: Business value analysis
            await self._analyze_business_value(feedback_item, feature_details)
            
            # Step 4: Technical analysis
            await self._analyze_technical_requirements(feedback_item, feature_details)
            
            # Step 5: Platform analysis
            await self._detect_platforms(feedback_item, feature_details)
            
            # Step 6: Complexity and effort estimation
            if self.estimate_complexity:
                await self._estimate_complexity_and_effort(feedback_item, feature_details)
            
            # Step 7: Dependency analysis
            if self.identify_dependencies:
                await self._identify_dependencies(feedback_item, feature_details)
            
            # Step 8: Generate acceptance criteria
            await self._generate_acceptance_criteria(feedback_item, feature_details)
            
            # Step 9: AI-enhanced analysis
            await self._ai_enhanced_analysis(feedback_item, feature_details)
            
            # Step 10: Final validation and confidence scoring
            await self._validate_and_score(feature_details)
            
            return feature_details
            
        except Exception as e:
            self.logger.error(f"Error extracting feature from {feedback_item.id}: {str(e)}")
            # Return basic feature details with error information
            return FeatureDetails(
                feedback_id=feedback_item.id,
                feature_title="Feature extraction failed",
                description=feedback_item.content,
                confidence_score=0.0,
                metadata={"extraction_error": str(e)}
            )
    
    async def _analyze_feature_type(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Analyze and categorize the type of feature request."""
        content_lower = feedback_item.content.lower()
        
        # Determine feature category
        category_scores = {}
        for category, keywords in self.feature_patterns.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            feature_details.category = max(category_scores, key=category_scores.get)
        else:
            feature_details.category = "enhancement"  # Default
        
        # Generate initial title if not provided
        if not feature_details.feature_title:
            feature_details.feature_title = self._generate_feature_title(feedback_item.content)
    
    def _generate_feature_title(self, content: str) -> str:
        """Generate a concise feature title from content."""
        # Remove common request phrases
        clean_content = content
        request_phrases = [
            "please add", "could you add", "would like to see", "wish you could",
            "it would be great if", "i would love to", "can you implement",
            "suggestion:", "feature request:", "enhancement:"
        ]
        
        for phrase in request_phrases:
            clean_content = re.sub(phrase, "", clean_content, flags=re.IGNORECASE)
        
        # Take first sentence or first 80 characters
        sentences = clean_content.split('.')
        title = sentences[0].strip()
        
        if len(title) > 80:
            title = title[:77] + "..."
        
        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]
        
        return title or "Feature Request"
    
    async def _extract_use_cases(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Extract use cases from the feedback content."""
        content = feedback_item.content
        use_cases = []
        
        # Look for explicit use case indicators
        for indicator in self.use_case_indicators:
            if indicator in content.lower():
                # Extract sentences containing the indicator
                sentences = content.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        use_case = sentence.strip()
                        if len(use_case) > 10:  # Filter out very short fragments
                            use_cases.append(use_case)
        
        # Look for scenario patterns
        scenario_patterns = [
            r"when (.*?),",
            r"if (.*?),",
            r"scenario:? (.*?)[\.\n]",
            r"use case:? (.*?)[\.\n]",
            r"example:? (.*?)[\.\n]"
        ]
        
        for pattern in scenario_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                use_case = match.group(1).strip()
                if len(use_case) > 5:
                    use_cases.append(use_case)
        
        # Remove duplicates and add to feature details
        unique_use_cases = list(dict.fromkeys(use_cases))[:5]  # Limit to 5
        feature_details.use_cases.extend(unique_use_cases)
    
    async def _generate_user_stories(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Generate user stories based on extracted information."""
        if not feature_details.use_cases:
            return
        
        # Infer user type from feedback source
        user_type = self._infer_user_type(feedback_item)
        
        # Generate user stories from use cases
        for use_case in feature_details.use_cases[:3]:  # Limit to 3 stories
            # Try to extract action and benefit from use case
            action = self._extract_action_from_use_case(use_case)
            benefit = self._extract_benefit_from_use_case(use_case)
            
            if action:
                template = self.user_story_templates[0]  # Use primary template
                user_story = template.format(
                    user_type=user_type,
                    functionality=action,
                    benefit=benefit or "achieve my goal"
                )
                feature_details.user_stories.append(user_story)
    
    def _infer_user_type(self, feedback_item: FeedbackItem) -> str:
        """Infer user type from feedback item."""
        content_lower = feedback_item.content.lower()
        
        # Check for explicit user type mentions
        if any(word in content_lower for word in ["admin", "administrator", "manager"]):
            return "administrator"
        elif any(word in content_lower for word in ["developer", "programmer", "api"]):
            return "developer"
        elif any(word in content_lower for word in ["customer", "client", "buyer"]):
            return "customer"
        elif any(word in content_lower for word in ["student", "learner"]):
            return "student"
        else:
            return "user"  # Default
    
    def _extract_action_from_use_case(self, use_case: str) -> str:
        """Extract the main action from a use case."""
        # Look for action verbs
        action_patterns = [
            r"(add|create|build|implement|generate|make) (.*?)(?:\s+so|\s+to|\s*$)",
            r"(edit|modify|update|change|adjust) (.*?)(?:\s+so|\s+to|\s*$)",
            r"(view|see|display|show|list) (.*?)(?:\s+so|\s+to|\s*$)",
            r"(search|find|filter|sort) (.*?)(?:\s+so|\s+to|\s*$)",
            r"(delete|remove|cancel) (.*?)(?:\s+so|\s+to|\s*$)"
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, use_case, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {match.group(2)}".strip()
        
        # Fallback: return cleaned use case
        return re.sub(r"when|if|so that|in order to", "", use_case, flags=re.IGNORECASE).strip()
    
    def _extract_benefit_from_use_case(self, use_case: str) -> Optional[str]:
        """Extract the benefit/goal from a use case."""
        benefit_patterns = [
            r"so that (.*?)(?:\.|$)",
            r"in order to (.*?)(?:\.|$)",
            r"to (.*?)(?:\.|$)",
            r"because (.*?)(?:\.|$)"
        ]
        
        for pattern in benefit_patterns:
            match = re.search(pattern, use_case, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    async def _analyze_business_value(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Analyze the business value and impact of the feature."""
        content_lower = feedback_item.content.lower()
        
        # Identify business value indicators
        value_indicators = []
        for value_type, keywords in self.business_value_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    value_indicators.append(value_type)
                    break
        
        # Generate business value description
        if value_indicators:
            primary_value = value_indicators[0]
            
            value_descriptions = {
                "revenue": "This feature could drive revenue growth by improving user conversion and retention",
                "efficiency": "This feature will improve operational efficiency and reduce time-to-completion",
                "user_satisfaction": "This feature will enhance user satisfaction and overall user experience",
                "retention": "This feature will improve user retention and engagement metrics",
                "competitive": "This feature will provide competitive advantage and market differentiation",
                "cost_reduction": "This feature will reduce operational costs and resource requirements"
            }
            
            feature_details.business_value = value_descriptions.get(
                primary_value, 
                "This feature will provide value to the business and users"
            )
        
        # Estimate market demand based on content analysis
        demand_indicators = {
            "very_high": ["everyone", "all users", "critical", "essential", "must have"],
            "high": ["many users", "important", "significant", "major"],
            "medium": ["some users", "useful", "helpful", "beneficial"],
            "low": ["few users", "nice to have", "optional", "minor"]
        }
        
        for demand_level, indicators in demand_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                feature_details.market_demand = demand_level
                break
        
        if not feature_details.market_demand:
            feature_details.market_demand = "medium"  # Default
    
    async def _analyze_technical_requirements(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Analyze technical requirements and considerations."""
        content_lower = feedback_item.content.lower()
        
        # Identify technical requirements
        technical_patterns = {
            "api": ["api", "rest", "endpoint", "integration", "webhook"],
            "database": ["database", "storage", "data", "save", "persist"],
            "ui": ["interface", "screen", "page", "form", "button", "menu"],
            "performance": ["fast", "speed", "performance", "optimize", "cache"],
            "security": ["secure", "privacy", "encrypt", "auth", "permission"],
            "mobile": ["mobile", "phone", "app", "touch", "responsive"],
            "real_time": ["real-time", "live", "instant", "immediate", "push"],
            "offline": ["offline", "sync", "cache", "local storage"]
        }
        
        for requirement_type, keywords in technical_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                requirement_desc = f"Implementation requires {requirement_type} considerations"
                feature_details.technical_requirements.append(requirement_desc)
        
        # UI/UX requirements
        ui_indicators = ["design", "layout", "interface", "user experience", "usability"]
        if any(indicator in content_lower for indicator in ui_indicators):
            feature_details.ui_ux_requirements.append("User interface design and user experience optimization")
        
        # Performance requirements
        perf_indicators = ["fast", "quick", "speed", "performance", "responsive"]
        if any(indicator in content_lower for indicator in perf_indicators):
            feature_details.performance_requirements.append("High performance and responsiveness required")
        
        # Security considerations
        security_indicators = ["secure", "privacy", "safe", "protection", "confidential"]
        if any(indicator in content_lower for indicator in security_indicators):
            feature_details.security_considerations.append("Security and privacy protection measures needed")
    
    async def _detect_platforms(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Detect target platforms for the feature."""
        content_lower = feedback_item.content.lower()
        
        detected_platforms = []
        for platform, indicators in self.platform_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                detected_platforms.append(platform)
        
        # If no specific platform mentioned, check source
        if not detected_platforms:
            if feedback_item.source.is_mobile():
                detected_platforms.append(Platform.IOS)
                detected_platforms.append(Platform.ANDROID)
            else:
                detected_platforms.append(Platform.WEB_CHROME)  # Default to web
        
        feature_details.platforms_affected = detected_platforms
    
    async def _estimate_complexity_and_effort(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Estimate complexity and development effort."""
        content_lower = feedback_item.content.lower()
        
        # Calculate complexity score
        complexity_score = 0
        complexity_reasons = []
        
        for complexity_level, indicators in self.complexity_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in content_lower)
            if matches > 0:
                level_weights = {"simple": 1, "medium": 2, "complex": 3, "very_complex": 4}
                complexity_score += matches * level_weights[complexity_level]
                complexity_reasons.append(f"{matches} {complexity_level} indicators")
        
        # Additional complexity factors
        complexity_score += len(feature_details.technical_requirements)
        complexity_score += len(feature_details.platforms_affected)
        complexity_score += len(feature_details.dependencies) * 2
        
        # Determine complexity level
        if complexity_score >= 15:
            feature_details.complexity = "very_high"
            feature_details.effort_estimate = "xl"
        elif complexity_score >= 10:
            feature_details.complexity = "high"
            feature_details.effort_estimate = "large"
        elif complexity_score >= 5:
            feature_details.complexity = "medium"
            feature_details.effort_estimate = "medium"
        else:
            feature_details.complexity = "low"
            feature_details.effort_estimate = "small"
        
        # Add complexity reasoning to metadata
        feature_details.metadata["complexity_analysis"] = {
            "score": complexity_score,
            "reasons": complexity_reasons
        }
    
    async def _identify_dependencies(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Identify potential dependencies and prerequisites."""
        content_lower = feedback_item.content.lower()
        
        # Common dependency patterns
        dependency_patterns = {
            "authentication": ["login", "auth", "user account", "sign in", "credentials"],
            "user_management": ["user profile", "user settings", "account management"],
            "data_storage": ["database", "save data", "store information", "persistence"],
            "payment_system": ["payment", "billing", "subscription", "purchase", "checkout"],
            "notification_system": ["notify", "alert", "email", "push notification", "reminder"],
            "search_functionality": ["search", "find", "query", "filter", "sort"],
            "file_management": ["upload", "download", "file", "document", "attachment"],
            "integration_apis": ["api", "third party", "external", "integration", "sync"],
            "mobile_app": ["mobile app", "ios", "android", "phone", "tablet"],
            "admin_panel": ["admin", "administrator", "management", "configuration"]
        }
        
        identified_dependencies = []
        for dependency, keywords in dependency_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                identified_dependencies.append(dependency.replace("_", " ").title())
        
        # Add explicit dependency mentions
        dependency_indicators = [
            "depends on", "requires", "needs", "prerequisite", "must have",
            "based on", "built on", "integration with"
        ]
        
        for indicator in dependency_indicators:
            if indicator in content_lower:
                # Try to extract what comes after the indicator
                pattern = rf"{indicator}\s+([^.]*)"
                match = re.search(pattern, content_lower)
                if match:
                    dependency = match.group(1).strip()
                    if len(dependency) > 3 and dependency not in identified_dependencies:
                        identified_dependencies.append(dependency)
        
        feature_details.dependencies = identified_dependencies[:5]  # Limit to 5
    
    async def _generate_acceptance_criteria(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Generate acceptance criteria based on feature analysis."""
        criteria = []
        
        # Basic functionality criteria
        if feature_details.feature_title:
            criteria.append(f"The {feature_details.category} functionality is implemented and working")
        
        # Use case based criteria
        for use_case in feature_details.use_cases[:3]:
            criteria.append(f"User can {use_case.lower()}")
        
        # Platform specific criteria
        if feature_details.platforms_affected:
            platforms = [p.get_display_name() for p in feature_details.platforms_affected]
            criteria.append(f"Feature works correctly on {', '.join(platforms)}")
        
        # Performance criteria
        if any("performance" in req.lower() for req in feature_details.performance_requirements):
            criteria.append("Feature meets performance requirements (response time < 2 seconds)")
        
        # Security criteria
        if feature_details.security_considerations:
            criteria.append("Security requirements are met and validated")
        
        # UI/UX criteria
        if feature_details.ui_ux_requirements:
            criteria.append("User interface is intuitive and follows design guidelines")
        
        # Testing criteria
        criteria.append("Feature is fully tested (unit, integration, and user acceptance tests)")
        criteria.append("Documentation is updated to reflect new functionality")
        
        feature_details.acceptance_criteria = criteria[:8]  # Limit to 8 criteria
    
    async def _ai_enhanced_analysis(self, feedback_item: FeedbackItem, feature_details: FeatureDetails):
        """Use AI to enhance feature analysis with more sophisticated insights."""
        try:
            # Create comprehensive prompt for AI analysis
            prompt = self._create_ai_analysis_prompt(feedback_item, feature_details)
            
            # Call OpenAI API
            response = await openai.ChatCompletion.acreate(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self._get_ai_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30
            )
            
            # Parse and integrate AI insights
            ai_analysis = self._parse_ai_analysis(response.choices[0].message.content)
            self._integrate_ai_insights(feature_details, ai_analysis)
            
        except Exception as e:
            self.logger.warning(f"AI enhanced analysis failed: {str(e)}")
            # Continue without AI enhancement
    
    def _create_ai_analysis_prompt(self, feedback_item: FeedbackItem, feature_details: FeatureDetails) -> str:
        """Create prompt for AI-enhanced feature analysis."""
        return f"""Analyze this feature request and provide enhanced insights:

Original Feedback: "{feedback_item.content}"
Feature Title: "{feature_details.feature_title}"
Category: {feature_details.category}
Initial Use Cases: {feature_details.use_cases}

Please provide:
1. Refined user benefit description
2. Implementation approach suggestions
3. Potential risks and mitigation strategies
4. Testing strategy recommendations
5. Documentation requirements
6. Additional use cases or edge cases
7. Competitor analysis (if applicable)
8. Priority recommendation with reasoning

Focus on practical, actionable insights that would help a development team implement this feature successfully."""
    
    def _get_ai_system_prompt(self) -> str:
        """Get system prompt for AI analysis."""
        return """You are a senior product manager and technical architect. Your job is to analyze feature requests and provide comprehensive, actionable insights for development teams. 

Consider:
- User experience and business value
- Technical feasibility and implementation complexity
- Risk assessment and mitigation strategies
- Testing and quality assurance requirements
- Market competitiveness and user needs

Provide practical, detailed recommendations that development teams can act upon immediately."""
    
    def _parse_ai_analysis(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI analysis response."""
        analysis = {
            "user_benefit": "",
            "implementation_approach": "",
            "risks": [],
            "testing_strategy": "",
            "documentation_needs": [],
            "additional_use_cases": [],
            "competitor_analysis": "",
            "priority_recommendation": ""
        }
        
        try:
            # Try to parse structured response
            if "{" in ai_response and "}" in ai_response:
                import json
                analysis.update(json.loads(ai_response))
            else:
                # Parse unstructured response
                self._parse_unstructured_ai_response(ai_response, analysis)
        except Exception as e:
            self.logger.warning(f"Error parsing AI analysis: {str(e)}")
        
        return analysis
    
    def _parse_unstructured_ai_response(self, response: str, analysis: Dict[str, Any]):
        """Parse unstructured AI response using pattern matching."""
        sections = {
            "user_benefit": r"(?:user benefit|benefit)[:\s]*(.*?)(?:\n\n|\d\.)",
            "implementation": r"(?:implementation|approach)[:\s]*(.*?)(?:\n\n|\d\.)",
            "risks": r"(?:risks?|challenges?)[:\s]*(.*?)(?:\n\n|\d\.)",
            "testing": r"(?:testing|test)[:\s]*(.*?)(?:\n\n|\d\.)",
            "documentation": r"(?:documentation|docs)[:\s]*(.*?)(?:\n\n|\d\.)",
            "priority": r"(?:priority|recommendation)[:\s]*(.*?)(?:\n\n|\d\.)"
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if key == "risks":
                    analysis["risks"] = [risk.strip() for risk in content.split('\n') if risk.strip()]
                elif key == "testing":
                    analysis["testing_strategy"] = content
                elif key == "implementation":
                    analysis["implementation_approach"] = content
                elif key == "priority":
                    analysis["priority_recommendation"] = content
                else:
                    analysis[key] = content
    
    def _integrate_ai_insights(self, feature_details: FeatureDetails, ai_analysis: Dict[str, Any]):
        """Integrate AI insights into feature details."""
        # Update user benefit
        if ai_analysis.get("user_benefit"):
            feature_details.user_benefit = ai_analysis["user_benefit"]
        
        # Update implementation approach
        if ai_analysis.get("implementation_approach"):
            feature_details.implementation_approach = ai_analysis["implementation_approach"]
        
        # Add risks
        if ai_analysis.get("risks"):
            feature_details.risks.extend(ai_analysis["risks"][:3])  # Limit to 3 risks
        
        # Update testing considerations
        if ai_analysis.get("testing_strategy"):
            feature_details.testing_considerations.append(ai_analysis["testing_strategy"])
        
        # Update documentation needs
        if ai_analysis.get("documentation_needs"):
            feature_details.documentation_needs.extend(ai_analysis["documentation_needs"])
        
        # Add competitor analysis
        if ai_analysis.get("competitor_analysis"):
            feature_details.competitor_analysis = ai_analysis["competitor_analysis"]
        
        # Store AI insights in metadata
        feature_details.metadata["ai_insights"] = ai_analysis
    
    async def _validate_and_score(self, feature_details: FeatureDetails):
        """Validate feature details and calculate confidence score."""
        confidence_score = 0.0
        max_score = 10.0
        
        # Basic information completeness (3 points)
        if feature_details.feature_title and len(feature_details.feature_title) > 5:
            confidence_score += 1.0
        if feature_details.description and len(feature_details.description) > 20:
            confidence_score += 1.0
        if feature_details.category and feature_details.category != "other":
            confidence_score += 1.0
        
        # Use cases and requirements (3 points)
        if feature_details.use_cases:
            confidence_score += min(len(feature_details.use_cases) * 0.5, 1.0)
        if feature_details.user_stories:
            confidence_score += min(len(feature_details.user_stories) * 0.5, 1.0)
        if feature_details.acceptance_criteria:
            confidence_score += min(len(feature_details.acceptance_criteria) * 0.2, 1.0)
        
        # Business analysis (2 points)
        if feature_details.business_value:
            confidence_score += 1.0
        if feature_details.user_benefit:
            confidence_score += 1.0
        
        # Technical analysis (2 points)
        if feature_details.technical_requirements:
            confidence_score += 1.0
        if feature_details.effort_estimate and feature_details.complexity:
            confidence_score += 1.0
        
        # Normalize to 0-1 range
        feature_details.confidence_score = min(confidence_score / max_score, 1.0)
        
        # Additional validation
        if feature_details.confidence_score < self.confidence_threshold:
            feature_details.metadata["requires_review"] = True
            feature_details.metadata["validation_notes"] = "Low confidence score - requires manual review"
    
    def _update_statistics(self, feature_details_list: List[FeatureDetails]):
        """Update internal statistics."""
        self.extraction_stats["total_processed"] += len(feature_details_list)
        
        for feature in feature_details_list:
            if feature.confidence_score >= self.confidence_threshold:
                self.extraction_stats["successful_extractions"] += 1
            else:
                self.extraction_stats["failed_extractions"] += 1
            
            # Update category distribution
            self.extraction_stats["feature_categories"][feature.category] += 1
            
            # Update complexity distribution
            if feature.complexity:
                self.extraction_stats["complexity_distribution"][feature.complexity] += 1
        
        # Update average confidence
        if feature_details_list:
            total_confidence = sum(f.confidence_score for f in feature_details_list)
            new_avg = total_confidence / len(feature_details_list)
            
            total_items = self.extraction_stats["total_processed"]
            old_weight = (total_items - len(feature_details_list)) / total_items
            new_weight = len(feature_details_list) / total_items
            
            self.extraction_stats["average_confidence"] = (
                self.extraction_stats["average_confidence"] * old_weight + 
                new_avg * new_weight
            )
    
    async def extract_single_feature(self, feedback_item: FeedbackItem) -> FeatureDetails:
        """
        Extract feature details from a single feedback item (convenience method).
        
        Args:
            feedback_item: Feedback item to analyze
            
        Returns:
            FeatureDetails object
        """
        result = await self.process(feedback_item)
        if result.success and result.data and result.data["feature_details"]:
            return result.data["feature_details"][0]
        else:
            return FeatureDetails(
                feedback_id=feedback_item.id,
                feature_title="Extraction failed",
                description=feedback_item.content,
                confidence_score=0.0
            )
    
    async def batch_extract_features(self, feedback_items: List[FeedbackItem], 
                                   callback: Optional[callable] = None) -> List[FeatureDetails]:
        """
        Extract features from multiple feedback items with progress callback.
        
        Args:
            feedback_items: List of feedback items to analyze
            callback: Optional progress callback function
            
        Returns:
            List of FeatureDetails objects
        """
        all_features = []
        total_batches = (len(feedback_items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(feedback_items), self.batch_size):
            batch = feedback_items[batch_idx:batch_idx + self.batch_size]
            
            try:
                batch_features = await self._extract_features_batch(batch)
                all_features.extend(batch_features)
                
                # Call progress callback if provided
                if callback:
                    progress = (batch_idx // self.batch_size + 1) / total_batches
                    await callback(progress, len(all_features), len(feedback_items))
                    
            except Exception as e:
                self.logger.error(f"Error in batch feature extraction {batch_idx}: {str(e)}")
                continue
        
        return all_features
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get current extraction statistics."""
        stats = self.extraction_stats.copy()
        
        # Add percentage calculations
        total = stats["total_processed"]
        if total > 0:
            stats["success_rate"] = (stats["successful_extractions"] / total) * 100
            stats["failure_rate"] = (stats["failed_extractions"] / total) * 100
            
            # Convert category distribution to percentages
            category_percentages = {}
            for category, count in stats["feature_categories"].items():
                category_percentages[category] = (count / total) * 100
            stats["category_percentages"] = category_percentages
            
            # Convert complexity distribution to percentages
            complexity_percentages = {}
            for complexity, count in stats["complexity_distribution"].items():
                complexity_percentages[complexity] = (count / total) * 100
            stats["complexity_percentages"] = complexity_percentages
        
        return stats
    
    async def generate_feature_specification(self, feature_details: FeatureDetails) -> str:
        """
        Generate a comprehensive feature specification document.
        
        Args:
            feature_details: Feature details to document
            
        Returns:
            Formatted specification document
        """
        spec = f"""# Feature Specification: {feature_details.feature_title}

## Overview
**Category:** {feature_details.category.title()}
**Priority:** {feature_details.priority.value.title()}
**Effort Estimate:** {feature_details.effort_estimate or 'TBD'}
**Complexity:** {feature_details.complexity or 'TBD'}

## Description
{feature_details.description}

## Business Value
{feature_details.business_value or 'TBD'}

## User Benefit
{feature_details.user_benefit or 'TBD'}

## Use Cases
"""
        
        for i, use_case in enumerate(feature_details.use_cases, 1):
            spec += f"{i}. {use_case}\n"
        
        if feature_details.user_stories:
            spec += "\n## User Stories\n"
            for i, story in enumerate(feature_details.user_stories, 1):
                spec += f"{i}. {story}\n"
        
        if feature_details.acceptance_criteria:
            spec += "\n## Acceptance Criteria\n"
            for i, criteria in enumerate(feature_details.acceptance_criteria, 1):
                spec += f"{i}. {criteria}\n"
        
        if feature_details.technical_requirements:
            spec += "\n## Technical Requirements\n"
            for req in feature_details.technical_requirements:
                spec += f"- {req}\n"
        
        if feature_details.platforms_affected:
            platforms = [p.get_display_name() for p in feature_details.platforms_affected]
            spec += f"\n## Target Platforms\n{', '.join(platforms)}\n"
        
        if feature_details.dependencies:
            spec += "\n## Dependencies\n"
            for dep in feature_details.dependencies:
                spec += f"- {dep}\n"
        
        if feature_details.risks:
            spec += "\n## Risks and Mitigation\n"
            for risk in feature_details.risks:
                spec += f"- {risk}\n"
        
        if feature_details.testing_considerations:
            spec += "\n## Testing Strategy\n"
            for test in feature_details.testing_considerations:
                spec += f"- {test}\n"
        
        spec += f"\n## Metadata\n"
        spec += f"- **Confidence Score:** {feature_details.confidence_score:.2f}\n"
        spec += f"- **Extracted By:** {feature_details.extracted_by}\n"
        spec += f"- **Extraction Date:** {feature_details.extraction_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        spec += f"- **Source Feedback ID:** {feature_details.feedback_id}\n"
        
        return spec
    
    async def compare_features(self, feature1: FeatureDetails, feature2: FeatureDetails) -> Dict[str, Any]:
        """
        Compare two features and identify similarities/differences.
        
        Args:
            feature1: First feature to compare
            feature2: Second feature to compare
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "similarity_score": 0.0,
            "similarities": [],
            "differences": [],
            "recommendations": []
        }
        
        # Compare categories
        if feature1.category == feature2.category:
            comparison["similarities"].append(f"Both are {feature1.category} requests")
            comparison["similarity_score"] += 0.2
        else:
            comparison["differences"].append(f"Different categories: {feature1.category} vs {feature2.category}")
        
        # Compare complexity
        if feature1.complexity == feature2.complexity:
            comparison["similarities"].append(f"Similar complexity: {feature1.complexity}")
            comparison["similarity_score"] += 0.1
        
        # Compare platforms
        common_platforms = set(feature1.platforms_affected) & set(feature2.platforms_affected)
        if common_platforms:
            platforms = [p.get_display_name() for p in common_platforms]
            comparison["similarities"].append(f"Common platforms: {', '.join(platforms)}")
            comparison["similarity_score"] += 0.1
        
        # Compare use cases (text similarity)
        use_cases1 = ' '.join(feature1.use_cases).lower()
        use_cases2 = ' '.join(feature2.use_cases).lower()
        
        if use_cases1 and use_cases2:
            # Simple word overlap calculation
            words1 = set(use_cases1.split())
            words2 = set(use_cases2.split())
            overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
            comparison["similarity_score"] += overlap * 0.3
            
            if overlap > 0.3:
                comparison["similarities"].append("Similar use cases and functionality")
        
        # Generate recommendations
        if comparison["similarity_score"] > 0.6:
            comparison["recommendations"].append("Consider combining these features into a single implementation")
            comparison["recommendations"].append("Look for opportunities to share common components")
        elif comparison["similarity_score"] > 0.3:
            comparison["recommendations"].append("Consider implementing these features in the same release cycle")
            comparison["recommendations"].append("Evaluate shared dependencies and requirements")
        else:
            comparison["recommendations"].append("These features can be prioritized independently")
        
        return comparison
    
    def export_extraction_config(self) -> Dict[str, Any]:
        """Export current extraction configuration."""
        return {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "extract_use_cases": self.extract_use_cases,
                "estimate_complexity": self.estimate_complexity,
                "identify_dependencies": self.identify_dependencies,
                "confidence_threshold": self.confidence_threshold,
                "openai_model": self.openai_model,
                "temperature": self.temperature
            },
            "patterns": {
                "feature_patterns": self.feature_patterns,
                "complexity_indicators": self.complexity_indicators,
                "business_value_keywords": self.business_value_keywords
            },
            "templates": {
                "user_story_templates": self.user_story_templates,
                "acceptance_criteria_templates": self.acceptance_criteria_templates
            },
            "statistics": self.extraction_stats
        }