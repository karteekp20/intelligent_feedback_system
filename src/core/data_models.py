"""
Core data models for the Intelligent User Feedback Analysis System.
Contains all data structures, enums, and model classes used throughout the system.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import uuid


class FeedbackSource(Enum):
    """Enumeration of possible feedback sources."""
    
    APP_STORE = "app_store"
    GOOGLE_PLAY = "google_play"
    SUPPORT_EMAIL = "support_email"
    CUSTOMER_SURVEY = "customer_survey"
    SOCIAL_MEDIA = "social_media"
    LIVE_CHAT = "live_chat"
    PHONE_SUPPORT = "phone_support"
    COMMUNITY_FORUM = "community_forum"
    USER_TESTING = "user_testing"
    BETA_FEEDBACK = "beta_feedback"
    INTERNAL_TESTING = "internal_testing"
    PRODUCT_REVIEW = "product_review"
    WEBSITE_FEEDBACK = "website_feedback"
    API_FEEDBACK = "api_feedback"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, source_str: str) -> 'FeedbackSource':
        """
        Create FeedbackSource from string, with intelligent mapping.
        
        Args:
            source_str: String representation of the source
            
        Returns:
            FeedbackSource enum value
        """
        if not source_str:
            return cls.OTHER
        
        source_str = source_str.lower().strip()
        
        # Direct mapping
        for source in cls:
            if source.value == source_str:
                return source
        
        # Intelligent mapping based on keywords
        if any(keyword in source_str for keyword in ['app store', 'appstore', 'ios store']):
            return cls.APP_STORE
        elif any(keyword in source_str for keyword in ['google play', 'play store', 'android']):
            return cls.GOOGLE_PLAY
        elif any(keyword in source_str for keyword in ['email', 'support', 'ticket']):
            return cls.SUPPORT_EMAIL
        elif any(keyword in source_str for keyword in ['survey', 'questionnaire', 'poll']):
            return cls.CUSTOMER_SURVEY
        elif any(keyword in source_str for keyword in ['twitter', 'facebook', 'instagram', 'linkedin', 'social']):
            return cls.SOCIAL_MEDIA
        elif any(keyword in source_str for keyword in ['chat', 'live chat', 'webchat']):
            return cls.LIVE_CHAT
        elif any(keyword in source_str for keyword in ['phone', 'call', 'telephone']):
            return cls.PHONE_SUPPORT
        elif any(keyword in source_str for keyword in ['forum', 'community', 'discussion']):
            return cls.COMMUNITY_FORUM
        elif any(keyword in source_str for keyword in ['testing', 'usability', 'user test']):
            return cls.USER_TESTING
        elif any(keyword in source_str for keyword in ['beta', 'preview', 'early access']):
            return cls.BETA_FEEDBACK
        elif any(keyword in source_str for keyword in ['internal', 'qa', 'quality assurance']):
            return cls.INTERNAL_TESTING
        elif any(keyword in source_str for keyword in ['review', 'product review']):
            return cls.PRODUCT_REVIEW
        elif any(keyword in source_str for keyword in ['website', 'web', 'site']):
            return cls.WEBSITE_FEEDBACK
        elif any(keyword in source_str for keyword in ['api', 'integration', 'developer']):
            return cls.API_FEEDBACK
        else:
            return cls.OTHER
    
    def get_display_name(self) -> str:
        """Get human-readable display name for the source."""
        display_names = {
            self.APP_STORE: "App Store",
            self.GOOGLE_PLAY: "Google Play Store",
            self.SUPPORT_EMAIL: "Support Email",
            self.CUSTOMER_SURVEY: "Customer Survey",
            self.SOCIAL_MEDIA: "Social Media",
            self.LIVE_CHAT: "Live Chat",
            self.PHONE_SUPPORT: "Phone Support",
            self.COMMUNITY_FORUM: "Community Forum",
            self.USER_TESTING: "User Testing",
            self.BETA_FEEDBACK: "Beta Feedback",
            self.INTERNAL_TESTING: "Internal Testing",
            self.PRODUCT_REVIEW: "Product Review",
            self.WEBSITE_FEEDBACK: "Website Feedback",
            self.API_FEEDBACK: "API Feedback",
            self.OTHER: "Other"
        }
        return display_names.get(self, self.value.replace('_', ' ').title())
    
    def get_priority_weight(self) -> float:
        """
        Get priority weight for this source type.
        Higher weights indicate more critical sources.
        """
        weights = {
            self.APP_STORE: 0.9,  # High visibility
            self.GOOGLE_PLAY: 0.9,  # High visibility
            self.SUPPORT_EMAIL: 0.8,  # Direct customer contact
            self.PHONE_SUPPORT: 0.8,  # Direct customer contact
            self.CUSTOMER_SURVEY: 0.7,  # Structured feedback
            self.BETA_FEEDBACK: 0.7,  # Early insights
            self.LIVE_CHAT: 0.6,  # Real-time but may be less detailed
            self.USER_TESTING: 0.6,  # Controlled environment
            self.SOCIAL_MEDIA: 0.5,  # Public but may be less structured
            self.COMMUNITY_FORUM: 0.5,  # Community-driven
            self.WEBSITE_FEEDBACK: 0.4,  # General feedback
            self.PRODUCT_REVIEW: 0.4,  # Third-party reviews
            self.API_FEEDBACK: 0.6,  # Technical feedback
            self.INTERNAL_TESTING: 0.3,  # Internal perspective
            self.OTHER: 0.3  # Unknown source
        }
        return weights.get(self, 0.3)
    
    def is_public_facing(self) -> bool:
        """Check if this source is public-facing (visible to other customers)."""
        public_sources = {
            self.APP_STORE, self.GOOGLE_PLAY, self.SOCIAL_MEDIA,
            self.COMMUNITY_FORUM, self.PRODUCT_REVIEW
        }
        return self in public_sources
    
    def requires_response(self) -> bool:
        """Check if this source typically requires a response."""
        response_required = {
            self.SUPPORT_EMAIL, self.PHONE_SUPPORT, self.LIVE_CHAT,
            self.CUSTOMER_SURVEY, self.BETA_FEEDBACK
        }
        return self in response_required
class Priority(Enum):
    """Priority levels for tickets and feedback."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    def get_numeric_value(self) -> int:
        """Get numeric representation for sorting."""
        values = {
            self.CRITICAL: 4,
            self.HIGH: 3,
            self.MEDIUM: 2,
            self.LOW: 1
        }
        return values[self]
    
    @classmethod
    def from_numeric(cls, value: int) -> 'Priority':
        """Create Priority from numeric value."""
        mapping = {4: cls.CRITICAL, 3: cls.HIGH, 2: cls.MEDIUM, 1: cls.LOW}
        return mapping.get(value, cls.MEDIUM)



class Platform(Enum):
    """Enumeration of supported platforms and environments."""
    
    # Mobile Platforms
    IOS = "ios"
    ANDROID = "android"
    
    # Desktop Platforms
    DESKTOP = "desktop"  
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    
    # Web Platforms
    WEB = "web"
    WEB_CHROME = "web_chrome"
    WEB_FIREFOX = "web_firefox"
    WEB_SAFARI = "web_safari"
    WEB_EDGE = "web_edge"
    WEB_OTHER = "web_other"
    
    # Gaming Platforms
    XBOX = "xbox"
    PLAYSTATION = "playstation"
    NINTENDO_SWITCH = "nintendo_switch"
    PC_GAMING = "pc_gaming"
    
    # IoT and Smart Devices
    SMART_TV = "smart_tv"
    SMART_WATCH = "smart_watch"
    TABLET = "tablet"
    
    # Cloud and Server
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"
    
    # Development Environments
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    
    # API and Integration
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    WEBHOOK = "webhook"
    
    # Other
    UNKNOWN = "unknown"
    MULTIPLE = "multiple"
    
    @classmethod
    def from_string(cls, platform_str: str) -> 'Platform':
        """Create Platform from string with intelligent detection."""
        if not platform_str:
            return cls.UNKNOWN
        
        platform_str = platform_str.lower().strip()
        
        # Direct mapping
        for platform in cls:
            if platform.value == platform_str:
                return platform
        
        # Intelligent mapping
        if any(keyword in platform_str for keyword in ['ios', 'iphone', 'ipad', 'apple']):
            return cls.IOS
        elif any(keyword in platform_str for keyword in ['android', 'google play']):
            return cls.ANDROID
        elif any(keyword in platform_str for keyword in ['windows', 'win10', 'win11', 'pc']):
            return cls.WINDOWS
        elif any(keyword in platform_str for keyword in ['mac', 'macos', 'osx']):
            return cls.MACOS
        elif any(keyword in platform_str for keyword in ['linux', 'ubuntu', 'debian']):
            return cls.LINUX
        elif any(keyword in platform_str for keyword in ['chrome', 'chromium']):
            return cls.WEB_CHROME
        elif any(keyword in platform_str for keyword in ['firefox', 'mozilla']):
            return cls.WEB_FIREFOX
        elif any(keyword in platform_str for keyword in ['safari', 'webkit']):
            return cls.WEB_SAFARI
        elif any(keyword in platform_str for keyword in ['edge', 'ie', 'internet explorer']):
            return cls.WEB_EDGE
        elif any(keyword in platform_str for keyword in ['web', 'browser', 'http']):
            return cls.WEB_OTHER
        elif any(keyword in platform_str for keyword in ['xbox', 'microsoft console']):
            return cls.XBOX
        elif any(keyword in platform_str for keyword in ['playstation', 'ps4', 'ps5', 'sony']):
            return cls.PLAYSTATION
        elif any(keyword in platform_str for keyword in ['nintendo', 'switch']):
            return cls.NINTENDO_SWITCH
        elif any(keyword in platform_str for keyword in ['tablet', 'ipad']):
            return cls.TABLET
        elif any(keyword in platform_str for keyword in ['api', 'rest', 'graphql']):
            return cls.REST_API
        else:
            return cls.UNKNOWN
    
    def get_display_name(self) -> str:
        """Get human-readable display name."""
        display_names = {
            self.IOS: "iOS",
            self.ANDROID: "Android",
            self.WINDOWS: "Windows",
            self.MACOS: "macOS",
            self.LINUX: "Linux",
            self.WEB_CHROME: "Chrome Browser",
            self.WEB_FIREFOX: "Firefox Browser",
            self.WEB_SAFARI: "Safari Browser",
            self.WEB_EDGE: "Edge Browser",
            self.WEB_OTHER: "Other Browser",
            self.XBOX: "Xbox",
            self.PLAYSTATION: "PlayStation",
            self.NINTENDO_SWITCH: "Nintendo Switch",
            self.PC_GAMING: "PC Gaming",
            self.SMART_TV: "Smart TV",
            self.SMART_WATCH: "Smart Watch",
            self.TABLET: "Tablet",
            self.AWS: "Amazon AWS",
            self.AZURE: "Microsoft Azure",
            self.GCP: "Google Cloud Platform",
            self.ON_PREMISE: "On-Premise",
            self.DEVELOPMENT: "Development Environment",
            self.STAGING: "Staging Environment",
            self.PRODUCTION: "Production Environment",
            self.REST_API: "REST API",
            self.GRAPHQL_API: "GraphQL API",
            self.WEBHOOK: "Webhook",
            self.UNKNOWN: "Unknown Platform",
            self.MULTIPLE: "Multiple Platforms"
        }
        return display_names.get(self, self.value.replace('_', ' ').title())
    
    def is_mobile(self) -> bool:
        """Check if this is a mobile platform."""
        return self in {self.IOS, self.ANDROID, self.SMART_WATCH}
    
    def is_web(self) -> bool:
        """Check if this is a web platform."""
        return self in {self.WEB, self.WEB_CHROME, self.WEB_FIREFOX, self.WEB_SAFARI, self.WEB_EDGE, self.WEB_OTHER}
    
    def is_desktop(self) -> bool:
        """Check if this is a desktop platform."""
        return self in {self.WINDOWS, self.MACOS, self.LINUX}
    
    def is_gaming(self) -> bool:
        """Check if this is a gaming platform."""
        return self in {self.XBOX, self.PLAYSTATION, self.NINTENDO_SWITCH, self.PC_GAMING}
    
    def is_cloud(self) -> bool:
        """Check if this is a cloud platform."""
        return self in {self.AWS, self.AZURE, self.GCP}


@dataclass
class BugDetails:
    """Detailed information about a bug extracted from feedback."""
    
    feedback_id: str
    bug_title: str
    description: str
    severity: Priority = Priority.MEDIUM
    platform: Platform = Platform.UNKNOWN
    affected_versions: List[str] = field(default_factory=list)
    steps_to_reproduce: List[str] = field(default_factory=list)
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    error_messages: List[str] = field(default_factory=list)
    stack_traces: List[str] = field(default_factory=list)
    log_entries: List[str] = field(default_factory=list)
    screenshots_mentioned: bool = False
    video_mentioned: bool = False
    frequency: Optional[str] = None  # "always", "sometimes", "rarely", "once"
    user_environment: Dict[str, Any] = field(default_factory=dict)
    workaround_available: bool = False
    workaround_description: Optional[str] = None
    impact_on_users: Optional[str] = None
    business_impact: Optional[str] = None
    related_features: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    extracted_by: str = "bug_analysis_agent"
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and processing."""
        # Validate confidence score
        if not 0.0 <= self.confidence_score <= 1.0:
            self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        
        # Auto-generate title if not provided
        if not self.bug_title and self.description:
            self.bug_title = self._generate_title()
        
        # Set default frequency if steps are provided
        if self.steps_to_reproduce and not self.frequency:
            self.frequency = "reproducible"
        
        # Extract error patterns
        self._extract_error_patterns()
    
    def _generate_title(self) -> str:
        """Generate a concise title from the description."""
        # Take first sentence or first 80 characters
        title = self.description.split('.')[0].strip()
        if len(title) > 80:
            title = title[:77] + "..."
        
        # Add platform info if available
        if self.platform != Platform.UNKNOWN:
            title = f"[{self.platform.get_display_name()}] {title}"
        
        return title
    
    def _extract_error_patterns(self):
        """Extract common error patterns and categorize them."""
        error_patterns = {
            'crash': ['crash', 'crashed', 'crashing', 'segfault', 'core dump'],
            'freeze': ['freeze', 'frozen', 'hang', 'unresponsive', 'stuck'],
            'performance': ['slow', 'lag', 'performance', 'timeout', 'delay'],
            'ui': ['button', 'click', 'display', 'layout', 'rendering'],
            'data': ['data', 'save', 'load', 'sync', 'database'],
            'network': ['network', 'connection', 'api', 'request', 'response'],
            'authentication': ['login', 'auth', 'password', 'token', 'session']
        }
        
        text = f"{self.description} {' '.join(self.error_messages)}".lower()
        
        for pattern_type, keywords in error_patterns.items():
            if any(keyword in text for keyword in keywords):
                if pattern_type not in self.tags:
                    self.tags.append(pattern_type)
    
    def get_severity_suggestion(self) -> Priority:
        """Suggest severity based on bug characteristics."""
        # Critical conditions
        critical_indicators = [
            'crash', 'data loss', 'security', 'cannot start', 'system down',
            'corruption', 'vulnerability', 'exploit'
        ]
        
        # High priority conditions
        high_indicators = [
            'major feature', 'blocking', 'cannot use', 'broken',
            'incorrect data', 'performance degradation'
        ]
        
        # Medium priority conditions
        medium_indicators = [
            'minor feature', 'workaround available', 'cosmetic',
            'inconsistent', 'confusing'
        ]
        
        text = f"{self.description} {' '.join(self.error_messages)}".lower()
        
        if any(indicator in text for indicator in critical_indicators):
            return Priority.CRITICAL
        elif any(indicator in text for indicator in high_indicators):
            return Priority.HIGH
        elif any(indicator in text for indicator in medium_indicators):
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def is_reproducible(self) -> bool:
        """Check if the bug appears to be reproducible."""
        return (
            bool(self.steps_to_reproduce) or
            self.frequency in ["always", "reproducible"] or
            "reproduce" in self.description.lower()
        )
    
    def get_complexity_estimate(self) -> str:
        """Estimate fix complexity based on bug characteristics."""
        complexity_factors = {
            'simple': ['ui', 'cosmetic', 'text', 'label', 'color'],
            'medium': ['validation', 'form', 'display', 'calculation'],
            'complex': ['algorithm', 'performance', 'database', 'integration'],
            'very_complex': ['architecture', 'security', 'data migration', 'core system']
        }
        
        text = f"{self.description} {' '.join(self.tags)}".lower()
        
        for complexity, indicators in complexity_factors.items():
            if any(indicator in text for indicator in indicators):
                return complexity
        
        return "medium"  # Default
    
    def add_reproduction_step(self, step: str):
        """Add a step to reproduce the bug."""
        if step and step not in self.steps_to_reproduce:
            self.steps_to_reproduce.append(step)
    
    def add_error_message(self, error: str):
        """Add an error message."""
        if error and error not in self.error_messages:
            self.error_messages.append(error)
    
    def add_stack_trace(self, trace: str):
        """Add a stack trace."""
        if trace and trace not in self.stack_traces:
            self.stack_traces.append(trace)
    
    def set_workaround(self, description: str):
        """Set workaround information."""
        self.workaround_available = True
        self.workaround_description = description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'feedback_id': self.feedback_id,
            'bug_title': self.bug_title,
            'description': self.description,
            'severity': self.severity.value,
            'platform': self.platform.value,
            'affected_versions': self.affected_versions,
            'steps_to_reproduce': self.steps_to_reproduce,
            'expected_behavior': self.expected_behavior,
            'actual_behavior': self.actual_behavior,
            'error_messages': self.error_messages,
            'stack_traces': self.stack_traces,
            'log_entries': self.log_entries,
            'screenshots_mentioned': self.screenshots_mentioned,
            'video_mentioned': self.video_mentioned,
            'frequency': self.frequency,
            'user_environment': self.user_environment,
            'workaround_available': self.workaround_available,
            'workaround_description': self.workaround_description,
            'impact_on_users': self.impact_on_users,
            'business_impact': self.business_impact,
            'related_features': self.related_features,
            'tags': self.tags,
            'confidence_score': self.confidence_score,
            'extracted_by': self.extracted_by,
            'extraction_timestamp': self.extraction_timestamp.isoformat(),
            'metadata': self.metadata,
            'severity_suggestion': self.get_severity_suggestion().value,
            'is_reproducible': self.is_reproducible(),
            'complexity_estimate': self.get_complexity_estimate()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BugDetails':
        """Create BugDetails from dictionary."""
        return cls(
            feedback_id=data['feedback_id'],
            bug_title=data['bug_title'],
            description=data['description'],
            severity=Priority(data.get('severity', 'medium')),
            platform=Platform(data.get('platform', 'unknown')),
            affected_versions=data.get('affected_versions', []),
            steps_to_reproduce=data.get('steps_to_reproduce', []),
            expected_behavior=data.get('expected_behavior'),
            actual_behavior=data.get('actual_behavior'),
            error_messages=data.get('error_messages', []),
            stack_traces=data.get('stack_traces', []),
            log_entries=data.get('log_entries', []),
            screenshots_mentioned=data.get('screenshots_mentioned', False),
            video_mentioned=data.get('video_mentioned', False),
            frequency=data.get('frequency'),
            user_environment=data.get('user_environment', {}),
            workaround_available=data.get('workaround_available', False),
            workaround_description=data.get('workaround_description'),
            impact_on_users=data.get('impact_on_users'),
            business_impact=data.get('business_impact'),
            related_features=data.get('related_features', []),
            tags=data.get('tags', []),
            confidence_score=data.get('confidence_score', 0.0),
            extracted_by=data.get('extracted_by', 'bug_analysis_agent'),
            extraction_timestamp=datetime.fromisoformat(
                data.get('extraction_timestamp', datetime.now().isoformat())
            ),
            metadata=data.get('metadata', {})
        )


@dataclass
class FeatureDetails:
    """Detailed information about a feature request extracted from feedback."""
    
    feedback_id: str
    feature_title: str
    description: str
    category: str = "enhancement"  # enhancement, new_feature, improvement
    priority: Priority = Priority.MEDIUM
    use_cases: List[str] = field(default_factory=list)
    user_stories: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    business_value: Optional[str] = None
    user_benefit: Optional[str] = None
    target_users: List[str] = field(default_factory=list)
    effort_estimate: Optional[str] = None  # small, medium, large, xl
    complexity: Optional[str] = None  # low, medium, high, very_high
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    alternatives_considered: List[str] = field(default_factory=list)
    technical_requirements: List[str] = field(default_factory=list)
    ui_ux_requirements: List[str] = field(default_factory=list)
    performance_requirements: List[str] = field(default_factory=list)
    security_considerations: List[str] = field(default_factory=list)
    platforms_affected: List[Platform] = field(default_factory=list)
    related_features: List[str] = field(default_factory=list)
    competitor_analysis: Optional[str] = None
    market_demand: Optional[str] = None  # low, medium, high, very_high
    implementation_approach: Optional[str] = None
    testing_considerations: List[str] = field(default_factory=list)
    documentation_needs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    extracted_by: str = "feature_extractor_agent"
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and processing."""
        # Validate confidence score
        if not 0.0 <= self.confidence_score <= 1.0:
            self.confidence_score = max(0.0, min(1.0, self.confidence_score))
        
        # Auto-generate title if not provided
        if not self.feature_title and self.description:
            self.feature_title = self._generate_title()
        
        # Extract feature patterns and tags
        self._extract_feature_patterns()
        
        # Auto-estimate effort if not provided
        if not self.effort_estimate:
            self.effort_estimate = self._estimate_effort()
    
    def _generate_title(self) -> str:
        """Generate a concise title from the description."""
        # Take first sentence or first 80 characters
        title = self.description.split('.')[0].strip()
        if len(title) > 80:
            title = title[:77] + "..."
        
        # Add category prefix
        if self.category == "new_feature":
            title = f"[New Feature] {title}"
        elif self.category == "improvement":
            title = f"[Improvement] {title}"
        else:
            title = f"[Enhancement] {title}"
        
        return title
    
    def _extract_feature_patterns(self):
        """Extract common feature patterns and categorize them."""
        feature_patterns = {
            'ui_ux': ['interface', 'design', 'layout', 'button', 'menu', 'theme'],
            'integration': ['api', 'connect', 'sync', 'import', 'export', 'webhook'],
            'automation': ['automatic', 'auto', 'schedule', 'batch', 'bulk'],
            'analytics': ['report', 'analytics', 'dashboard', 'chart', 'metrics'],
            'search': ['search', 'filter', 'find', 'query', 'sort'],
            'notification': ['notify', 'alert', 'reminder', 'email', 'push'],
            'collaboration': ['share', 'team', 'collaborate', 'comment', 'review'],
            'mobile': ['mobile', 'phone', 'app', 'touch', 'gesture'],
            'performance': ['fast', 'speed', 'optimize', 'cache', 'performance'],
            'security': ['security', 'privacy', 'encrypt', 'auth', 'permission']
        }
        
        text = f"{self.description} {' '.join(self.use_cases)}".lower()
        
        for pattern_type, keywords in feature_patterns.items():
            if any(keyword in text for keyword in keywords):
                if pattern_type not in self.tags:
                    self.tags.append(pattern_type)
    
    def _estimate_effort(self) -> str:
        """Estimate development effort based on feature characteristics."""
        effort_indicators = {
            'small': ['button', 'color', 'text', 'label', 'icon', 'tooltip'],
            'medium': ['form', 'page', 'filter', 'sort', 'validation', 'format'],
            'large': ['integration', 'api', 'workflow', 'automation', 'report'],
            'xl': ['architecture', 'system', 'platform', 'framework', 'migration']
        }
        
        text = f"{self.description} {' '.join(self.technical_requirements)}".lower()
        
        # Count complexity indicators
        complexity_score = 0
        complexity_score += len(self.dependencies) * 2
        complexity_score += len(self.technical_requirements)
        complexity_score += len(self.platforms_affected)
        
        # Check text indicators
        for effort, indicators in effort_indicators.items():
            if any(indicator in text for indicator in indicators):
                if effort == 'xl':
                    return 'xl'
                elif effort == 'large':
                    complexity_score += 5
                elif effort == 'medium':
                    complexity_score += 3
                else:
                    complexity_score += 1
        
        # Determine effort based on score
        if complexity_score >= 10:
            return 'xl'
        elif complexity_score >= 6:
            return 'large'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'small'
    
    def get_priority_suggestion(self) -> Priority:
        """Suggest priority based on feature characteristics."""
        priority_score = 0
        
        # Business value indicators
        if self.business_value and any(word in self.business_value.lower() 
                                     for word in ['revenue', 'critical', 'competitive']):
            priority_score += 3
        
        # User demand indicators
        if self.market_demand == 'very_high':
            priority_score += 3
        elif self.market_demand == 'high':
            priority_score += 2
        elif self.market_demand == 'medium':
            priority_score += 1
        
        # Effort vs value
        if self.effort_estimate == 'small' and len(self.use_cases) > 2:
            priority_score += 2
        
        # Dependencies and risks
        if len(self.dependencies) > 3 or len(self.risks) > 2:
            priority_score -= 1
        
        # Determine priority
        if priority_score >= 5:
            return Priority.CRITICAL
        elif priority_score >= 3:
            return Priority.HIGH
        elif priority_score >= 1:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    def add_use_case(self, use_case: str):
        """Add a use case for the feature."""
        if use_case and use_case not in self.use_cases:
            self.use_cases.append(use_case)
    
    def add_user_story(self, story: str):
        """Add a user story."""
        if story and story not in self.user_stories:
            self.user_stories.append(story)
    
    def add_acceptance_criteria(self, criteria: str):
        """Add acceptance criteria."""
        if criteria and criteria not in self.acceptance_criteria:
            self.acceptance_criteria.append(criteria)
    
    def add_dependency(self, dependency: str):
        """Add a dependency."""
        if dependency and dependency not in self.dependencies:
            self.dependencies.append(dependency)
    
    def add_risk(self, risk: str):
        """Add a risk consideration."""
        if risk and risk not in self.risks:
            self.risks.append(risk)
    
    def set_business_value(self, value: str):
        """Set business value description."""
        self.business_value = value
    
    def set_user_benefit(self, benefit: str):
        """Set user benefit description."""
        self.user_benefit = benefit
    
    def get_roi_estimate(self) -> str:
        """Estimate return on investment."""
        if not self.business_value or not self.effort_estimate:
            return "unknown"
        
        # Simple ROI calculation based on business value keywords and effort
        value_score = 0
        if any(word in self.business_value.lower() 
               for word in ['revenue', 'profit', 'sales']):
            value_score += 3
        if any(word in self.business_value.lower() 
               for word in ['efficiency', 'productivity', 'time']):
            value_score += 2
        if any(word in self.business_value.lower() 
               for word in ['satisfaction', 'retention', 'engagement']):
            value_score += 1
        
        effort_map = {'small': 1, 'medium': 2, 'large': 3, 'xl': 4}
        effort_score = effort_map.get(self.effort_estimate, 2)
        
        roi_ratio = value_score / effort_score if effort_score > 0 else 0
        
        if roi_ratio >= 2:
            return "high"
        elif roi_ratio >= 1:
            return "medium"
        elif roi_ratio >= 0.5:
            return "low"
        else:
            return "very_low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'feedback_id': self.feedback_id,
            'feature_title': self.feature_title,
            'description': self.description,
            'category': self.category,
            'priority': self.priority.value,
            'use_cases': self.use_cases,
            'user_stories': self.user_stories,
            'acceptance_criteria': self.acceptance_criteria,
            'business_value': self.business_value,
            'user_benefit': self.user_benefit,
            'target_users': self.target_users,
            'effort_estimate': self.effort_estimate,
            'complexity': self.complexity,
            'dependencies': self.dependencies,
            'risks': self.risks,
            'alternatives_considered': self.alternatives_considered,
            'technical_requirements': self.technical_requirements,
            'ui_ux_requirements': self.ui_ux_requirements,
            'performance_requirements': self.performance_requirements,
            'security_considerations': self.security_considerations,
            'platforms_affected': [p.value for p in self.platforms_affected],
            'related_features': self.related_features,
            'competitor_analysis': self.competitor_analysis,
            'market_demand': self.market_demand,
            'implementation_approach': self.implementation_approach,
            'testing_considerations': self.testing_considerations,
            'documentation_needs': self.documentation_needs,
            'tags': self.tags,
            'confidence_score': self.confidence_score,
            'extracted_by': self.extracted_by,
            'extraction_timestamp': self.extraction_timestamp.isoformat(),
            'metadata': self.metadata,
            'priority_suggestion': self.get_priority_suggestion().value,
            'roi_estimate': self.get_roi_estimate()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureDetails':
        """Create FeatureDetails from dictionary."""
        platforms_affected = []
        if 'platforms_affected' in data:
            platforms_affected = [Platform(p) for p in data['platforms_affected']]
        
        return cls(
            feedback_id=data['feedback_id'],
            feature_title=data['feature_title'],
            description=data['description'],
            category=data.get('category', 'enhancement'),
            priority=Priority(data.get('priority', 'medium')),
            use_cases=data.get('use_cases', []),
            user_stories=data.get('user_stories', []),
            acceptance_criteria=data.get('acceptance_criteria', []),
            business_value=data.get('business_value'),
            user_benefit=data.get('user_benefit'),
            target_users=data.get('target_users', []),
            effort_estimate=data.get('effort_estimate'),
            complexity=data.get('complexity'),
            dependencies=data.get('dependencies', []),
            risks=data.get('risks', []),
            alternatives_considered=data.get('alternatives_considered', []),
            technical_requirements=data.get('technical_requirements', []),
            ui_ux_requirements=data.get('ui_ux_requirements', []),
            performance_requirements=data.get('performance_requirements', []),
            security_considerations=data.get('security_considerations', []),
            platforms_affected=platforms_affected,
            related_features=data.get('related_features', []),
            competitor_analysis=data.get('competitor_analysis'),
            market_demand=data.get('market_demand'),
            implementation_approach=data.get('implementation_approach'),
            testing_considerations=data.get('testing_considerations', []),
            documentation_needs=data.get('documentation_needs', []),
            tags=data.get('tags', []),
            confidence_score=data.get('confidence_score', 0.0),
            extracted_by=data.get('extracted_by', 'feature_extractor_agent'),
            extraction_timestamp=datetime.fromisoformat(
                data.get('extraction_timestamp', datetime.now().isoformat())
            ),
            metadata=data.get('metadata', {})
        )


class FeedbackCategory(Enum):
    """Categories for classifying user feedback."""
    
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    PERFORMANCE_ISSUE = "performance_issue"
    UI_UX_ISSUE = "ui_ux_issue"
    SECURITY_CONCERN = "security_concern"
    INTEGRATION_ISSUE = "integration_issue"
    DOCUMENTATION_REQUEST = "documentation_request"
    PRAISE = "praise"
    COMPLAINT = "complaint"
    QUESTION = "question"
    SPAM = "spam"
    OTHER = "other"
    
    @classmethod
    def from_string(cls, category_str: str) -> 'FeedbackCategory':
        """Create FeedbackCategory from string with intelligent mapping."""
        if not category_str:
            return cls.OTHER
        
        category_str = category_str.lower().strip()
        
        # Direct mapping
        for category in cls:
            if category.value == category_str:
                return category
        
        # Intelligent mapping
        if any(keyword in category_str for keyword in ['bug', 'error', 'crash', 'broken', 'issue', 'problem']):
            return cls.BUG
        elif any(keyword in category_str for keyword in ['feature', 'enhancement', 'improvement', 'add', 'request']):
            return cls.FEATURE_REQUEST
        elif any(keyword in category_str for keyword in ['slow', 'performance', 'speed', 'lag', 'loading']):
            return cls.PERFORMANCE_ISSUE
        elif any(keyword in category_str for keyword in ['ui', 'ux', 'interface', 'design', 'usability']):
            return cls.UI_UX_ISSUE
        elif any(keyword in category_str for keyword in ['security', 'privacy', 'vulnerability', 'hack']):
            return cls.SECURITY_CONCERN
        elif any(keyword in category_str for keyword in ['integration', 'api', 'connect', 'sync']):
            return cls.INTEGRATION_ISSUE
        elif any(keyword in category_str for keyword in ['documentation', 'docs', 'help', 'guide', 'tutorial']):
            return cls.DOCUMENTATION_REQUEST
        elif any(keyword in category_str for keyword in ['great', 'excellent', 'love', 'amazing', 'perfect', 'praise']):
            return cls.PRAISE
        elif any(keyword in category_str for keyword in ['hate', 'terrible', 'awful', 'worst', 'complaint']):
            return cls.COMPLAINT
        elif any(keyword in category_str for keyword in ['question', 'how', 'what', 'why', 'when', 'where']):
            return cls.QUESTION
        elif any(keyword in category_str for keyword in ['spam', 'advertisement', 'promotion']):
            return cls.SPAM
        else:
            return cls.OTHER


class ProcessingStatus(Enum):
    """Status of feedback processing."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class FeedbackItem:
    """Represents a single piece of user feedback."""
    
    id: str
    content: str
    source: FeedbackSource
    timestamp: datetime
    category: Optional[FeedbackCategory] = None
    priority: Optional[Priority] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure timestamps are datetime objects
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        
        # Generate ID if not provided
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Update modified timestamp
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'content': self.content,
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value if self.category else None,
            'priority': self.priority.value if self.priority else None,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata,
            'processing_status': self.processing_status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create FeedbackItem from dictionary."""
        return cls(
            id=data['id'],
            content=data['content'],
            source=FeedbackSource(data['source']),
            timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
            category=FeedbackCategory(data['category']) if data.get('category') else None,
            priority=Priority(data['priority']) if data.get('priority') else None,
            confidence_score=data.get('confidence_score'),
            metadata=data.get('metadata', {}),
            processing_status=ProcessingStatus(data.get('processing_status', 'pending')),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        )
    
    def update_status(self, status: ProcessingStatus):
        """Update processing status and timestamp."""
        self.processing_status = status
        self.updated_at = datetime.now()
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata entry."""
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def get_display_title(self) -> str:
        """Get a display-friendly title for the feedback."""
        # Truncate content for title
        max_length = 100
        title = self.content.strip()
        if len(title) > max_length:
            title = title[:max_length] + "..."
        
        # Add source and category info
        source_name = self.source.get_display_name()
        category_name = self.category.value.replace('_', ' ').title() if self.category else "Uncategorized"
        
        return f"[{source_name}] [{category_name}] {title}"


@dataclass
class Ticket:
    """Represents a generated ticket from feedback analysis."""
    
    id: str
    title: str
    description: str
    category: FeedbackCategory
    priority: Priority
    assigned_team: Optional[str] = None
    source_feedback_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "Open"
    source_id: Optional[str] = None  # ← ADD: item.id
    source: Optional[FeedbackSource] = None  # ← ADD: item.source
    technical_details: Dict[str, Any] = field(default_factory=dict)  # ← ADD
    agent_confidence: Optional[float] = None  # ← ADD: classification.confidence
    tags: List[str] = field(default_factory=list)  # ← ADD
    estimated_effort: Optional[str] = None  # ← ADD


    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            self.id = f"TICKET-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category.value,
            'priority': self.priority.value,
            'assigned_team': self.assigned_team,
            'source_feedback_ids': self.source_feedback_ids,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status,
            'source_id': self.source_id,
            'source': self.source.value if self.source else None,
            'technical_details': self.technical_details,
            'agent_confidence': self.agent_confidence,
            'tags': self.tags,
            'estimated_effort': self.estimated_effort,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Ticket':
        """Create Ticket from dictionary."""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            category=FeedbackCategory(data['category']),
            priority=Priority(data['priority']),
            assigned_team=data.get('assigned_team'),
            source_feedback_ids=data.get('source_feedback_ids', []),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            status=data.get('status', 'Open'),
            source_id=data.get('source_id'),
            source=FeedbackSource(data['source']) if data.get('source') else None,
            technical_details=data.get('technical_details', {}),
            agent_confidence=data.get('agent_confidence'),
            tags=data.get('tags', []),
            estimated_effort=data.get('estimated_effort'),
        )


@dataclass
class Classification:
    """Represents the classification result for a piece of feedback."""
    
    feedback_id: str
    predicted_category: FeedbackCategory
    confidence_score: float
    category_scores: Dict[FeedbackCategory, float] = field(default_factory=dict)
    reasoning: Optional[str] = None
    keywords_found: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1.0 (negative) to 1.0 (positive)
    emotion_detected: Optional[str] = None  # anger, joy, fear, sadness, etc.
    urgency_level: Optional[str] = None  # low, medium, high, critical
    is_actionable: bool = True
    requires_human_review: bool = False
    classifier_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    sentiment: Optional[str] = "neutral"  # ← THIS IS THE MISSING FIELD!
    
    def __post_init__(self):
        """Post-initialization validation and processing."""
        # Validate confidence score
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        
        # Validate sentiment score if provided
        if self.sentiment_score is not None and not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        
        # Set requires_human_review based on confidence
        if self.confidence_score < 0.7:
            self.requires_human_review = True
        
        # Validate category scores sum to approximately 1.0 if provided
        if self.category_scores:
            total_score = sum(self.category_scores.values())
            if not 0.95 <= total_score <= 1.05:  # Allow small floating point errors
                # Normalize scores
                self.category_scores = {
                    cat: score / total_score 
                    for cat, score in self.category_scores.items()
                }

    @property
    def keywords(self) -> List[str]:
        """Backward compatibility for keywords access."""
        return self.keywords_found
    
    @property
    def confidence(self) -> float:
        """Backward compatibility for confidence access."""
        return self.confidence_score
    
    def get_alternative_categories(self, min_score: float = 0.1) -> List[tuple]:
        """
        Get alternative categories with scores above threshold.
        
        Args:
            min_score: Minimum score threshold for alternatives
            
        Returns:
            List of (category, score) tuples sorted by score descending
        """
        if not self.category_scores:
            return []
        
        alternatives = [
            (cat, score) for cat, score in self.category_scores.items()
            if cat != self.predicted_category and score >= min_score
        ]
        
        return sorted(alternatives, key=lambda x: x[1], reverse=True)
    
    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.confidence_score >= 0.9:
            return "Very High"
        elif self.confidence_score >= 0.8:
            return "High"
        elif self.confidence_score >= 0.7:
            return "Medium"
        elif self.confidence_score >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def get_sentiment_label(self) -> str:
        """Get human-readable sentiment label."""
        if self.sentiment_score is None:
            return "Unknown"
        elif self.sentiment_score >= 0.5:
            return "Very Positive"
        elif self.sentiment_score >= 0.1:
            return "Positive"
        elif self.sentiment_score >= -0.1:
            return "Neutral"
        elif self.sentiment_score >= -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def should_escalate(self) -> bool:
        """
        Determine if this classification should be escalated for review.
        
        Returns:
            True if escalation is recommended
        """
        escalation_conditions = [
            self.confidence_score < 0.6,  # Low confidence
            self.predicted_category == FeedbackCategory.SECURITY_CONCERN,  # Security issues
            self.urgency_level == "critical",  # Critical urgency
            self.sentiment_score is not None and self.sentiment_score < -0.7,  # Very negative
            self.emotion_detected in ["anger", "frustration", "disappointment"],  # Negative emotions
            len(self.get_alternative_categories(0.3)) >= 2,  # Multiple viable alternatives
        ]
        
        return any(escalation_conditions) or self.requires_human_review
    
    def get_priority_suggestion(self) -> Priority:
        """
        Suggest priority level based on classification results.
        
        Returns:
            Suggested Priority enum value
        """
        # Critical conditions
        if (self.predicted_category == FeedbackCategory.SECURITY_CONCERN or
            self.urgency_level == "critical" or
            (self.sentiment_score is not None and self.sentiment_score < -0.8)):
            return Priority.CRITICAL
        
        # High priority conditions
        if (self.predicted_category == FeedbackCategory.BUG or
            self.urgency_level == "high" or
            (self.sentiment_score is not None and self.sentiment_score < -0.5)):
            return Priority.HIGH
        
        # Medium priority conditions
        if (self.predicted_category in [FeedbackCategory.PERFORMANCE_ISSUE, 
                                       FeedbackCategory.UI_UX_ISSUE,
                                       FeedbackCategory.INTEGRATION_ISSUE] or
            self.urgency_level == "medium"):
            return Priority.MEDIUM
        
        # Default to low priority
        return Priority.LOW
    
    def add_keyword(self, keyword: str, weight: float = 1.0):
        """
        Add a keyword that influenced the classification.
        
        Args:
            keyword: The keyword found
            weight: Importance weight of the keyword
        """
        if keyword not in self.keywords_found:
            self.keywords_found.append(keyword)
            self.metadata[f"keyword_{keyword}_weight"] = weight
    
    def update_reasoning(self, additional_reasoning: str):
        """Add additional reasoning to the classification."""
        if self.reasoning:
            self.reasoning += f" {additional_reasoning}"
        else:
            self.reasoning = additional_reasoning
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'feedback_id': self.feedback_id,
            'predicted_category': self.predicted_category.value,
            'confidence_score': self.confidence_score,
            'category_scores': {cat.value: score for cat, score in self.category_scores.items()},
            'reasoning': self.reasoning,
            'keywords_found': self.keywords_found,
            'sentiment_score': self.sentiment_score,
            'emotion_detected': self.emotion_detected,
            'urgency_level': self.urgency_level,
            'is_actionable': self.is_actionable,
            'requires_human_review': self.requires_human_review,
            'classifier_version': self.classifier_version,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'confidence_level': self.get_confidence_level(),
            'sentiment_label': self.get_sentiment_label(),
            'should_escalate': self.should_escalate(),
            'priority_suggestion': self.get_priority_suggestion().value,
            'alternative_categories': self.get_alternative_categories()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Classification':
        """Create Classification from dictionary."""
        # Convert category scores back to enum keys
        category_scores = {}
        if 'category_scores' in data and data['category_scores']:
            category_scores = {
                FeedbackCategory(cat): score 
                for cat, score in data['category_scores'].items()
            }
        
        return cls(
            feedback_id=data['feedback_id'],
            predicted_category=FeedbackCategory(data['predicted_category']),
            confidence_score=float(data['confidence_score']),
            category_scores=category_scores,
            reasoning=data.get('reasoning'),
            keywords_found=data.get('keywords_found', []),
            sentiment_score=data.get('sentiment_score'),
            emotion_detected=data.get('emotion_detected'),
            urgency_level=data.get('urgency_level'),
            is_actionable=data.get('is_actionable', True),
            requires_human_review=data.get('requires_human_review', False),
            classifier_version=data.get('classifier_version', '1.0'),
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the classification."""
        summary_parts = [
            f"Category: {self.predicted_category.value.replace('_', ' ').title()}",
            f"Confidence: {self.get_confidence_level()} ({self.confidence_score:.2f})"
        ]
        
        if self.sentiment_score is not None:
            summary_parts.append(f"Sentiment: {self.get_sentiment_label()}")
        
        if self.emotion_detected:
            summary_parts.append(f"Emotion: {self.emotion_detected.title()}")
        
        if self.urgency_level:
            summary_parts.append(f"Urgency: {self.urgency_level.title()}")
        
        if self.should_escalate():
            summary_parts.append("⚠️ Requires Review")
        
        return " | ".join(summary_parts)
    
    @classmethod
    def create_low_confidence(cls, feedback_id: str, category: FeedbackCategory, 
                            confidence: float, reason: str = "Low confidence classification") -> 'Classification':
        """
        Create a low-confidence classification that requires human review.
        
        Args:
            feedback_id: ID of the feedback
            category: Best guess category
            confidence: Confidence score (should be < 0.7)
            reason: Reason for low confidence
            
        Returns:
            Classification instance with human review flag set
        """
        return cls(
            feedback_id=feedback_id,
            predicted_category=category,
            confidence_score=confidence,
            reasoning=reason,
            requires_human_review=True,
            is_actionable=False
        )
    
    @classmethod
    def create_high_confidence(cls, feedback_id: str, category: FeedbackCategory,
                             confidence: float, reasoning: str, 
                             keywords: List[str] = None) -> 'Classification':
        """
        Create a high-confidence classification.
        
        Args:
            feedback_id: ID of the feedback
            category: Predicted category
            confidence: Confidence score (should be >= 0.8)
            reasoning: Explanation for the classification
            keywords: Keywords that influenced the decision
            
        Returns:
            Classification instance
        """
        return cls(
            feedback_id=feedback_id,
            predicted_category=category,
            confidence_score=confidence,
            reasoning=reasoning,
            keywords_found=keywords or [],
            requires_human_review=False,
            is_actionable=True
        )


@dataclass
class ClassificationBatch:
    """Represents a batch of classifications for performance tracking."""
    
    batch_id: str
    classifications: List[Classification] = field(default_factory=list)
    batch_size: int = 0
    processing_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    average_confidence: float = 0.0
    model_version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate batch statistics."""
        self.batch_size = len(self.classifications)
        self.success_count = len([c for c in self.classifications if c.confidence_score >= 0.7])
        self.failure_count = self.batch_size - self.success_count
        
        if self.classifications:
            self.average_confidence = sum(c.confidence_score for c in self.classifications) / len(self.classifications)
        
        if not self.batch_id:
            self.batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:8]}"
    
    def add_classification(self, classification: Classification):
        """Add a classification to the batch and update statistics."""
        self.classifications.append(classification)
        self.__post_init__()  # Recalculate statistics
    
    def get_category_distribution(self) -> Dict[FeedbackCategory, int]:
        """Get distribution of categories in this batch."""
        distribution = {}
        for classification in self.classifications:
            category = classification.predicted_category
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels."""
        distribution = {"Very High": 0, "High": 0, "Medium": 0, "Low": 0, "Very Low": 0}
        for classification in self.classifications:
            level = classification.get_confidence_level()
            distribution[level] += 1
        return distribution
    
    def get_review_required_count(self) -> int:
        """Get count of classifications requiring human review."""
        return len([c for c in self.classifications if c.requires_human_review])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'batch_id': self.batch_id,
            'classifications': [c.to_dict() for c in self.classifications],
            'batch_size': self.batch_size,
            'processing_time': self.processing_time,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'average_confidence': self.average_confidence,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'category_distribution': {cat.value: count for cat, count in self.get_category_distribution().items()},
            'confidence_distribution': self.get_confidence_distribution(),
            'review_required_count': self.get_review_required_count()
        }


@dataclass
class AgentResult:
    """Represents the result of an agent's processing."""
    
    agent_name: str
    success: bool
    data: Optional[Any] = None
    confidence: Optional[float] = None
    error_message: Optional[str] = None
    details: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'agent_name': self.agent_name,
            'success': self.success,
            'confidence': self.confidence,
            'error_message': self.error_message,
            'details': self.details,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PipelineResult:
    """Represents the result of the entire pipeline processing."""
    
    id: str
    input_feedback_count: int
    processed_feedback_count: int
    generated_tickets_count: int
    success_rate: float
    total_processing_time: float
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    generated_tickets: List[Ticket] = field(default_factory=list)
    failed_items: List[str] = field(default_factory=list)
    processing_logs: List[str] = field(default_factory=list)  # THIS LINE MUST BE HERE
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    bug_details: Optional['BugDetails'] = None
    feature_details: Optional['FeatureDetails'] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            self.id = f"PIPELINE-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:8]}"
    

    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence from agent results"""
        if not self.agent_results:
            return 0.0
        
        confidences = []
        for agent_result in self.agent_results.values():
            if hasattr(agent_result, 'confidence') and agent_result.confidence is not None:
                confidences.append(agent_result.confidence)
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
        
    # @property   
    # def classification(self) -> Optional['Classification']:
    #     """Get the classification from agent results"""
    #     for result in self.agent_results:
    #         if result.agent_name == 'feedback_classifier' and result.data:
    #             return result.data
    #     return None
    
    classification: Optional['Classification'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'input_feedback_count': self.input_feedback_count,
            'processed_feedback_count': self.processed_feedback_count,
            'generated_tickets_count': self.generated_tickets_count,
            'success_rate': self.success_rate,
            'total_processing_time': self.total_processing_time,
            'agent_results': [result.to_dict() for result in self.agent_results],
            'generated_tickets': [ticket.to_dict() for ticket in self.generated_tickets],
            'failed_items': self.failed_items,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'bug_details': self.bug_details.to_dict() if self.bug_details else None,
            'feature_details': self.feature_details.to_dict() if self.feature_details else None,
        }

    @property
    def generated_ticket(self) -> Optional['Ticket']:
        """Backward compatibility property"""
        return self.generated_tickets[0] if self.generated_tickets else None

    @generated_ticket.setter   
    def generated_ticket(self, ticket: Optional['Ticket']):
        """Setter for backward compatibility"""
        if ticket is not None:
            # Clear existing tickets and add the new one
            self.generated_tickets = [ticket]
        else:
            self.generated_tickets = []

    @property  
    def success(self) -> bool:
        """Whether the pipeline processing was successful"""
        return self.success_rate > 0.0 and len(self.failed_items) == 0
    
    def is_successful(self) -> bool:
        """Check if this pipeline result was successful"""
        return self.success_rate > 0.0 and len(self.failed_items) == 0

@dataclass
class ProcessingLog:
    """Represents a log entry for processing activities."""
    
    id: str
    level: str  # INFO, WARNING, ERROR, DEBUG
    message: str
    agent_name: Optional[str] = None
    feedback_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'level': self.level,
            'message': self.message,
            'agent_name': self.agent_name,
            'feedback_id': self.feedback_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class Metric:
    """Represents a performance or quality metric."""
    
    name: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    category: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'category': self.category,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class SystemStats:
    """System-wide statistics and performance metrics."""
    
    total_feedback_processed: int = 0
    total_tickets_generated: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    agent_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    source_distribution: Dict[str, int] = field(default_factory=dict)
    priority_distribution: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def update_stats(self, *args, **kwargs):
        """Update system statistics"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.now()


# Helper functions for creating feedback items from different sources
def create_feedback_from_review(review_data: Dict[str, Any]) -> FeedbackItem:
    """Create FeedbackItem from app store review data."""
    return FeedbackItem(
        id=str(review_data.get('review_id', '')),
        content=str(review_data.get('review_text', '')),
        source=FeedbackSource.APP_STORE,
        timestamp=review_data.get('date', datetime.now()),
        metadata={
            'user_id': str(review_data.get('user_id', '')),
            'rating': review_data.get('rating'),
            'app_version': review_data.get('app_version'),
            'device_type': review_data.get('device_type'),
            'country': review_data.get('country')
        }
    )


def create_feedback_from_email(email_data: Dict[str, Any]) -> FeedbackItem:
    """Create FeedbackItem from support email data."""
    subject = str(email_data.get('subject', ''))
    body = str(email_data.get('body', ''))
    content = f"{subject}\n\n{body}" if subject and body else (subject or body)
    
    return FeedbackItem(
        id=str(email_data.get('email_id', '')),
        content=content,
        source=FeedbackSource.SUPPORT_EMAIL,
        timestamp=email_data.get('received_date', datetime.now()),
        metadata={
            'sender_email': str(email_data.get('sender_email', '')),
            'subject': subject,
            'priority': email_data.get('priority'),
            'category': email_data.get('category'),
            'status': email_data.get('status')
        }
    )


def create_feedback_from_survey(survey_data: Dict[str, Any]) -> FeedbackItem:
    """Create FeedbackItem from customer survey data."""
    return FeedbackItem(
        id=str(survey_data.get('response_id', '')),
        content=str(survey_data.get('feedback_text', '')),
        source=FeedbackSource.CUSTOMER_SURVEY,
        timestamp=survey_data.get('submission_date', datetime.now()),
        metadata={
            'survey_id': str(survey_data.get('survey_id', '')),
            'question_id': str(survey_data.get('question_id', '')),
            'rating': survey_data.get('rating'),
            'respondent_id': str(survey_data.get('respondent_id', ''))
        }
    )


def create_feedback_from_social_media(social_data: Dict[str, Any]) -> FeedbackItem:
    """Create FeedbackItem from social media post data."""
    return FeedbackItem(
        id=str(social_data.get('post_id', '')),
        content=str(social_data.get('content', '')),
        source=FeedbackSource.SOCIAL_MEDIA,
        timestamp=social_data.get('posted_date', datetime.now()),
        metadata={
            'platform': str(social_data.get('platform', '')),
            'author': str(social_data.get('author', '')),
            'likes': social_data.get('likes', 0),
            'shares': social_data.get('shares', 0),
            'hashtags': social_data.get('hashtags', []),
            'mentions': social_data.get('mentions', [])
        }
    )


# Validation functions
def validate_feedback_item(item: FeedbackItem) -> bool:
    """Validate a feedback item for required fields and data integrity."""
    if not item.id or not item.content:
        return False
    
    if not isinstance(item.source, FeedbackSource):
        return False
    
    if len(item.content.strip()) < 5:  # Minimum content length
        return False
    
    return True


def validate_ticket(ticket: Ticket) -> bool:
    """Validate a ticket for required fields and data integrity."""
    if not ticket.id or not ticket.title or not ticket.description:
        return False
    
    if not isinstance(ticket.category, FeedbackCategory):
        return False
    
    if not isinstance(ticket.priority, Priority):
        return False
    
    if len(ticket.title.strip()) < 5 or len(ticket.description.strip()) < 10:
        return False
    
    return True

def create_pipeline_result(
    id: str,
    input_feedback_count: int = 0,
    processed_feedback_count: int = 0,
    generated_tickets_count: int = 0,
    success_rate: float = 0.0,
    total_processing_time: float = 0.0,
    **kwargs
) -> 'PipelineResult':
    """Create a PipelineResult with required parameters"""
    return PipelineResult(
        id=id,
        input_feedback_count=input_feedback_count,
        processed_feedback_count=processed_feedback_count,
        generated_tickets_count=generated_tickets_count,
        success_rate=success_rate,
        total_processing_time=total_processing_time,
        agent_results=kwargs.get('agent_results', {}),
        generated_tickets=kwargs.get('generated_tickets', []),
        failed_items=kwargs.get('failed_items', []),
        processing_logs=kwargs.get('processing_logs', []),
        metadata=kwargs.get('metadata', {}),
        timestamp=kwargs.get('timestamp', datetime.now())
    )