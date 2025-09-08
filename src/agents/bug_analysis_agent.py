"""
Bug Analysis Agent for extracting technical details from bug reports.
"""

import re
from typing import Dict, Any, Optional, List
from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.core.data_models import AgentResult, BugDetails, Platform, FeedbackItem, Classification
from src.core.nlp_utils import NLPUtils
from config.settings import OPENAI_API_KEY, AGENT_SETTINGS


class BugAnalysisAgent(BaseAgent):
    """Agent responsible for analyzing bug reports and extracting technical details."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Bug Analysis Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("bug_analyzer", config)
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.nlp_utils = NLPUtils()
        
        # Bug severity indicators
        self.severity_keywords = {
            "Critical": [
                "crash", "data loss", "security", "cannot access", "completely broken",
                "unusable", "critical", "urgent", "cannot login", "payment"
            ],
            "High": [
                "error", "broken", "not working", "failed", "issue", "problem",
                "bug", "incorrect", "missing"
            ],
            "Medium": [
                "slow", "lag", "minor issue", "inconsistent", "sometimes",
                "occasionally", "small problem"
            ],
            "Low": [
                "cosmetic", "minor", "suggestion", "improvement", "enhancement",
                "nice to have", "polish"
            ]
        }
        
        # Platform detection patterns
        self.platform_patterns = {
            Platform.IOS: [
                r'\biphone\b', r'\bipad\b', r'\bios\b', r'\bapple\b',
                r'\bsafari\b', r'\biphone\s+\d+', r'\bipad\s+\w+'
            ],
            Platform.ANDROID: [
                r'\bandroid\b', r'\bgoogle\s+play\b', r'\bsamsung\b',
                r'\bgalaxy\b', r'\bpixel\b', r'\bchrome\b'
            ],
            Platform.WEB: [
                r'\bbrowser\b', r'\bchrome\b', r'\bfirefox\b', r'\bsafari\b',
                r'\bedge\b', r'\bweb\b', r'\bwebsite\b'
            ],
            Platform.DESKTOP: [
                r'\bwindows\b', r'\bmac\b', r'\blinux\b', r'\bpc\b',
                r'\bdesktop\b', r'\blaptop\b'
            ]
        }
        
        # Error message patterns
        self.error_patterns = [
            r'error\s+(?:code\s+)?(\d+)',
            r'(E\d{3,4})',
            r'(\d{3,4}\s+error)',
            r'error[:\s]+"([^"]+)"',
            r'exception[:\s]+"([^"]+)"',
            r'failed[:\s]+"([^"]+)"'
        ]
    
    async def process(self, data: Any) -> AgentResult:
        """
        Analyze bug reports and extract technical details.
        
        Args:
            data: Dictionary containing feedback item and classification
            
        Returns:
            AgentResult containing BugDetails object
        """
        try:
            # Validate input
            if not isinstance(data, dict) or "item" not in data:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="Input must contain 'item' key with FeedbackItem"
                )
            
            feedback_item = data["item"]
            classification = data.get("classification")
            
            if not isinstance(feedback_item, FeedbackItem):
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="Item must be a FeedbackItem instance"
                )
            
            # Only process if classified as bug
            if classification and classification.category.value != "Bug":
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    data=None,
                    confidence=0.0,
                    details="Skipped - not classified as bug"
                )
            
            # Extract bug details
            bug_details = await self._analyze_bug_report(feedback_item)
            
            # Calculate confidence based on extracted information
            confidence = self._calculate_confidence(bug_details, feedback_item)
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=bug_details,
                confidence=confidence
            )
            
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"Bug analysis failed: {str(e)}"
            )
    
    async def _analyze_bug_report(self, item: FeedbackItem) -> BugDetails:
        """
        Analyze a bug report and extract technical details.
        
        Args:
            item: Feedback item to analyze
            
        Returns:
            BugDetails object with extracted information
        """
        content = item.content.lower()
        
        # Extract basic bug details
        severity = self._determine_severity(content, item)
        platform = self._detect_platform(content, item)
        app_version = self._extract_app_version(content, item)
        device_info = self._extract_device_info(content)
        
        # Extract reproduction steps
        steps = self._extract_steps_to_reproduce(content)
        
        # Extract behaviors
        expected_behavior = self._extract_expected_behavior(content)
        actual_behavior = self._extract_actual_behavior(content)
        
        # Extract error messages
        error_messages = self._extract_error_messages(content)
        
        # Detect if screenshots are mentioned
        screenshots_mentioned = self._detect_screenshots(content)
        
        # Determine reproducibility
        reproducibility = self._determine_reproducibility(content)
        
        # Extract affected features
        affected_features = self._extract_affected_features(content)
        
        # Use OpenAI for enhanced analysis if available
        if self.client and self.get_config_value("use_openai", True):
            try:
                enhanced_details = await self._enhance_with_openai(item, {
                    "severity": severity,
                    "platform": platform.value,
                    "steps": steps,
                    "errors": error_messages
                })
                
                # Merge enhanced details
                if enhanced_details:
                    steps.extend(enhanced_details.get("additional_steps", []))
                    error_messages.extend(enhanced_details.get("additional_errors", []))
                    affected_features.extend(enhanced_details.get("features", []))
                    
                    # Update severity if AI suggests higher
                    ai_severity = enhanced_details.get("severity")
                    if ai_severity and self._severity_level(ai_severity) > self._severity_level(severity):
                        severity = ai_severity
                        
            except Exception as e:
                self.logger.warning(f"OpenAI enhancement failed: {e}")
        
        return BugDetails(
            severity=severity,
            platform=platform,
            app_version=app_version,
            device_info=device_info,
            steps_to_reproduce=list(set(steps)),  # Remove duplicates
            expected_behavior=expected_behavior,
            actual_behavior=actual_behavior,
            error_messages=list(set(error_messages)),  # Remove duplicates
            screenshots_mentioned=screenshots_mentioned,
            reproducibility=reproducibility,
            affected_features=list(set(affected_features))  # Remove duplicates
        )
    
    def _determine_severity(self, content: str, item: FeedbackItem) -> str:
        """Determine bug severity based on content and context."""
        severity_scores = {severity: 0 for severity in self.severity_keywords.keys()}
        
        # Score based on keywords
        for severity, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    severity_scores[severity] += 1
        
        # Context-based adjustments
        if item.source_type == "app_store_review" and item.rating and item.rating <= 1:
            severity_scores["Critical"] += 2
            severity_scores["High"] += 1
        elif item.source_type == "support_email":
            if "urgent" in content or "critical" in content:
                severity_scores["Critical"] += 3
            if item.metadata.get("priority") == "High":
                severity_scores["High"] += 2
        
        # Additional severity indicators
        if any(word in content for word in ["crash", "data loss", "cannot access"]):
            severity_scores["Critical"] += 3
        
        if any(word in content for word in ["not working", "broken", "failed"]):
            severity_scores["High"] += 2
        
        # Return highest scoring severity
        max_severity = max(severity_scores.items(), key=lambda x: x[1])
        return max_severity[0] if max_severity[1] > 0 else "Medium"
    
    def _detect_platform(self, content: str, item: FeedbackItem) -> Platform:
        """Detect platform from content and metadata."""
        platform_scores = {platform: 0 for platform in Platform}
        
        # Score based on patterns
        for platform, patterns in self.platform_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                platform_scores[platform] += matches
        
        # Use metadata if available
        if item.platform:
            platform_lower = item.platform.lower()
            if "ios" in platform_lower or "app store" in platform_lower:
                platform_scores[Platform.IOS] += 5
            elif "android" in platform_lower or "google play" in platform_lower:
                platform_scores[Platform.ANDROID] += 5
        
        # Return highest scoring platform
        max_platform = max(platform_scores.items(), key=lambda x: x[1])
        return max_platform[0] if max_platform[1] > 0 else Platform.UNKNOWN
    
    def _extract_app_version(self, content: str, item: FeedbackItem) -> Optional[str]:
        """Extract app version from content or metadata."""
        # Check metadata first
        if item.metadata and "app_version" in item.metadata:
            return item.metadata["app_version"]
        
        # Look for version patterns in content
        version_patterns = [
            r'v?(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)',
            r'version\s+(\d+\.\d+(?:\.\d+)?)',
            r'update\s+(\d+\.\d+(?:\.\d+)?)',
            r'build\s+(\d+)'
        ]
        
        for pattern in version_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def _extract_device_info(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract device information from content."""
        device_info = {}
        
        # Device model patterns
        device_patterns = [
            r'(iphone\s+\d+(?:\s+\w+)?)',
            r'(ipad\s+\w+)',
            r'(galaxy\s+\w+)',
            r'(pixel\s+\d+)',
            r'(samsung\s+\w+)',
            r'(oneplus\s+\d+)',
            r'(xiaomi\s+\w+)'
        ]
        
        for pattern in device_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                device_info["model"] = matches[0]
                break
        
        # OS version patterns
        os_patterns = [
            r'ios\s+(\d+(?:\.\d+)?)',
            r'android\s+(\d+(?:\.\d+)?)',
            r'windows\s+(\d+)',
            r'macos\s+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in os_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                device_info["os_version"] = matches[0]
                break
        
        return device_info if device_info else None
    
    def _extract_steps_to_reproduce(self, content: str) -> List[str]:
        """Extract steps to reproduce the bug."""
        steps = []
        
        # Look for numbered steps
        step_patterns = [
            r'(\d+\.\s+[^.!?]+[.!?])',
            r'(step\s+\d+[:\-]\s+[^.!?]+[.!?])'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            steps.extend([match.strip() for match in matches])
        
        # Look for sequential indicators
        sequence_indicators = [
            r'(first[,\s]+[^.!?]+[.!?])',
            r'(then[,\s]+[^.!?]+[.!?])',
            r'(next[,\s]+[^.!?]+[.!?])',
            r'(after[,\s]+[^.!?]+[.!?])',
            r'(finally[,\s]+[^.!?]+[.!?])'
        ]
        
        for pattern in sequence_indicators:
            matches = re.findall(pattern, content, re.IGNORECASE)
            steps.extend([match.strip() for match in matches])
        
        # Clean up steps
        cleaned_steps = []
        for step in steps:
            step = step.strip()
            if len(step) > 10 and step not in cleaned_steps:  # Avoid duplicates and too short steps
                cleaned_steps.append(step)
        
        return cleaned_steps[:10]  # Limit to 10 steps
    
    def _extract_expected_behavior(self, content: str) -> Optional[str]:
        """Extract expected behavior description."""
        expected_patterns = [
            r'expected[:\s]+([^.!?]+[.!?])',
            r'should[:\s]+([^.!?]+[.!?])',
            r'supposed to[:\s]+([^.!?]+[.!?])'
        ]
        
        for pattern in expected_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _extract_actual_behavior(self, content: str) -> Optional[str]:
        """Extract actual behavior description."""
        actual_patterns = [
            r'actually[:\s]+([^.!?]+[.!?])',
            r'instead[:\s]+([^.!?]+[.!?])',
            r'but[:\s]+([^.!?]+[.!?])',
            r'however[:\s]+([^.!?]+[.!?])'
        ]
        
        for pattern in actual_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _extract_error_messages(self, content: str) -> List[str]:
        """Extract error messages from content."""
        errors = []
        
        for pattern in self.error_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            errors.extend(matches)
        
        # Look for quoted error messages
        quoted_errors = re.findall(r'"([^"]*error[^"]*)"', content, re.IGNORECASE)
        errors.extend(quoted_errors)
        
        # Clean and deduplicate
        cleaned_errors = []
        for error in errors:
            error = error.strip()
            if len(error) > 3 and error not in cleaned_errors:
                cleaned_errors.append(error)
        
        return cleaned_errors
    
    def _detect_screenshots(self, content: str) -> bool:
        """Detect if screenshots are mentioned."""
        screenshot_indicators = [
            "screenshot", "screen shot", "image", "photo", "picture",
            "attached", "attachment", "see attached"
        ]
        
        return any(indicator in content for indicator in screenshot_indicators)
    
    def _determine_reproducibility(self, content: str) -> Optional[str]:
        """Determine bug reproducibility."""
        if any(word in content for word in ["always", "every time", "consistently"]):
            return "Always"
        elif any(word in content for word in ["sometimes", "occasionally", "intermittent"]):
            return "Sometimes"
        elif any(word in content for word in ["rarely", "once", "happened once"]):
            return "Rarely"
        
        return None
    
    def _extract_affected_features(self, content: str) -> List[str]:
        """Extract affected app features."""
        feature_keywords = [
            "login", "sync", "save", "export", "import", "search", "share",
            "notification", "backup", "settings", "profile", "dashboard",
            "upload", "download", "camera", "gallery", "contacts"
        ]
        
        affected = []
        for feature in feature_keywords:
            if feature in content:
                affected.append(feature)
        
        return affected
    
    async def _enhance_with_openai(self, item: FeedbackItem, basic_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use OpenAI to enhance bug analysis."""
        prompt = f"""
        Analyze this bug report and extract additional technical details:
        
        Content: "{item.content}"
        
        Already identified:
        - Severity: {basic_details.get('severity', 'Unknown')}
        - Platform: {basic_details.get('platform', 'Unknown')}
        - Steps found: {len(basic_details.get('steps', []))}
        - Errors found: {len(basic_details.get('errors', []))}
        
        Please provide additional analysis in JSON format:
        {{
            "severity": "Critical|High|Medium|Low",
            "additional_steps": ["step1", "step2"],
            "additional_errors": ["error1", "error2"],
            "features": ["affected_feature1", "affected_feature2"],
            "technical_summary": "brief technical summary"
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.get_config_value("openai_model", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            enhanced_details = json.loads(result_text)
            
            return enhanced_details
            
        except Exception as e:
            self.logger.error(f"OpenAI enhancement error: {e}")
            return None
    
    def _calculate_confidence(self, bug_details: BugDetails, item: FeedbackItem) -> float:
        """Calculate confidence score for the bug analysis."""
        confidence = 0.0
        
        # Base confidence from having basic details
        if bug_details.severity != "Medium":  # Non-default severity
            confidence += 0.2
        
        if bug_details.platform != Platform.UNKNOWN:
            confidence += 0.2
        
        if bug_details.app_version:
            confidence += 0.1
        
        if bug_details.device_info:
            confidence += 0.1
        
        # Steps to reproduce boost confidence significantly
        if bug_details.steps_to_reproduce:
            confidence += min(0.3, len(bug_details.steps_to_reproduce) * 0.1)
        
        # Error messages are valuable
        if bug_details.error_messages:
            confidence += min(0.2, len(bug_details.error_messages) * 0.1)
        
        # Additional details
        if bug_details.expected_behavior or bug_details.actual_behavior:
            confidence += 0.1
        
        if bug_details.reproducibility:
            confidence += 0.1
        
        if bug_details.affected_features:
            confidence += 0.05
        
        # Context adjustments
        if item.source_type == "support_email":
            confidence += 0.1  # Support emails often have more detail
        
        if item.rating and item.rating <= 2:
            confidence += 0.05  # Low ratings often indicate real issues
        
        return min(confidence, 1.0)
    
    def _severity_level(self, severity: str) -> int:
        """Convert severity to numeric level for comparison."""
        levels = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        return levels.get(severity, 2)
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about analysis capabilities."""
        return {
            "supported_platforms": [platform.value for platform in Platform],
            "severity_levels": list(self.severity_keywords.keys()),
            "extracts_steps": True,
            "extracts_errors": True,
            "detects_device_info": True,
            "uses_openai_enhancement": bool(self.client),
            "supported_features": [
                "version_extraction", "device_detection", "error_parsing",
                "reproducibility_assessment", "severity_classification"
            ]
        }