"""
Ticket Creator Agent for generating structured tickets from analyzed feedback.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from .base_agent import BaseAgent
from ..core.data_models import (
    AgentResult, Ticket, FeedbackItem, Classification, 
    BugDetails, FeatureDetails, FeedbackCategory, Priority
)
from ..utils.logger import get_logger


class TicketCreatorAgent(BaseAgent):
    """Agent responsible for creating structured tickets from analyzed feedback."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Ticket Creator Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("ticket_creator", config)
        
        # Ticket templates for different categories
        self.ticket_templates = {
            FeedbackCategory.BUG: {
                "title_format": "Bug: {summary}",
                "description_format": """
**Bug Report**

**Original Feedback:** {original_content}

**User Information:**
- Source: {source}
- User: {user_info}
- Date: {date}

**Technical Details:**
{technical_details}

**Severity:** {severity}
**Platform:** {platform}
**Reproducibility:** {reproducibility}

**Steps to Reproduce:**
{steps}

**Expected vs Actual Behavior:**
- Expected: {expected_behavior}
- Actual: {actual_behavior}

**Error Messages:**
{error_messages}

**Affected Features:** {affected_features}
                """.strip()
            },
            
            FeedbackCategory.FEATURE_REQUEST: {
                "title_format": "Feature Request: {summary}",
                "description_format": """
**Feature Request**

**Original Feedback:** {original_content}

**User Information:**
- Source: {source}
- User: {user_info}
- Date: {date}

**Requested Feature:** {requested_feature}

**Use Case:** {use_case}

**User Impact:** {user_impact}
**Estimated Complexity:** {complexity}
**Business Value:** {business_value}

**User Segment:** {user_segment}
**Dependencies:** {dependencies}

**Similar Apps Mentioned:** {similar_apps}
                """.strip()
            },
            
            FeedbackCategory.COMPLAINT: {
                "title_format": "User Complaint: {summary}",
                "description_format": """
**User Complaint**

**Original Feedback:** {original_content}

**User Information:**
- Source: {source}
- User: {user_info}
- Date: {date}

**Issue Summary:** {summary}

**Sentiment Analysis:** {sentiment}
**Key Concerns:** {key_concerns}

**Suggested Actions:**
{suggested_actions}
                """.strip()
            },
            
            FeedbackCategory.PRAISE: {
                "title_format": "Positive Feedback: {summary}",
                "description_format": """
**Positive User Feedback**

**Original Feedback:** {original_content}

**User Information:**
- Source: {source}
- User: {user_info}
- Date: {date}

**Praised Features:** {praised_features}
**Sentiment:** {sentiment}

**Potential Use Cases:**
- Feature promotion
- Marketing testimonial
- Product validation
                """.strip()
            },
            
            FeedbackCategory.SPAM: {
                "title_format": "Spam Report: {summary}",
                "description_format": """
**Spam Report**

**Content:** {original_content}

**Source:** {source}
**Date:** {date}

**Spam Indicators:** {spam_indicators}

**Action Required:** Review and remove if confirmed spam
                """.strip()
            }
        }
        
        # Priority mapping rules
        self.priority_rules = {
            FeedbackCategory.BUG: {
                "Critical": Priority.CRITICAL,
                "High": Priority.HIGH,
                "Medium": Priority.MEDIUM,
                "Low": Priority.LOW
            },
            FeedbackCategory.FEATURE_REQUEST: {
                "High": Priority.HIGH,
                "Medium": Priority.MEDIUM,
                "Low": Priority.LOW
            },
            FeedbackCategory.COMPLAINT: Priority.MEDIUM,
            FeedbackCategory.PRAISE: Priority.LOW,
            FeedbackCategory.SPAM: Priority.LOW
        }
        
        # Team assignment rules
        self.team_assignment = {
            FeedbackCategory.BUG: "Engineering",
            FeedbackCategory.FEATURE_REQUEST: "Product",
            FeedbackCategory.COMPLAINT: "Customer Success",
            FeedbackCategory.PRAISE: "Marketing",
            FeedbackCategory.SPAM: "Moderation"
        }
    
    async def process(self, data: Any) -> AgentResult:
        """
        Create structured tickets from analyzed feedback data.
        
        Args:
            data: Dictionary containing analyzed feedback data
            
        Returns:
            AgentResult containing generated Ticket object
        """
        try:
            # Validate input
            validation_errors = self._validate_input(data)
            if validation_errors:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message=f"Input validation failed: {'; '.join(validation_errors)}"
                )
            
            feedback_item = data["item"]
            classification = data["classification"]
            bug_details = data.get("bug_details")
            feature_details = data.get("feature_details")
            
            # Create ticket
            ticket = await self._create_ticket(
                feedback_item, 
                classification, 
                bug_details, 
                feature_details
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(ticket, data)
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=ticket,
                confidence=confidence
            )
            
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"Ticket creation failed: {str(e)}"
            )
    
    def _validate_input(self, data: Dict[str, Any]) -> list[str]:
        """Validate input data for ticket creation."""
        errors = []
        
        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return errors
        
        # Check required fields
        required_fields = ["item", "classification"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None:
                errors.append(f"Field cannot be None: {field}")
        
        # Validate types
        if "item" in data and not isinstance(data["item"], FeedbackItem):
            errors.append("'item' must be a FeedbackItem instance")
        
        if "classification" in data and not isinstance(data["classification"], Classification):
            errors.append("'classification' must be a Classification instance")
        
        return errors
    
    async def _create_ticket(
        self,
        item: FeedbackItem,
        classification: Classification,
        bug_details: Optional[BugDetails] = None,
        feature_details: Optional[FeatureDetails] = None
    ) -> Ticket:
        """Create a structured ticket from the analyzed data."""
        
        # Generate unique ticket ID
        ticket_id = self._generate_ticket_id(item, classification)
        
        # Determine priority
        priority = self._determine_priority(classification, bug_details, feature_details)
        
        # Generate title
        title = self._generate_title(item, classification, bug_details, feature_details)
        
        # Generate description
        description = self._generate_description(
            item, classification, bug_details, feature_details
        )
        
        # Build technical details
        technical_details = self._build_technical_details(
            classification, bug_details, feature_details
        )
        
        # Generate tags
        tags = self._generate_tags(item, classification, bug_details, feature_details)
        
        # Estimate effort
        estimated_effort = self._estimate_effort(classification, bug_details, feature_details)
        
        # Assign team
        assigned_team = self.team_assignment.get(classification.predicted_category, "General")
        
        # Create ticket
        ticket = Ticket(
            id=ticket_id,  # ← FIXED: changed from ticket_id to id
            source_id=item.id,
            source=item.source,
            category=classification.predicted_category,
            priority=priority,
            title=title,
            description=description,
            technical_details=technical_details,
            created_at=datetime.utcnow(),
            agent_confidence=classification.confidence,
            tags=tags,
            estimated_effort=estimated_effort,
            assigned_team=assigned_team
        )
        
        return ticket
    
    def _generate_ticket_id(self, item: FeedbackItem, classification: Classification) -> str:
        """Generate a unique ticket ID."""
        # Format: CATEGORY-YYYYMMDD-XXXX
        date_str = datetime.utcnow().strftime("%Y%m%d")
        category_code = {
            FeedbackCategory.BUG: "BUG",
            FeedbackCategory.FEATURE_REQUEST: "FEAT",
            FeedbackCategory.COMPLAINT: "COMP",
            FeedbackCategory.PRAISE: "PRAI",
            FeedbackCategory.SPAM: "SPAM"
        }.get(classification.predicted_category, "MISC")
        
        # Use last 4 chars of source ID for uniqueness
        source_suffix = item.id[-4:] if len(item.id) >= 4 else item.id
        
        return f"{category_code}-{date_str}-{source_suffix}"
    
    def _determine_priority(
        self,
        classification: Classification,
        bug_details: Optional[BugDetails] = None,
        feature_details: Optional[FeatureDetails] = None
    ) -> Priority:
        """Determine ticket priority based on analysis results."""
        
        category = classification.predicted_category
        
        if category == FeedbackCategory.BUG and bug_details:
            # Use bug severity to determine priority
            severity_priority = self.priority_rules[category].get(
                bug_details.severity, Priority.MEDIUM
            )
            return severity_priority
        
        elif category == FeedbackCategory.FEATURE_REQUEST and feature_details:
            # Use user impact to determine priority
            impact_priority = self.priority_rules[category].get(
                feature_details.user_impact, Priority.MEDIUM
            )
            return impact_priority
        
        elif category in self.priority_rules:
            # Use default priority for category
            default_priority = self.priority_rules[category]
            if isinstance(default_priority, Priority):
                return default_priority
            else:
                return Priority.MEDIUM
        
        return Priority.MEDIUM
    
    def _generate_title(
        self,
        item: FeedbackItem,
        classification: Classification,
        bug_details: Optional[BugDetails] = None,
        feature_details: Optional[FeatureDetails] = None
    ) -> str:
        """Generate a concise, actionable ticket title."""
        
        category = classification.predicted_category
        
        # Generate summary based on category
        if category == FeedbackCategory.BUG and bug_details:
            if bug_details.affected_features:
                summary = f"{bug_details.affected_features[0]} not working properly"
            else:
                summary = "Application issue reported"
        
        elif category == FeedbackCategory.FEATURE_REQUEST and feature_details:
            summary = feature_details.requested_feature[:50]
        
        else:
            # Extract key phrases for summary
            summary = self._extract_key_summary(item.content)
        
        # Use template to format title
        template = self.ticket_templates.get(category, {})
        title_format = template.get("title_format", "{summary}")
        
        title = title_format.format(summary=summary)
        
        # Ensure title length is reasonable
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title
    
    def _generate_description(
        self,
        item: FeedbackItem,
        classification: Classification,
        bug_details: Optional[BugDetails] = None,
        feature_details: Optional[FeatureDetails] = None
    ) -> str:
        """Generate detailed ticket description."""
        
        category = classification.predicted_category
        template = self.ticket_templates.get(category, {})
        description_format = template.get("description_format", "{original_content}")
        
        # Prepare template variables
        template_vars = {
            "original_content": item.content,
            "source": item.source,
            "user_info": self._format_user_info(item),
            "date": item.timestamp.strftime("%Y-%m-%d %H:%M:%S") if item.timestamp else "Unknown",
            "sentiment": classification.sentiment or "Not analyzed"
        }
        
        # Add category-specific variables
        if category == FeedbackCategory.BUG and bug_details:
            template_vars.update({
                "technical_details": self._format_technical_details(bug_details),
                "severity": bug_details.severity,
                "platform": bug_details.platform.value,
                "reproducibility": bug_details.reproducibility or "Unknown",
                "steps": self._format_steps(bug_details.steps_to_reproduce),
                "expected_behavior": bug_details.expected_behavior or "Not specified",
                "actual_behavior": bug_details.actual_behavior or "Not specified",
                "error_messages": self._format_error_messages(bug_details.error_messages),
                "affected_features": ", ".join(bug_details.affected_features) or "Not specified"
            })
        
        elif category == FeedbackCategory.FEATURE_REQUEST and feature_details:
            template_vars.update({
                "requested_feature": feature_details.requested_feature,
                "use_case": feature_details.use_case or "Not specified",
                "user_impact": feature_details.user_impact,
                "complexity": feature_details.estimated_complexity,
                "business_value": feature_details.business_value or "Not assessed",
                "user_segment": feature_details.user_segment or "General",
                "dependencies": ", ".join(feature_details.dependencies) or "None identified",
                "similar_apps": ", ".join(feature_details.similar_apps_mentioned) or "None mentioned"
            })
        
        elif category == FeedbackCategory.COMPLAINT:
            template_vars.update({
                "summary": self._extract_key_summary(item.content),
                "key_concerns": self._extract_key_concerns(item.content),
                "suggested_actions": self._suggest_complaint_actions(item, classification)
            })
        
        elif category == FeedbackCategory.PRAISE:
            template_vars.update({
                "praised_features": self._extract_praised_features(item.content),
                "summary": self._extract_key_summary(item.content)
            })
        
        elif category == FeedbackCategory.SPAM:
            template_vars.update({
                "spam_indicators": ", ".join(classification.keywords) if classification.keywords else "Automated detection",
                "summary": "Potential spam content"
            })
        
        # Format description using template
        try:
            description = description_format.format(**template_vars)
        except KeyError as e:
            self.logger.warning(f"Missing template variable: {e}")
            description = f"**Original Feedback:** {item.content}\n\n**Category:** {category.value}"
        
        return description
    
    def _build_technical_details(
        self,
        classification: Classification,
        bug_details: Optional[BugDetails] = None,
        feature_details: Optional[FeatureDetails] = None
    ) -> Dict[str, Any]:
        """Build technical details dictionary for the ticket."""
        
        details = {
            "classification_confidence": classification.confidence,
            "category": classification.predicted_category.value,
            "sentiment": classification.sentiment,
            "keywords": classification.keywords
        }
        
        if bug_details:
            details.update({
                "bug_severity": bug_details.severity,
                "platform": bug_details.platform.value,
                "app_version": bug_details.app_version,
                "device_info": bug_details.device_info,
                "reproducibility": bug_details.reproducibility,
                "error_messages": bug_details.error_messages,
                "affected_features": bug_details.affected_features
            })
        
        if feature_details:
            details.update({
                "user_impact": feature_details.user_impact,
                "complexity": feature_details.estimated_complexity,
                "business_value": feature_details.business_value,
                "user_segment": feature_details.user_segment,
                "dependencies": feature_details.dependencies
            })
        
        return details
    
    def _generate_tags(
        self,
        item: FeedbackItem,
        classification: Classification,
        bug_details: Optional[BugDetails] = None,
        feature_details: Optional[FeatureDetails] = None
    ) -> list[str]:
        """Generate relevant tags for the ticket."""
        
        tags = [classification.predicted_category.value.lower().replace(" ", "-")]
        
        # Add source type
        tags.append(item.source.value.replace("_", "-"))
        
        # Add platform if available
        if bug_details and bug_details.platform != "Unknown":
            tags.append(bug_details.platform.value.lower())
        
        # Add severity/priority indicators
        if bug_details:
            tags.append(f"severity-{bug_details.severity.lower()}")
            if bug_details.affected_features:
                tags.extend([f"feature-{feature.lower()}" for feature in bug_details.affected_features[:3]])
        
        if feature_details:
            tags.append(f"impact-{feature_details.user_impact.lower()}")
            tags.append(f"complexity-{feature_details.estimated_complexity.lower()}")
        
        # Add sentiment tag
        if classification.sentiment:
            tags.append(f"sentiment-{classification.sentiment}")
        
        # Add rating tag if available
        if getattr(item, "rating", None):
            tags.append(f"rating-{getattr(item, "rating", None)}-stars")
        
        return list(set(tags))  # Remove duplicates
    
    def _estimate_effort(
        self,
        classification: Classification,
        bug_details: Optional[BugDetails] = None,
        feature_details: Optional[FeatureDetails] = None
    ) -> Optional[str]:
        """Estimate implementation effort for the ticket."""
        
        category = classification.predicted_category
        
        if category == FeedbackCategory.BUG and bug_details:
            severity_effort = {
                "Critical": "High",
                "High": "Medium",
                "Medium": "Low",
                "Low": "Low"
            }
            return severity_effort.get(bug_details.severity, "Medium")
        
        elif category == FeedbackCategory.FEATURE_REQUEST and feature_details:
            complexity_effort = {
                "Simple": "Low",
                "Moderate": "Medium",
                "Complex": "High"
            }
            return complexity_effort.get(feature_details.estimated_complexity, "Medium")
        
        elif category == FeedbackCategory.COMPLAINT:
            return "Low"  # Usually requires investigation rather than development
        
        return None
    
    def _format_user_info(self, item: FeedbackItem) -> str:
        """Format user information for display."""
        user_parts = []
        
        if item.metadata.get("user_name"):
            user_parts.append(f"Name: {item.metadata['user_name']}")
        
        if item.metadata.get("sender_email"):
            user_parts.append(f"Email: {item.metadata['sender_email']}")
        
        if getattr(item, "rating", None):
            user_parts.append(f"Rating: {getattr(item, "rating", None)}/5 stars")
        
        return ", ".join(user_parts) or "Anonymous user"
    
    def _format_technical_details(self, bug_details: BugDetails) -> str:
        """Format technical details for display."""
        details = []
        
        if bug_details.device_info:
            details.append(f"Device: {bug_details.device_info}")
        
        if bug_details.app_version:
            details.append(f"App Version: {bug_details.app_version}")
        
        return "\n".join(details) or "No technical details available"
    
    def _format_steps(self, steps: list[str]) -> str:
        """Format reproduction steps for display."""
        if not steps:
            return "No steps provided"
        
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            formatted_steps.append(f"{i}. {step}")
        
        return "\n".join(formatted_steps)
    
    def _format_error_messages(self, errors: list[str]) -> str:
        """Format error messages for display."""
        if not errors:
            return "No error messages reported"
        
        return "\n".join([f"- {error}" for error in errors])
    
    def _extract_key_summary(self, content: str) -> str:
        """Extract a key summary from content."""
        # Simple extraction - take first sentence or first 50 characters
        sentences = content.split('.')
        if sentences and len(sentences[0]) > 10:
            summary = sentences[0].strip()
            return summary[:50] + ("..." if len(summary) > 50 else "")
        
        return content[:50] + ("..." if len(content) > 50 else "")
    
    def _extract_key_concerns(self, content: str) -> str:
        """Extract key concerns from complaint content."""
        # Look for common complaint patterns
        concerns = []
        content_lower = content.lower()
        
        complaint_patterns = {
            "performance": ["slow", "lag", "performance", "speed"],
            "usability": ["difficult", "confusing", "hard to use", "complicated"],
            "reliability": ["crash", "error", "bug", "broken", "not working"],
            "pricing": ["expensive", "cost", "price", "money", "subscription"],
            "features": ["missing", "lack", "need", "want", "should have"]
        }
        
        for concern_type, keywords in complaint_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                concerns.append(concern_type)
        
        return ", ".join(concerns) or "General dissatisfaction"
    
    def _suggest_complaint_actions(self, item: FeedbackItem, classification: Classification) -> str:
        """Suggest actions for handling complaints."""
        actions = ["Respond to user with acknowledgment"]
        
        if getattr(item, "rating", None) and getattr(item, "rating", None) <= 2:
            actions.append("Priority follow-up required")
        
        if "refund" in item.content.lower() or "money back" in item.content.lower():
            actions.append("Escalate to billing team")
        
        if "bug" in item.content.lower() or "error" in item.content.lower():
            actions.append("Forward to engineering team")
        
        return "\n".join([f"- {action}" for action in actions])
    
    def _extract_praised_features(self, content: str) -> str:
        """Extract praised features from positive feedback."""
        # Simple keyword extraction for praised features
        feature_keywords = [
            "interface", "design", "functionality", "feature", "performance",
            "speed", "ease", "simple", "intuitive", "helpful"
        ]
        
        content_lower = content.lower()
        praised = [keyword for keyword in feature_keywords if keyword in content_lower]
        
        return ", ".join(praised) or "General positive feedback"
    
    def _calculate_confidence(self, ticket: Ticket, data: Dict[str, Any]) -> float:
        """Calculate confidence in the generated ticket."""
        confidence = 0.5  # Base confidence
        
        # Classification confidence contributes significantly
        classification = data["classification"]
        confidence += classification.confidence * 0.3
        
        # Detailed analysis boosts confidence
        if data.get("bug_details"):
            confidence += 0.1
        
        if data.get("feature_details"):
            confidence += 0.1
        
        # Complete tickets have higher confidence
        if len(ticket.description) > 100:
            confidence += 0.05
        
        if ticket.technical_details:
            confidence += 0.05
        
        if ticket.tags:
            confidence += 0.05
        
        return min(confidence, 1.0)