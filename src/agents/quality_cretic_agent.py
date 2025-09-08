"""
Quality Critic Agent for evaluating and improving the quality of generated tickets.
Performs comprehensive quality assessment and provides improvement recommendations.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from .base_agent import BaseAgent
from ..core.data_models import AgentResult, Ticket, FeedbackCategory, Priority
from ..core.nlp_utils import NLPUtils
from ..utils.logger import get_logger


class QualityDimension(Enum):
    """Quality dimensions for ticket evaluation."""
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    ACTIONABILITY = "actionability"
    PRIORITY_ACCURACY = "priority_accuracy"
    CATEGORIZATION = "categorization"
    TECHNICAL_ACCURACY = "technical_accuracy"
    BUSINESS_VALUE = "business_value"


class QualityScore:
    """Represents a quality score for a specific dimension."""
    
    def __init__(self, dimension: QualityDimension, score: float, 
                 max_score: float = 10.0, feedback: str = "", suggestions: List[str] = None):
        self.dimension = dimension
        self.score = score
        self.max_score = max_score
        self.feedback = feedback
        self.suggestions = suggestions or []
        self.percentage = (score / max_score) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "feedback": self.feedback,
            "suggestions": self.suggestions
        }


class QualityReport:
    """Comprehensive quality report for a ticket."""
    
    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.scores: Dict[QualityDimension, QualityScore] = {}
        self.overall_score = 0.0
        self.overall_grade = ""
        self.critical_issues: List[str] = []
        self.improvement_recommendations: List[str] = []
        self.approval_status = False
        self.timestamp = datetime.now()
    
    def add_score(self, quality_score: QualityScore):
        """Add a quality score for a specific dimension."""
        self.scores[quality_score.dimension] = quality_score
    
    def calculate_overall_score(self):
        """Calculate overall quality score."""
        if not self.scores:
            self.overall_score = 0.0
            return
        
        # Weighted scoring - some dimensions are more important
        weights = {
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.CLARITY: 0.18,
            QualityDimension.ACTIONABILITY: 0.20,
            QualityDimension.PRIORITY_ACCURACY: 0.12,
            QualityDimension.CATEGORIZATION: 0.10,
            QualityDimension.TECHNICAL_ACCURACY: 0.15,
            QualityDimension.BUSINESS_VALUE: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in self.scores.items():
            weight = weights.get(dimension, 0.1)
            weighted_sum += score.percentage * weight
            total_weight += weight
        
        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        self._determine_grade()
    
    def _determine_grade(self):
        """Determine letter grade based on overall score."""
        if self.overall_score >= 90:
            self.overall_grade = "A"
        elif self.overall_score >= 80:
            self.overall_grade = "B"
        elif self.overall_score >= 70:
            self.overall_grade = "C"
        elif self.overall_score >= 60:
            self.overall_grade = "D"
        else:
            self.overall_grade = "F"
    
    def should_approve(self, min_score: float = 75.0) -> bool:
        """Determine if ticket should be approved based on quality score."""
        self.approval_status = (
            self.overall_score >= min_score and 
            len(self.critical_issues) == 0 and
            all(score.score >= 6.0 for score in self.scores.values())
        )
        return self.approval_status
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "approval_status": self.approval_status,
            "scores": {dim.value: score.to_dict() for dim, score in self.scores.items()},
            "critical_issues": self.critical_issues,
            "improvement_recommendations": self.improvement_recommendations,
            "timestamp": self.timestamp.isoformat()
        }


class QualityCriticAgent(BaseAgent):
    """Agent responsible for evaluating and improving ticket quality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Quality Critic Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("quality_critic", config)
        self.nlp_utils = NLPUtils()
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.min_approval_score = config.get("min_approval_score", 75.0) if config else 75.0
        self.detailed_analysis = config.get("detailed_analysis", True) if config else True
        self.auto_fix_enabled = config.get("auto_fix_enabled", False) if config else False
        self.strict_mode = config.get("strict_mode", False) if config else False
        
        # Quality criteria
        self.quality_criteria = self._load_quality_criteria()
        
    def _load_quality_criteria(self) -> Dict[QualityDimension, Dict[str, Any]]:
        """Load quality criteria for each dimension."""
        return {
            QualityDimension.COMPLETENESS: {
                "min_title_length": 10,
                "min_description_length": 50,
                "required_fields": ["title", "description", "category", "priority"],
                "context_required": True
            },
            QualityDimension.CLARITY: {
                "max_title_length": 100,
                "readability_threshold": 0.6,
                "jargon_tolerance": 0.2,
                "grammar_threshold": 0.8
            },
            QualityDimension.ACTIONABILITY: {
                "requires_action_items": True,
                "requires_acceptance_criteria": True,
                "specificity_threshold": 0.7
            },
            QualityDimension.PRIORITY_ACCURACY: {
                "severity_keywords": {
                    Priority.CRITICAL: ["crash", "down", "broken", "urgent", "critical", "security"],
                    Priority.HIGH: ["important", "major", "significant", "affecting"],
                    Priority.MEDIUM: ["moderate", "minor", "improvement"],
                    Priority.LOW: ["nice to have", "future", "optional"]
                }
            },
            QualityDimension.CATEGORIZATION: {
                "category_confidence_threshold": 0.8,
                "requires_justification": True
            },
            QualityDimension.TECHNICAL_ACCURACY: {
                "technical_depth_required": True,
                "error_analysis_required": True,
                "solution_feasibility": True
            },
            QualityDimension.BUSINESS_VALUE: {
                "impact_assessment_required": True,
                "user_benefit_required": True,
                "effort_estimation_required": True
            }
        }
    
    async def process(self, data: Any) -> AgentResult:
        """
        Evaluate ticket quality and provide improvement recommendations.
        
        Args:
            data: List of Ticket objects or single Ticket object
            
        Returns:
            AgentResult containing quality reports and improved tickets
        """
        try:
            self.logger.info("Starting quality evaluation process")
            start_time = datetime.now()
            
            # Handle different input types
            if isinstance(data, Ticket):
                tickets = [data]
            elif isinstance(data, list):
                tickets = [item for item in data if isinstance(item, Ticket)]
            else:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="Input must be Ticket object or list of Tickets"
                )
            
            if not tickets:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="No valid tickets found for quality evaluation"
                )
            
            # Evaluate each ticket
            quality_reports = []
            improved_tickets = []
            evaluation_stats = {
                "total_tickets": len(tickets),
                "approved_tickets": 0,
                "rejected_tickets": 0,
                "average_score": 0.0,
                "critical_issues_found": 0
            }
            
            for ticket in tickets:
                try:
                    # Evaluate ticket quality
                    quality_report = await self._evaluate_ticket_quality(ticket)
                    quality_reports.append(quality_report)
                    
                    # Update statistics
                    if quality_report.approval_status:
                        evaluation_stats["approved_tickets"] += 1
                    else:
                        evaluation_stats["rejected_tickets"] += 1
                    
                    evaluation_stats["critical_issues_found"] += len(quality_report.critical_issues)
                    
                    # Auto-fix if enabled and ticket needs improvement
                    if self.auto_fix_enabled and not quality_report.approval_status:
                        improved_ticket = await self._auto_fix_ticket(ticket, quality_report)
                        improved_tickets.append(improved_ticket)
                    else:
                        improved_tickets.append(ticket)
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating ticket {ticket.id}: {str(e)}")
                    continue
            
            # Calculate average score
            if quality_reports:
                evaluation_stats["average_score"] = sum(r.overall_score for r in quality_reports) / len(quality_reports)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result_data = {
                "quality_reports": [report.to_dict() for report in quality_reports],
                "improved_tickets": improved_tickets,
                "evaluation_stats": evaluation_stats,
                "processing_time_seconds": processing_time
            }
            
            self.logger.info(f"Quality evaluation completed. {evaluation_stats['approved_tickets']}"
                           f"/{evaluation_stats['total_tickets']} tickets approved. "
                           f"Average score: {evaluation_stats['average_score']:.1f}")
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result_data,
                confidence=0.9,  # High confidence in quality evaluation
                details=f"Evaluated {len(tickets)} tickets with average score {evaluation_stats['average_score']:.1f}",
                metadata={
                    "evaluation_stats": evaluation_stats,
                    "approval_rate": evaluation_stats["approved_tickets"] / evaluation_stats["total_tickets"] * 100
                }
            )
            
        except Exception as e:
            self.logger.error(f"Quality evaluation failed: {str(e)}", exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"Quality evaluation failed: {str(e)}"
            )
    
    async def _evaluate_ticket_quality(self, ticket: Ticket) -> QualityReport:
        """
        Perform comprehensive quality evaluation of a ticket.
        
        Args:
            ticket: Ticket to evaluate
            
        Returns:
            QualityReport with detailed analysis
        """
        report = QualityReport(ticket.id)
        
        # Evaluate each quality dimension
        for dimension in QualityDimension:
            try:
                score = await self._evaluate_dimension(ticket, dimension)
                report.add_score(score)
            except Exception as e:
                self.logger.warning(f"Error evaluating dimension {dimension.value}: {str(e)}")
                # Add default failing score
                report.add_score(QualityScore(
                    dimension=dimension,
                    score=0.0,
                    feedback=f"Evaluation failed: {str(e)}",
                    suggestions=["Manual review required"]
                ))
        
        # Calculate overall score and determine approval
        report.calculate_overall_score()
        report.should_approve(self.min_approval_score)
        
        # Identify critical issues
        self._identify_critical_issues(ticket, report)
        
        # Generate improvement recommendations
        self._generate_improvement_recommendations(ticket, report)
        
        return report
    
    async def _evaluate_dimension(self, ticket: Ticket, dimension: QualityDimension) -> QualityScore:
        """Evaluate a specific quality dimension."""
        
        if dimension == QualityDimension.COMPLETENESS:
            return await self._evaluate_completeness(ticket)
        elif dimension == QualityDimension.CLARITY:
            return await self._evaluate_clarity(ticket)
        elif dimension == QualityDimension.ACTIONABILITY:
            return await self._evaluate_actionability(ticket)
        elif dimension == QualityDimension.PRIORITY_ACCURACY:
            return await self._evaluate_priority_accuracy(ticket)
        elif dimension == QualityDimension.CATEGORIZATION:
            return await self._evaluate_categorization(ticket)
        elif dimension == QualityDimension.TECHNICAL_ACCURACY:
            return await self._evaluate_technical_accuracy(ticket)
        elif dimension == QualityDimension.BUSINESS_VALUE:
            return await self._evaluate_business_value(ticket)
        else:
            return QualityScore(dimension, 5.0, feedback="Unknown dimension")
    
    async def _evaluate_completeness(self, ticket: Ticket) -> QualityScore:
        """Evaluate completeness of ticket information."""
        criteria = self.quality_criteria[QualityDimension.COMPLETENESS]
        score = 0.0
        max_score = 10.0
        feedback_parts = []
        suggestions = []
        
        # Check required fields
        required_fields = criteria["required_fields"]
        missing_fields = []
        
        for field in required_fields:
            if hasattr(ticket, field):
                field_value = getattr(ticket, field)
                if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                    missing_fields.append(field)
                else:
                    score += 2.0  # 2 points per required field
            else:
                missing_fields.append(field)
        
        if missing_fields:
            feedback_parts.append(f"Missing required fields: {', '.join(missing_fields)}")
            suggestions.extend([f"Add {field}" for field in missing_fields])
        
        # Check minimum lengths
        if ticket.title and len(ticket.title) >= criteria["min_title_length"]:
            score += 1.0
        else:
            feedback_parts.append("Title too short")
            suggestions.append("Provide more descriptive title")
        
        if ticket.description and len(ticket.description) >= criteria["min_description_length"]:
            score += 1.0
        else:
            feedback_parts.append("Description too short")
            suggestions.append("Add more detailed description")
        
        feedback = "; ".join(feedback_parts) if feedback_parts else "All completeness criteria met"
        
        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=min(score, max_score),
            feedback=feedback,
            suggestions=suggestions
        )
    
    async def _evaluate_clarity(self, ticket: Ticket) -> QualityScore:
        """Evaluate clarity of ticket content."""
        criteria = self.quality_criteria[QualityDimension.CLARITY]
        score = 5.0  # Start with baseline
        max_score = 10.0
        feedback_parts = []
        suggestions = []
        
        # Check title length
        if ticket.title:
            if len(ticket.title) <= criteria["max_title_length"]:
                score += 1.0
            else:
                feedback_parts.append("Title too long")
                suggestions.append("Shorten title for clarity")
            
            # Check for clear, actionable language
            if any(word in ticket.title.lower() for word in ["fix", "add", "update", "remove", "improve"]):
                score += 1.0
            else:
                feedback_parts.append("Title lacks clear action")
                suggestions.append("Use action verbs in title")
        
        # Evaluate readability of description
        if ticket.description:
            readability_score = await self._calculate_readability(ticket.description)
            if readability_score >= criteria["readability_threshold"]:
                score += 2.0
            else:
                feedback_parts.append("Description difficult to read")
                suggestions.append("Simplify language and sentence structure")
            
            # Check for excessive jargon
            jargon_ratio = await self._calculate_jargon_ratio(ticket.description)
            if jargon_ratio <= criteria["jargon_tolerance"]:
                score += 1.0
            else:
                feedback_parts.append("Too much technical jargon")
                suggestions.append("Explain technical terms or use simpler language")
        
        feedback = "; ".join(feedback_parts) if feedback_parts else "Content is clear and well-written"
        
        return QualityScore(
            dimension=QualityDimension.CLARITY,
            score=min(score, max_score),
            feedback=feedback,
            suggestions=suggestions
        )
    
    async def _evaluate_actionability(self, ticket: Ticket) -> QualityScore:
        """Evaluate how actionable the ticket is."""
        score = 0.0
        max_score = 10.0
        feedback_parts = []
        suggestions = []
        
        # Check for specific action items
        if ticket.description:
            action_indicators = ["should", "must", "need to", "implement", "create", "fix", "update"]
            if any(indicator in ticket.description.lower() for indicator in action_indicators):
                score += 3.0
            else:
                feedback_parts.append("No clear action items identified")
                suggestions.append("Add specific action items or tasks")
        
        # Check for acceptance criteria
        if ticket.description and ("acceptance criteria" in ticket.description.lower() or 
                                 "definition of done" in ticket.description.lower()):
            score += 3.0
        else:
            feedback_parts.append("Missing acceptance criteria")
            suggestions.append("Add clear acceptance criteria")
        
        # Check for specificity
        if ticket.description:
            specificity_score = await self._calculate_specificity(ticket.description)
            score += specificity_score * 4.0  # Up to 4 points for specificity
        
        feedback = "; ".join(feedback_parts) if feedback_parts else "Ticket is highly actionable"
        
        return QualityScore(
            dimension=QualityDimension.ACTIONABILITY,
            score=min(score, max_score),
            feedback=feedback,
            suggestions=suggestions
        )
    
    async def _evaluate_priority_accuracy(self, ticket: Ticket) -> QualityScore:
        """Evaluate accuracy of priority assignment."""
        criteria = self.quality_criteria[QualityDimension.PRIORITY_ACCURACY]
        score = 5.0  # Start with baseline
        max_score = 10.0
        feedback_parts = []
        suggestions = []
        
        if not ticket.priority:
            return QualityScore(
                dimension=QualityDimension.PRIORITY_ACCURACY,
                score=0.0,
                feedback="No priority assigned",
                suggestions=["Assign appropriate priority level"]
            )
        
        # Check if priority matches content severity
        content = f"{ticket.title} {ticket.description}".lower()
        severity_keywords = criteria["severity_keywords"]
        
        detected_priority = None
        for priority, keywords in severity_keywords.items():
            if any(keyword in content for keyword in keywords):
                detected_priority = priority
                break
        
        if detected_priority:
            if ticket.priority == detected_priority:
                score += 3.0
                feedback_parts.append("Priority matches content severity")
            else:
                score -= 2.0
                feedback_parts.append(f"Priority mismatch: assigned {ticket.priority.value}, "
                                    f"content suggests {detected_priority.value}")
                suggestions.append(f"Consider changing priority to {detected_priority.value}")
        else:
            score += 2.0  # No strong indicators, assume reasonable assignment
        
        feedback = "; ".join(feedback_parts) if feedback_parts else "Priority assignment appears appropriate"
        
        return QualityScore(
            dimension=QualityDimension.PRIORITY_ACCURACY,
            score=min(score, max_score),
            feedback=feedback,
            suggestions=suggestions
        )
    
    async def _evaluate_categorization(self, ticket: Ticket) -> QualityScore:
        """Evaluate accuracy of category assignment."""
        score = 5.0  # Start with baseline
        max_score = 10.0
        feedback_parts = []
        suggestions = []
        
        if not ticket.category:
            return QualityScore(
                dimension=QualityDimension.CATEGORIZATION,
                score=0.0,
                feedback="No category assigned",
                suggestions=["Assign appropriate category"]
            )
        
        # Use NLP to verify category appropriateness
        content = f"{ticket.title} {ticket.description}"
        try:
            predicted_category = await self.nlp_utils.predict_category(content)
            
            if predicted_category == ticket.category:
                score += 5.0
                feedback_parts.append("Category assignment matches content analysis")
            else:
                score += 2.0  # Partial credit
                feedback_parts.append(f"Category may be incorrect: assigned {ticket.category.value}, "
                                    f"analysis suggests {predicted_category.value}")
                suggestions.append(f"Consider changing category to {predicted_category.value}")
        except Exception as e:
            score += 3.0  # Default if analysis fails
            feedback_parts.append("Could not verify category accuracy")
        
        feedback = "; ".join(feedback_parts) if feedback_parts else "Category assignment verified"
        
        return QualityScore(
            dimension=QualityDimension.CATEGORIZATION,
            score=min(score, max_score),
            feedback=feedback,
            suggestions=suggestions
        )
    
    async def _evaluate_technical_accuracy(self, ticket: Ticket) -> QualityScore:
        """Evaluate technical accuracy and depth."""
        score = 3.0  # Start with baseline
        max_score = 10.0
        feedback_parts = []
        suggestions = []
        
        if not ticket.description:
            return QualityScore(
                dimension=QualityDimension.TECHNICAL_ACCURACY,
                score=0.0,
                feedback="No description to evaluate",
                suggestions=["Add detailed technical description"]
            )
        
        content = ticket.description.lower()
        
        # Check for technical details
        technical_indicators = [
            "error", "exception", "stack trace", "log", "version", 
            "browser", "device", "api", "database", "server"
        ]
        
        technical_details_found = sum(1 for indicator in technical_indicators if indicator in content)
        score += min(technical_details_found * 0.5, 3.0)
        
        if technical_details_found == 0:
            feedback_parts.append("Lacks technical details")
            suggestions.append("Add technical context (error messages, environment details, etc.)")
        
        # Check for reproduction steps (for bugs)
        if ticket.category == FeedbackCategory.BUG:
            if any(phrase in content for phrase in ["steps", "reproduce", "how to"]):
                score += 2.0
            else:
                feedback_parts.append("Missing reproduction steps")
                suggestions.append("Add clear reproduction steps")
        
        # Check for proposed solution or investigation
        if any(phrase in content for phrase in ["solution", "fix", "investigate", "research"]):
            score += 2.0
        else:
            suggestions.append("Consider adding potential solutions or investigation approach")
        
        feedback = "; ".join(feedback_parts) if feedback_parts else "Good technical depth and accuracy"
        
        return QualityScore(
            dimension=QualityDimension.TECHNICAL_ACCURACY,
            score=min(score, max_score),
            feedback=feedback,
            suggestions=suggestions
        )
    
    async def _evaluate_business_value(self, ticket: Ticket) -> QualityScore:
        """Evaluate business value and impact assessment."""
        score = 2.0  # Start with baseline
        max_score = 10.0
        feedback_parts = []
        suggestions = []
        
        if not ticket.description:
            return QualityScore(
                dimension=QualityDimension.BUSINESS_VALUE,
                score=0.0,
                feedback="No description to evaluate business value",
                suggestions=["Add business impact assessment"]
            )
        
        content = ticket.description.lower()
        
        # Check for impact indicators
        impact_indicators = [
            "users", "customers", "revenue", "cost", "efficiency", 
            "performance", "satisfaction", "retention", "conversion"
        ]
        
        impact_mentions = sum(1 for indicator in impact_indicators if indicator in content)
        score += min(impact_mentions * 1.0, 4.0)
        
        if impact_mentions == 0:
            feedback_parts.append("No clear business impact mentioned")
            suggestions.append("Describe impact on users or business")
        
        # Check for user benefit
        if any(phrase in content for phrase in ["user", "customer", "benefit", "improve", "better"]):
            score += 2.0
        else:
            suggestions.append("Explain benefits to users")
        
        # Check for effort consideration
        if any(phrase in content for phrase in ["effort", "time", "cost", "resource", "complexity"]):
            score += 2.0
        else:
            suggestions.append("Consider effort estimation")
        
        feedback = "; ".join(feedback_parts) if feedback_parts else "Clear business value articulated"
        
        return QualityScore(
            dimension=QualityDimension.BUSINESS_VALUE,
            score=min(score, max_score),
            feedback=feedback,
            suggestions=suggestions
        )
    
    def _identify_critical_issues(self, ticket: Ticket, report: QualityReport):
        """Identify critical issues that require immediate attention."""
        critical_issues = []
        
        # Check for critical scoring thresholds
        for dimension, score in report.scores.items():
            if score.score < 3.0:  # Critical threshold
                critical_issues.append(f"Critical failure in {dimension.value}: {score.feedback}")
        
        # Check for specific critical conditions
        if not ticket.title or len(ticket.title.strip()) == 0:
            critical_issues.append("Missing title")
        
        if not ticket.description or len(ticket.description.strip()) < 20:
            critical_issues.append("Description too short or missing")
        
        if ticket.category == FeedbackCategory.BUG and ticket.priority == Priority.CRITICAL:
            if "reproduction" not in ticket.description.lower():
                critical_issues.append("Critical bug missing reproduction steps")
        
        report.critical_issues = critical_issues
    
    def _generate_improvement_recommendations(self, ticket: Ticket, report: QualityReport):
        """Generate prioritized improvement recommendations."""
        recommendations = []
        
        # Collect all suggestions from dimension scores
        all_suggestions = []
        for score in report.scores.values():
            all_suggestions.extend(score.suggestions)
        
        # Remove duplicates and prioritize
        unique_suggestions = list(dict.fromkeys(all_suggestions))
        
        # Prioritize based on score impact
        low_scoring_dimensions = [
            dim for dim, score in report.scores.items() 
            if score.score < 6.0
        ]
        
        priority_suggestions = []
        other_suggestions = []
        
        for suggestion in unique_suggestions:
            if any(dim.value in suggestion.lower() for dim in low_scoring_dimensions):
                priority_suggestions.append(suggestion)
            else:
                other_suggestions.append(suggestion)
        
        recommendations = priority_suggestions + other_suggestions
        report.improvement_recommendations = recommendations[:10]  # Limit to top 10
    
    async def _auto_fix_ticket(self, ticket: Ticket, quality_report: QualityReport) -> Ticket:
        """Attempt to automatically fix ticket issues based on quality report."""
        if not self.auto_fix_enabled:
            return ticket
        
        improved_ticket = Ticket(
            id=ticket.id,
            title=ticket.title,
            description=ticket.description,
            category=ticket.category,
            priority=ticket.priority,
            assigned_team=ticket.assigned_team,
            metadata=ticket.metadata.copy() if ticket.metadata else {}
        )
        
        # Apply automatic fixes based on issues found
        for suggestion in quality_report.improvement_recommendations:
            try:
                if "title" in suggestion.lower() and "descriptive" in suggestion.lower():
                    if improved_ticket.category:
                        improved_ticket.title = f"[{improved_ticket.category.value.title()}] {improved_ticket.title}"
                
                elif "acceptance criteria" in suggestion.lower():
                    if "acceptance criteria" not in improved_ticket.description.lower():
                        improved_ticket.description += "\n\nAcceptance Criteria:\n- [ ] To be defined"
                
                elif "technical context" in suggestion.lower():
                    if improved_ticket.category == FeedbackCategory.BUG:
                        improved_ticket.description += "\n\nTechnical Details:\n- Environment: [To be specified]\n- Browser/Device: [To be specified]\n- Steps to Reproduce: [To be added]"
            
            except Exception as e:
                self.logger.warning(f"Error applying auto-fix: {str(e)}")
        
        # Add quality assessment metadata
        improved_ticket.metadata["quality_assessed"] = True
        improved_ticket.metadata["quality_score"] = quality_report.overall_score
        improved_ticket.metadata["auto_fixes_applied"] = len(quality_report.improvement_recommendations)
        
        return improved_ticket
    
    # Helper methods for NLP analysis
    async def _calculate_readability(self, text: str) -> float:
        """Calculate readability score of text."""
        try:
            return await self.nlp_utils.calculate_readability(text)
        except:
            # Fallback simple calculation
            sentences = text.count('.') + text.count('!') + text.count('?')
            words = len(text.split())
            if sentences == 0:
                return 0.5
            avg_sentence_length = words / sentences
            return max(0.0, min(1.0, 1.0 - (avg_sentence_length - 15) / 20))
    
    async def _calculate_jargon_ratio(self, text: str) -> float:
        """Calculate ratio of technical jargon in text."""
        try:
            return await self.nlp_utils.calculate_jargon_ratio(text)
        except:
            # Fallback simple calculation
            technical_words = ["api", "database", "server", "client", "framework", "library", "algorithm"]
            words = text.lower().split()
            if not words:
                return 0.0
            jargon_count = sum(1 for word in words if word in technical_words)
            return jargon_count / len(words)
    
    async def _calculate_specificity(self, text: str) -> float:
        """Calculate how specific the text is."""
        try:
            return await self.nlp_utils.calculate_specificity(text)
        except:
            # Fallback simple calculation
            specific_indicators = ["exactly", "specifically", "precisely", "step", "method", "process"]
            vague_indicators = ["something", "somehow", "maybe", "possibly", "might", "could"]
            
            words = text.lower().split()
            if not words:
                return 0.0
            
            specific_count = sum(1 for word in words if word in specific_indicators)
            vague_count = sum(1 for word in words if word in vague_indicators)
            
            return max(0.0, min(1.0, (specific_count - vague_count) / len(words) + 0.5))
    
    async def batch_evaluate(self, tickets: List[Ticket], batch_size: int = 10) -> List[QualityReport]:
        """
        Evaluate multiple tickets in batches for better performance.
        
        Args:
            tickets: List of tickets to evaluate
            batch_size: Number of tickets to process in each batch
            
        Returns:
            List of quality reports
        """
        all_reports = []
        
        for i in range(0, len(tickets), batch_size):
            batch = tickets[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickets) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            batch_tasks = [self._evaluate_ticket_quality(ticket) for ticket in batch]
            batch_reports = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, report in enumerate(batch_reports):
                if isinstance(report, Exception):
                    self.logger.error(f"Error evaluating ticket {batch[j].id}: {str(report)}")
                    # Create error report
                    error_report = QualityReport(batch[j].id)
                    error_report.critical_issues.append(f"Evaluation failed: {str(report)}")
                    all_reports.append(error_report)
                else:
                    all_reports.append(report)
        
        return all_reports
    
    def get_quality_summary(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """
        Generate summary statistics from quality reports.
        
        Args:
            reports: List of quality reports
            
        Returns:
            Dictionary with summary statistics
        """
        if not reports:
            return {"error": "No reports provided"}
        
        # Calculate overall statistics
        total_tickets = len(reports)
        approved_tickets = sum(1 for r in reports if r.approval_status)
        average_score = sum(r.overall_score for r in reports) / total_tickets
        
        # Grade distribution
        grade_distribution = {}
        for report in reports:
            grade = report.overall_grade
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
        
        # Dimension statistics
        dimension_stats = {}
        for dimension in QualityDimension:
            scores = [r.scores[dimension].score for r in reports if dimension in r.scores]
            if scores:
                dimension_stats[dimension.value] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "below_threshold": sum(1 for s in scores if s < 6.0)
                }
        
        # Common issues
        all_issues = []
        for report in reports:
            all_issues.extend(report.critical_issues)
        
        issue_frequency = {}
        for issue in all_issues:
            issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
        
        common_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_tickets": total_tickets,
            "approved_tickets": approved_tickets,
            "approval_rate": (approved_tickets / total_tickets) * 100,
            "average_score": average_score,
            "grade_distribution": grade_distribution,
            "dimension_statistics": dimension_stats,
            "common_issues": common_issues,
            "recommendations": self._generate_system_recommendations(reports)
        }
    
    def _generate_system_recommendations(self, reports: List[QualityReport]) -> List[str]:
        """Generate system-wide recommendations based on quality patterns."""
        recommendations = []
        
        # Analyze common failure patterns
        failing_dimensions = {}
        for report in reports:
            for dimension, score in report.scores.items():
                if score.score < 6.0:
                    failing_dimensions[dimension] = failing_dimensions.get(dimension, 0) + 1
        
        # Generate recommendations based on patterns
        total_reports = len(reports)
        for dimension, count in failing_dimensions.items():
            if count / total_reports > 0.3:  # More than 30% failing
                if dimension == QualityDimension.COMPLETENESS:
                    recommendations.append("Implement mandatory field validation before ticket creation")
                elif dimension == QualityDimension.CLARITY:
                    recommendations.append("Provide writing guidelines and templates for ticket creation")
                elif dimension == QualityDimension.ACTIONABILITY:
                    recommendations.append("Require acceptance criteria for all tickets")
                elif dimension == QualityDimension.TECHNICAL_ACCURACY:
                    recommendations.append("Add technical review step for bug reports")
        
        # Add general recommendations
        approval_rate = sum(1 for r in reports if r.approval_status) / total_reports
        if approval_rate < 0.7:
            recommendations.append("Consider implementing pre-submission quality checks")
        
        return recommendations[:5]  # Limit to top 5 recommendations