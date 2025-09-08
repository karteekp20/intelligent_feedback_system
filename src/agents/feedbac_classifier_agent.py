"""
Feedback Classifier Agent for categorizing user feedback into different types.
Uses NLP techniques and OpenAI API for intelligent classification.
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
    AgentResult, FeedbackItem, FeedbackCategory, Classification, 
    ClassificationBatch, FeedbackSource, Priority
)
from ..core.nlp_utils import NLPUtils
from ..utils.logger import get_logger


class FeedbackClassifierAgent(BaseAgent):
    """Agent responsible for classifying feedback into categories using AI and NLP."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Feedback Classifier Agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("feedback_classifier", config)
        self.nlp_utils = NLPUtils()
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.confidence_threshold = config.get("confidence_threshold", 0.8) if config else 0.8
        self.use_few_shot_learning = config.get("use_few_shot", True) if config else True
        self.enable_category_weights = config.get("enable_weights", False) if config else False
        self.batch_size = config.get("batch_size", 10) if config else 10
        self.enable_sentiment_analysis = config.get("enable_sentiment", True) if config else True
        self.enable_emotion_detection = config.get("enable_emotion", True) if config else True
        self.enable_urgency_detection = config.get("enable_urgency", True) if config else True
        
        # OpenAI configuration
        self.openai_model = config.get("openai_model", "gpt-3.5-turbo") if config else "gpt-3.5-turbo"
        self.max_tokens = config.get("max_tokens", 500) if config else 500
        self.temperature = config.get("temperature", 0.3) if config else 0.3
        
        # Classification patterns and keywords
        self.category_patterns = self._load_category_patterns()
        self.sentiment_keywords = self._load_sentiment_keywords()
        self.emotion_keywords = self._load_emotion_keywords()
        self.urgency_keywords = self._load_urgency_keywords()
        
        # Few-shot learning examples
        self.few_shot_examples = self._load_few_shot_examples()
        
        # Category weights (if enabled)
        self.category_weights = self._load_category_weights() if self.enable_category_weights else {}
        
        # Statistics tracking
        self.classification_stats = {
            "total_classified": 0,
            "high_confidence": 0,
            "low_confidence": 0,
            "category_distribution": Counter(),
            "average_confidence": 0.0
        }
    
    def _load_category_patterns(self) -> Dict[FeedbackCategory, Dict[str, List[str]]]:
        """Load keyword patterns for each category."""
        return {
            FeedbackCategory.BUG: {
                "strong": [
                    "crash", "crashed", "crashing", "bug", "error", "broken", "not working",
                    "issue", "problem", "fails", "failed", "failure", "exception", "fault"
                ],
                "medium": [
                    "wrong", "incorrect", "unexpected", "strange", "weird", "glitch",
                    "freeze", "frozen", "hang", "stuck", "slow", "lag"
                ],
                "weak": [
                    "doesn't work", "won't load", "can't", "unable", "impossible"
                ]
            },
            FeedbackCategory.FEATURE_REQUEST: {
                "strong": [
                    "feature", "add", "implement", "include", "support", "would like",
                    "request", "suggestion", "enhance", "improvement", "upgrade"
                ],
                "medium": [
                    "wish", "hope", "could", "should", "need", "want", "missing",
                    "lacking", "would be nice", "please add"
                ],
                "weak": [
                    "better", "more", "additional", "extra", "option"
                ]
            },
            FeedbackCategory.PERFORMANCE_ISSUE: {
                "strong": [
                    "slow", "fast", "speed", "performance", "lag", "delay", "timeout",
                    "loading", "response time", "optimization"
                ],
                "medium": [
                    "quick", "faster", "slower", "wait", "waiting", "takes forever",
                    "too long", "inefficient"
                ],
                "weak": [
                    "time", "duration", "instant", "immediately"
                ]
            },
            FeedbackCategory.UI_UX_ISSUE: {
                "strong": [
                    "interface", "design", "layout", "ui", "ux", "user experience",
                    "usability", "navigation", "menu", "button", "confusing"
                ],
                "medium": [
                    "look", "appearance", "visual", "display", "screen", "page",
                    "difficult", "hard to use", "intuitive"
                ],
                "weak": [
                    "color", "font", "size", "position", "placement"
                ]
            },
            FeedbackCategory.SECURITY_CONCERN: {
                "strong": [
                    "security", "hack", "hacked", "vulnerability", "breach", "privacy",
                    "data", "personal information", "password", "login", "unauthorized"
                ],
                "medium": [
                    "safe", "secure", "protection", "encrypt", "authentication",
                    "permission", "access", "confidential"
                ],
                "weak": [
                    "trust", "worried", "concern", "risk"
                ]
            },
            FeedbackCategory.INTEGRATION_ISSUE: {
                "strong": [
                    "integration", "api", "connect", "sync", "synchronization",
                    "import", "export", "third party", "plugin", "addon"
                ],
                "medium": [
                    "compatible", "compatibility", "works with", "support for",
                    "link", "connection"
                ],
                "weak": [
                    "together", "combine", "merge"
                ]
            },
            FeedbackCategory.DOCUMENTATION_REQUEST: {
                "strong": [
                    "documentation", "docs", "help", "guide", "tutorial", "manual",
                    "instructions", "how to", "explain", "unclear"
                ],
                "medium": [
                    "understand", "confusing", "not clear", "example", "demo",
                    "walkthrough", "step by step"
                ],
                "weak": [
                    "question", "what", "how", "why", "when"
                ]
            },
            FeedbackCategory.PRAISE: {
                "strong": [
                    "love", "amazing", "excellent", "perfect", "fantastic", "awesome",
                    "great", "wonderful", "brilliant", "outstanding", "superb"
                ],
                "medium": [
                    "good", "nice", "like", "enjoy", "satisfied", "happy",
                    "pleased", "impressed", "appreciate"
                ],
                "weak": [
                    "ok", "fine", "decent", "acceptable", "not bad"
                ]
            },
            FeedbackCategory.COMPLAINT: {
                "strong": [
                    "hate", "terrible", "awful", "worst", "horrible", "disgusting",
                    "angry", "frustrated", "disappointed", "unacceptable"
                ],
                "medium": [
                    "bad", "poor", "dislike", "unhappy", "unsatisfied", "annoying",
                    "irritating", "bothered", "upset"
                ],
                "weak": [
                    "not good", "not great", "could be better", "mediocre"
                ]
            },
            FeedbackCategory.QUESTION: {
                "strong": [
                    "question", "how", "what", "why", "when", "where", "which",
                    "who", "can you", "could you", "is it possible"
                ],
                "medium": [
                    "wondering", "curious", "clarify", "explain", "help me understand",
                    "not sure", "confused"
                ],
                "weak": [
                    "maybe", "perhaps", "suppose", "think"
                ]
            },
            FeedbackCategory.SPAM: {
                "strong": [
                    "viagra", "casino", "lottery", "winner", "congratulations",
                    "click here", "free money", "earn money", "get rich"
                ],
                "medium": [
                    "promotion", "advertisement", "sale", "discount", "offer",
                    "limited time", "act now"
                ],
                "weak": [
                    "buy", "purchase", "order", "subscribe"
                ]
            }
        }
    
    def _load_sentiment_keywords(self) -> Dict[str, List[str]]:
        """Load sentiment analysis keywords."""
        return {
            "very_positive": [
                "love", "amazing", "excellent", "perfect", "fantastic", "awesome",
                "brilliant", "outstanding", "superb", "incredible"
            ],
            "positive": [
                "good", "great", "nice", "like", "enjoy", "satisfied", "happy",
                "pleased", "wonderful", "appreciate", "thank"
            ],
            "neutral": [
                "ok", "fine", "decent", "acceptable", "average", "normal", "standard"
            ],
            "negative": [
                "bad", "poor", "dislike", "unhappy", "unsatisfied", "annoying",
                "disappointed", "frustrated", "upset", "concerned"
            ],
            "very_negative": [
                "hate", "terrible", "awful", "worst", "horrible", "disgusting",
                "angry", "furious", "outraged", "unacceptable"
            ]
        }
    
    def _load_emotion_keywords(self) -> Dict[str, List[str]]:
        """Load emotion detection keywords."""
        return {
            "joy": ["happy", "excited", "thrilled", "delighted", "joyful", "cheerful"],
            "anger": ["angry", "mad", "furious", "irritated", "annoyed", "outraged"],
            "sadness": ["sad", "disappointed", "upset", "depressed", "unhappy"],
            "fear": ["scared", "afraid", "worried", "concerned", "anxious", "nervous"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned"],
            "disgust": ["disgusted", "revolted", "appalled", "horrified", "sickened"],
            "anticipation": ["excited", "eager", "hopeful", "optimistic", "looking forward"],
            "trust": ["confident", "sure", "certain", "reliable", "trustworthy"]
        }
    
    def _load_urgency_keywords(self) -> Dict[str, List[str]]:
        """Load urgency detection keywords."""
        return {
            "critical": [
                "urgent", "critical", "emergency", "asap", "immediately", "now",
                "right away", "can't wait", "blocking", "showstopper"
            ],
            "high": [
                "important", "priority", "soon", "quickly", "fast", "major",
                "significant", "serious", "affecting many"
            ],
            "medium": [
                "moderate", "normal", "standard", "regular", "typical", "when possible"
            ],
            "low": [
                "minor", "small", "little", "eventually", "sometime", "nice to have",
                "not urgent", "low priority", "future"
            ]
        }
    
    def _load_few_shot_examples(self) -> List[Dict[str, str]]:
        """Load few-shot learning examples for better classification."""
        return [
            {
                "text": "The app crashes every time I try to open a document",
                "category": "bug",
                "reasoning": "Contains crash indicator and specific failure scenario"
            },
            {
                "text": "It would be great if you could add dark mode support",
                "category": "feature_request",
                "reasoning": "Clear request for new functionality with 'add' keyword"
            },
            {
                "text": "The app is really slow when loading large files",
                "category": "performance_issue",
                "reasoning": "Mentions speed and loading performance issues"
            },
            {
                "text": "The interface is confusing and hard to navigate",
                "category": "ui_ux_issue",
                "reasoning": "References interface usability problems"
            },
            {
                "text": "I'm concerned about how my personal data is being stored",
                "category": "security_concern",
                "reasoning": "Mentions data privacy and security concerns"
            },
            {
                "text": "Love this app! It's exactly what I needed",
                "category": "praise",
                "reasoning": "Positive sentiment with 'love' and satisfaction indicators"
            },
            {
                "text": "This is terrible, I want my money back",
                "category": "complaint",
                "reasoning": "Strong negative sentiment with complaint indicators"
            },
            {
                "text": "How do I reset my password?",
                "category": "question",
                "reasoning": "Direct question with interrogative structure"
            }
        ]
    
    def _load_category_weights(self) -> Dict[FeedbackCategory, float]:
        """Load category weights for biased classification if enabled."""
        return {
            FeedbackCategory.BUG: 1.2,  # Slightly prioritize bug detection
            FeedbackCategory.SECURITY_CONCERN: 1.5,  # Highly prioritize security issues
            FeedbackCategory.FEATURE_REQUEST: 1.0,
            FeedbackCategory.PERFORMANCE_ISSUE: 1.1,
            FeedbackCategory.UI_UX_ISSUE: 1.0,
            FeedbackCategory.INTEGRATION_ISSUE: 1.0,
            FeedbackCategory.DOCUMENTATION_REQUEST: 0.9,
            FeedbackCategory.PRAISE: 0.8,
            FeedbackCategory.COMPLAINT: 1.0,
            FeedbackCategory.QUESTION: 0.9,
            FeedbackCategory.SPAM: 0.5,  # Deprioritize spam detection
            FeedbackCategory.OTHER: 0.7
        }
    
    async def process(self, data: Any) -> AgentResult:
        """
        Classify feedback items into categories.
        
        Args:
            data: List of FeedbackItem objects or single FeedbackItem
            
        Returns:
            AgentResult containing classifications
        """
        try:
            self.logger.info("Starting feedback classification process")
            start_time = datetime.now()
            
            # Handle different input types
            if isinstance(data, FeedbackItem):
                feedback_items = [data]
            elif isinstance(data, list):
                feedback_items = [item for item in data if isinstance(item, FeedbackItem)]
            else:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="Input must be FeedbackItem or list of FeedbackItems"
                )
            
            if not feedback_items:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error_message="No valid feedback items found for classification"
                )
            
            # Process in batches for better performance
            all_classifications = []
            processing_stats = {
                "total_items": len(feedback_items),
                "successful_classifications": 0,
                "failed_classifications": 0,
                "high_confidence_count": 0,
                "low_confidence_count": 0
            }
            
            for i in range(0, len(feedback_items), self.batch_size):
                batch = feedback_items[i:i + self.batch_size]
                
                try:
                    batch_classifications = await self._classify_batch(batch)
                    all_classifications.extend(batch_classifications)
                    
                    # Update statistics
                    for classification in batch_classifications:
                        if classification.confidence_score >= self.confidence_threshold:
                            processing_stats["high_confidence_count"] += 1
                        else:
                            processing_stats["low_confidence_count"] += 1
                    
                    processing_stats["successful_classifications"] += len(batch_classifications)
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                    processing_stats["failed_classifications"] += len(batch)
                    continue
            
            # Calculate processing time and statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if all_classifications:
                average_confidence = sum(c.confidence_score for c in all_classifications) / len(all_classifications)
                processing_stats["average_confidence"] = average_confidence
                
                # Update global statistics
                self._update_statistics(all_classifications)
            
            # Create classification batch for tracking
            classification_batch = ClassificationBatch(
                batch_id=f"CLASSIFY-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                classifications=all_classifications,
                processing_time=processing_time,
                model_version=self.openai_model
            )
            
            result_data = {
                "classifications": all_classifications,
                "classification_batch": classification_batch,
                "processing_stats": processing_stats
            }
            
            self.logger.info(f"Classification completed. Processed {len(all_classifications)} items "
                           f"with average confidence {processing_stats.get('average_confidence', 0):.2f}")
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=result_data,
                confidence=processing_stats.get('average_confidence', 0),
                details=f"Classified {len(all_classifications)} feedback items",
                processing_time=processing_time,
                metadata={
                    "processing_stats": processing_stats,
                    "batch_count": (len(feedback_items) + self.batch_size - 1) // self.batch_size
                }
            )
            
        except Exception as e:
            self.logger.error(f"Classification process failed: {str(e)}", exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=f"Classification failed: {str(e)}"
            )
    
    async def _classify_batch(self, feedback_items: List[FeedbackItem]) -> List[Classification]:
        """
        Classify a batch of feedback items.
        
        Args:
            feedback_items: List of feedback items to classify
            
        Returns:
            List of Classification objects
        """
        classifications = []
        
        # Use concurrent processing for better performance
        classification_tasks = [
            self._classify_single_item(item) for item in feedback_items
        ]
        
        classification_results = await asyncio.gather(*classification_tasks, return_exceptions=True)
        
        for i, result in enumerate(classification_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error classifying item {feedback_items[i].id}: {str(result)}")
                # Create low confidence classification
                classifications.append(
                    Classification.create_low_confidence(
                        feedback_id=feedback_items[i].id,
                        category=FeedbackCategory.OTHER,
                        confidence=0.0,
                        reason=f"Classification failed: {str(result)}"
                    )
                )
            else:
                classifications.append(result)
        
        return classifications
    
    async def _classify_single_item(self, feedback_item: FeedbackItem) -> Classification:
        """
        Classify a single feedback item.
        
        Args:
            feedback_item: Feedback item to classify
            
        Returns:
            Classification object
        """
        try:
            # First, try rule-based classification
            rule_based_result = await self._rule_based_classification(feedback_item)
            
            # If rule-based classification has high confidence, use it
            if rule_based_result['confidence'] >= 0.9:
                classification = Classification(
                    feedback_id=feedback_item.id,
                    predicted_category=rule_based_result['category'],
                    confidence_score=rule_based_result['confidence'],
                    category_scores=rule_based_result['category_scores'],
                    reasoning=rule_based_result['reasoning'],
                    keywords_found=rule_based_result['keywords'],
                    classifier_version="rule_based_v1.0"
                )
            else:
                # Use AI-based classification for better accuracy
                ai_result = await self._ai_based_classification(feedback_item)
                
                # Combine results for final decision
                classification = await self._combine_classification_results(
                    feedback_item, rule_based_result, ai_result
                )
            
            # Add additional analysis
            if self.enable_sentiment_analysis:
                classification.sentiment_score = await self._analyze_sentiment(feedback_item.content)
            
            if self.enable_emotion_detection:
                classification.emotion_detected = await self._detect_emotion(feedback_item.content)
            
            if self.enable_urgency_detection:
                classification.urgency_level = await self._detect_urgency(feedback_item.content)
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Error classifying feedback {feedback_item.id}: {str(e)}")
            return Classification.create_low_confidence(
                feedback_id=feedback_item.id,
                category=FeedbackCategory.OTHER,
                confidence=0.0,
                reason=f"Classification error: {str(e)}"
            )
    
    async def _rule_based_classification(self, feedback_item: FeedbackItem) -> Dict[str, Any]:
        """
        Perform rule-based classification using keyword patterns.
        
        Args:
            feedback_item: Feedback item to classify
            
        Returns:
            Dictionary with classification results
        """
        content = feedback_item.content.lower()
        category_scores = {}
        keywords_found = []
        
        # Calculate scores for each category
        for category, patterns in self.category_patterns.items():
            score = 0.0
            category_keywords = []
            
            # Strong indicators (weight: 3)
            for keyword in patterns.get("strong", []):
                if keyword in content:
                    score += 3.0
                    category_keywords.append(keyword)
            
            # Medium indicators (weight: 2)
            for keyword in patterns.get("medium", []):
                if keyword in content:
                    score += 2.0
                    category_keywords.append(keyword)
            
            # Weak indicators (weight: 1)
            for keyword in patterns.get("weak", []):
                if keyword in content:
                    score += 1.0
                    category_keywords.append(keyword)
            
            # Apply category weights if enabled
            if self.enable_category_weights and category in self.category_weights:
                score *= self.category_weights[category]
            
            category_scores[category] = score
            if category_keywords:
                keywords_found.extend(category_keywords)
        
        # Normalize scores
        max_score = max(category_scores.values()) if category_scores.values() else 0
        if max_score > 0:
            category_scores = {cat: score/max_score for cat, score in category_scores.items()}
        
        # Find best category
        best_category = max(category_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on score separation
        sorted_scores = sorted(category_scores.values(), reverse=True)
        confidence = best_category[1]
        
        if len(sorted_scores) > 1:
            # Adjust confidence based on separation from second best
            separation = sorted_scores[0] - sorted_scores[1]
            confidence = min(confidence + separation * 0.2, 1.0)
        
        # Generate reasoning
        reasoning = f"Rule-based classification found {len(keywords_found)} matching keywords"
        if keywords_found[:3]:  # Show top 3 keywords
            reasoning += f": {', '.join(keywords_found[:3])}"
        
        return {
            'category': best_category[0],
            'confidence': confidence,
            'category_scores': category_scores,
            'reasoning': reasoning,
            'keywords': keywords_found[:10]  # Limit to top 10 keywords
        }
    
    async def _ai_based_classification(self, feedback_item: FeedbackItem) -> Dict[str, Any]:
        """
        Perform AI-based classification using OpenAI API.
        
        Args:
            feedback_item: Feedback item to classify
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Prepare the prompt
            prompt = self._create_classification_prompt(feedback_item)
            
            # Call OpenAI API
            response = await openai.ChatCompletion.acreate(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30
            )
            
            # Parse response
            result = self._parse_ai_response(response.choices[0].message.content)
            result['reasoning'] = f"AI classification using {self.openai_model}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"AI classification failed: {str(e)}")
            return {
                'category': FeedbackCategory.OTHER,
                'confidence': 0.0,
                'category_scores': {},
                'reasoning': f"AI classification failed: {str(e)}",
                'keywords': []
            }
    
    def _create_classification_prompt(self, feedback_item: FeedbackItem) -> str:
        """Create classification prompt for OpenAI API."""
        prompt = f"""Classify the following user feedback into one of these categories:

Categories:
- bug: Software defects, crashes, errors, broken functionality
- feature_request: Requests for new features or enhancements
- performance_issue: Speed, responsiveness, or efficiency problems
- ui_ux_issue: User interface or user experience problems
- security_concern: Privacy, security, or data protection issues
- integration_issue: Problems with third-party integrations or APIs
- documentation_request: Requests for help, guides, or documentation
- praise: Positive feedback, compliments, satisfaction
- complaint: General complaints or dissatisfaction
- question: Questions or requests for information
- spam: Promotional content or irrelevant messages
- other: Feedback that doesn't fit other categories

Feedback Content: "{feedback_item.content}"

Source: {feedback_item.source.get_display_name()}"""

        # Add few-shot examples if enabled
        if self.use_few_shot_learning:
            examples_text = "\n\nExamples:\n"
            for example in self.few_shot_examples[:3]:  # Use top 3 examples
                examples_text += f'"{example["text"]}" -> {example["category"]} ({example["reasoning"]})\n'
            prompt += examples_text

        prompt += """

Please respond with a JSON object containing:
{
  "category": "predicted_category",
  "confidence": 0.85,
  "reasoning": "explanation of classification",
  "category_scores": {
    "bug": 0.1,
    "feature_request": 0.85,
    "other": 0.05
  }
}"""

        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for OpenAI API."""
        return """You are an expert feedback classifier. Your job is to accurately categorize user feedback based on content, context, and intent. Consider:

1. The primary intent of the feedback
2. Keywords and phrases that indicate category
3. The emotional tone and urgency level
4. The source of the feedback (app store vs support email vs survey)

Provide accurate classifications with high confidence scores only when you're certain. If uncertain, use lower confidence scores and explain your reasoning."""
    
    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI response and extract classification results."""
        try:
            # Try to parse JSON response
            result = json.loads(response_content)
            
            # Validate and convert category
            category_str = result.get('category', 'other')
            category = FeedbackCategory.from_string(category_str)
            
            # Validate confidence
            confidence = float(result.get('confidence', 0.0))
            confidence = max(0.0, min(1.0, confidence))
            
            # Parse category scores
            category_scores = result.get('category_scores', {})
            parsed_scores = {}
            for cat_str, score in category_scores.items():
                try:
                    cat = FeedbackCategory.from_string(cat_str)
                    parsed_scores[cat] = float(score)
                except:
                    continue
            
            return {
                'category': category,
                'confidence': confidence,
                'category_scores': parsed_scores,
                'reasoning': result.get('reasoning', 'AI classification'),
                'keywords': []
            }
            
        except json.JSONDecodeError:
            # Fallback parsing for non-JSON responses
            self.logger.warning("AI response not in JSON format, attempting fallback parsing")
            return self._fallback_parse_response(response_content)
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {str(e)}")
            return {
                'category': FeedbackCategory.OTHER,
                'confidence': 0.0,
                'category_scores': {},
                'reasoning': f"Response parsing failed: {str(e)}",
                'keywords': []
            }
    
    def _fallback_parse_response(self, response_content: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON AI responses."""
        content = response_content.lower()
        
        # Try to find category mentions
        for category in FeedbackCategory:
            if category.value in content or category.value.replace('_', ' ') in content:
                # Extract confidence if mentioned
                confidence_match = re.search(r'confidence[:\s]*(\d+\.?\d*)', content)
                confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.5
                
                return {
                    'category': category,
                    'confidence': confidence,
                    'category_scores': {category: confidence},
                    'reasoning': 'Fallback parsing from AI response',
                    'keywords': []
                }
        
        return {
            'category': FeedbackCategory.OTHER,
            'confidence': 0.0,
            'category_scores': {},
            'reasoning': 'Could not parse AI response',
            'keywords': []
        }
    
    async def _combine_classification_results(self, feedback_item: FeedbackItem, 
                                            rule_result: Dict[str, Any], 
                                            ai_result: Dict[str, Any]) -> Classification:
        """
        Combine rule-based and AI-based classification results.
        
        Args:
            feedback_item: Original feedback item
            rule_result: Rule-based classification result
            ai_result: AI-based classification result
            
        Returns:
            Combined Classification object
        """
        # Weight the results (AI typically more accurate but rules provide good keywords)
        rule_weight = 0.3
        ai_weight = 0.7
        
        # If AI failed, use rule-based result
        if ai_result['confidence'] == 0.0:
            final_category = rule_result['category']
            final_confidence = rule_result['confidence']
            final_reasoning = f"Rule-based only: {rule_result['reasoning']}"
            keywords_found = rule_result['keywords']
        
        # If rule-based failed, use AI result
        elif rule_result['confidence'] == 0.0:
            final_category = ai_result['category']
            final_confidence = ai_result['confidence']
            final_reasoning = f"AI-based only: {ai_result['reasoning']}"
            keywords_found = ai_result.get('keywords', [])
        
        # If both agree on category, boost confidence
        elif rule_result['category'] == ai_result['category']:
            final_category = ai_result['category']
            final_confidence = min(
                rule_result['confidence'] * rule_weight + ai_result['confidence'] * ai_weight + 0.1,
                1.0
            )
            final_reasoning = f"Both methods agree: {ai_result['reasoning']}"
            keywords_found = rule_result['keywords']
        
        # If they disagree, use the one with higher confidence
        else:
            if ai_result['confidence'] > rule_result['confidence']:
                final_category = ai_result['category']
                final_confidence = ai_result['confidence'] * 0.8  # Reduce due to disagreement
                final_reasoning = f"AI preferred over rules: {ai_result['reasoning']}"
                keywords_found = rule_result['keywords']  # Keep rule keywords for insight
            else:
                final_category = rule_result['category']
                final_confidence = rule_result['confidence'] * 0.8  # Reduce due to disagreement
                final_reasoning = f"Rules preferred over AI: {rule_result['reasoning']}"
                keywords_found = rule_result['keywords']
        
        # Combine category scores
        combined_scores = {}
        for category in FeedbackCategory:
            rule_score = rule_result['category_scores'].get(category, 0.0)
            ai_score = ai_result['category_scores'].get(category, 0.0)
            combined_scores[category] = rule_score * rule_weight + ai_score * ai_weight
        
        # Create classification
        classification = Classification(
            feedback_id=feedback_item.id,
            predicted_category=final_category,
            confidence_score=final_confidence,
            category_scores=combined_scores,
            reasoning=final_reasoning,
            keywords_found=keywords_found,
            classifier_version=f"combined_v1.0_{self.openai_model}"
        )
        
        return classification
    
    async def _analyze_sentiment(self, content: str) -> float:
        """
        Analyze sentiment of the content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Sentiment score from -1.0 (very negative) to 1.0 (very positive)
        """
        try:
            content_lower = content.lower()
            sentiment_score = 0.0
            total_indicators = 0
            
            # Check sentiment keywords
            for sentiment_type, keywords in self.sentiment_keywords.items():
                for keyword in keywords:
                    if keyword in content_lower:
                        total_indicators += 1
                        if sentiment_type == "very_positive":
                            sentiment_score += 1.0
                        elif sentiment_type == "positive":
                            sentiment_score += 0.5
                        elif sentiment_type == "neutral":
                            sentiment_score += 0.0
                        elif sentiment_type == "negative":
                            sentiment_score -= 0.5
                        elif sentiment_type == "very_negative":
                            sentiment_score -= 1.0
            
            # Normalize sentiment score
            if total_indicators > 0:
                sentiment_score = sentiment_score / total_indicators
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Use NLP utils if available for more sophisticated analysis
            try:
                nlp_sentiment = await self.nlp_utils.analyze_sentiment(content)
                # Combine with keyword-based analysis
                sentiment_score = (sentiment_score + nlp_sentiment) / 2
            except:
                pass  # Fall back to keyword-based analysis
            
            return sentiment_score
            
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {str(e)}")
            return 0.0
    
    async def _detect_emotion(self, content: str) -> Optional[str]:
        """
        Detect primary emotion in the content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Detected emotion or None
        """
        try:
            content_lower = content.lower()
            emotion_scores = {}
            
            # Check emotion keywords
            for emotion, keywords in self.emotion_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in content_lower:
                        score += 1
                
                if score > 0:
                    emotion_scores[emotion] = score
            
            # Return emotion with highest score
            if emotion_scores:
                return max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Emotion detection failed: {str(e)}")
            return None
    
    async def _detect_urgency(self, content: str) -> Optional[str]:
        """
        Detect urgency level in the content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Urgency level (critical, high, medium, low) or None
        """
        try:
            content_lower = content.lower()
            urgency_scores = {}
            
            # Check urgency keywords
            for urgency_level, keywords in self.urgency_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in content_lower:
                        score += 1
                
                if score > 0:
                    urgency_scores[urgency_level] = score
            
            # Return urgency level with highest score
            if urgency_scores:
                return max(urgency_scores.items(), key=lambda x: x[1])[0]
            
            return "medium"  # Default urgency level
            
        except Exception as e:
            self.logger.warning(f"Urgency detection failed: {str(e)}")
            return None
    
    def _update_statistics(self, classifications: List[Classification]):
        """Update internal statistics with new classifications."""
        self.classification_stats["total_classified"] += len(classifications)
        
        for classification in classifications:
            # Update confidence statistics
            if classification.confidence_score >= self.confidence_threshold:
                self.classification_stats["high_confidence"] += 1
            else:
                self.classification_stats["low_confidence"] += 1
            
            # Update category distribution
            self.classification_stats["category_distribution"][classification.predicted_category] += 1
        
        # Update average confidence
        if classifications:
            total_confidence = sum(c.confidence_score for c in classifications)
            new_avg = total_confidence / len(classifications)
            
            # Calculate running average
            total_items = self.classification_stats["total_classified"]
            old_weight = (total_items - len(classifications)) / total_items
            new_weight = len(classifications) / total_items
            
            self.classification_stats["average_confidence"] = (
                self.classification_stats["average_confidence"] * old_weight + 
                new_avg * new_weight
            )
    
    async def classify_single(self, feedback_item: FeedbackItem) -> Classification:
        """
        Classify a single feedback item (convenience method).
        
        Args:
            feedback_item: Feedback item to classify
            
        Returns:
            Classification object
        """
        result = await self.process(feedback_item)
        if result.success and result.data:
            return result.data["classifications"][0]
        else:
            return Classification.create_low_confidence(
                feedback_id=feedback_item.id,
                category=FeedbackCategory.OTHER,
                confidence=0.0,
                reason="Classification failed"
            )
    
    async def bulk_classify(self, feedback_items: List[FeedbackItem], 
                          callback: Optional[callable] = None) -> List[Classification]:
        """
        Classify multiple feedback items with progress callback.
        
        Args:
            feedback_items: List of feedback items to classify
            callback: Optional progress callback function
            
        Returns:
            List of Classification objects
        """
        all_classifications = []
        total_batches = (len(feedback_items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(feedback_items), self.batch_size):
            batch = feedback_items[batch_idx:batch_idx + self.batch_size]
            
            try:
                batch_classifications = await self._classify_batch(batch)
                all_classifications.extend(batch_classifications)
                
                # Call progress callback if provided
                if callback:
                    progress = (batch_idx // self.batch_size + 1) / total_batches
                    await callback(progress, len(all_classifications), len(feedback_items))
                    
            except Exception as e:
                self.logger.error(f"Error in bulk classification batch {batch_idx}: {str(e)}")
                continue
        
        return all_classifications
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get current classification statistics."""
        stats = self.classification_stats.copy()
        
        # Add percentage calculations
        total = stats["total_classified"]
        if total > 0:
            stats["high_confidence_percentage"] = (stats["high_confidence"] / total) * 100
            stats["low_confidence_percentage"] = (stats["low_confidence"] / total) * 100
            
            # Convert category distribution to percentages
            category_percentages = {}
            for category, count in stats["category_distribution"].items():
                category_percentages[category.value] = (count / total) * 100
            stats["category_percentages"] = category_percentages
        
        return stats
    
    async def retrain_model(self, feedback_data: List[Tuple[FeedbackItem, FeedbackCategory]]):
        """
        Retrain/update the classification model with new labeled data.
        
        Args:
            feedback_data: List of (feedback_item, correct_category) tuples
        """
        self.logger.info(f"Retraining classification model with {len(feedback_data)} examples")
        
        # Update few-shot examples with new data
        for feedback_item, correct_category in feedback_data[-10:]:  # Use last 10 examples
            example = {
                "text": feedback_item.content,
                "category": correct_category.value,
                "reasoning": f"User-corrected classification for {feedback_item.source.value}"
            }
            
            # Replace oldest examples
            if len(self.few_shot_examples) >= 10:
                self.few_shot_examples.pop(0)
            self.few_shot_examples.append(example)
        
        # Update keyword patterns based on successful classifications
        await self._update_keyword_patterns(feedback_data)
        
        self.logger.info("Model retraining completed")
    
    async def _update_keyword_patterns(self, feedback_data: List[Tuple[FeedbackItem, FeedbackCategory]]):
        """Update keyword patterns based on new training data."""
        try:
            # Analyze successful classifications to extract new patterns
            for feedback_item, correct_category in feedback_data:
                content_words = set(feedback_item.content.lower().split())
                
                # Find words that might be good indicators for this category
                current_patterns = self.category_patterns.get(correct_category, {"strong": [], "medium": [], "weak": []})
                
                # Simple heuristic: if a word appears in this category but not in others, it's a good indicator
                for word in content_words:
                    if len(word) > 3 and word.isalpha():  # Filter meaningful words
                        if word not in sum([p.get("strong", []) + p.get("medium", []) + p.get("weak", []) 
                                          for p in self.category_patterns.values()], []):
                            # Add as weak indicator for now
                            if word not in current_patterns["weak"]:
                                current_patterns["weak"].append(word)
                                
                                # Limit pattern size
                                if len(current_patterns["weak"]) > 20:
                                    current_patterns["weak"] = current_patterns["weak"][-20:]
            
        except Exception as e:
            self.logger.warning(f"Failed to update keyword patterns: {str(e)}")
    
    async def evaluate_performance(self, test_data: List[Tuple[FeedbackItem, FeedbackCategory]]) -> Dict[str, float]:
        """
        Evaluate classification performance on test data.
        
        Args:
            test_data: List of (feedback_item, expected_category) tuples
            
        Returns:
            Performance metrics dictionary
        """
        if not test_data:
            return {"error": "No test data provided"}
        
        correct_predictions = 0
        total_predictions = len(test_data)
        category_correct = Counter()
        category_total = Counter()
        confidence_scores = []
        
        for feedback_item, expected_category in test_data:
            try:
                classification = await self.classify_single(feedback_item)
                
                category_total[expected_category] += 1
                confidence_scores.append(classification.confidence_score)
                
                if classification.predicted_category == expected_category:
                    correct_predictions += 1
                    category_correct[expected_category] += 1
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {feedback_item.id}: {str(e)}")
                continue
        
        # Calculate metrics
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        average_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Per-category accuracy
        category_accuracy = {}
        for category in category_total:
            category_accuracy[category.value] = (
                category_correct[category] / category_total[category] 
                if category_total[category] > 0 else 0
            )
        
        return {
            "overall_accuracy": overall_accuracy,
            "average_confidence": average_confidence,
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "category_accuracy": category_accuracy,
            "confidence_distribution": {
                "high_confidence": len([c for c in confidence_scores if c >= 0.8]),
                "medium_confidence": len([c for c in confidence_scores if 0.5 <= c < 0.8]),
                "low_confidence": len([c for c in confidence_scores if c < 0.5])
            }
        }
    
    def export_model_config(self) -> Dict[str, Any]:
        """Export current model configuration for backup/sharing."""
        return {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "confidence_threshold": self.confidence_threshold,
                "use_few_shot_learning": self.use_few_shot_learning,
                "enable_category_weights": self.enable_category_weights,
                "openai_model": self.openai_model,
                "temperature": self.temperature
            },
            "category_patterns": {
                cat.value: patterns for cat, patterns in self.category_patterns.items()
            },
            "few_shot_examples": self.few_shot_examples,
            "category_weights": {
                cat.value: weight for cat, weight in self.category_weights.items()
            } if self.category_weights else {},
            "statistics": self.classification_stats
        }
    
    def import_model_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Import model configuration from backup.
        
        Args:
            config_data: Configuration data to import
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            # Import configuration
            if "config" in config_data:
                config = config_data["config"]
                self.confidence_threshold = config.get("confidence_threshold", self.confidence_threshold)
                self.use_few_shot_learning = config.get("use_few_shot_learning", self.use_few_shot_learning)
                self.enable_category_weights = config.get("enable_category_weights", self.enable_category_weights)
                self.openai_model = config.get("openai_model", self.openai_model)
                self.temperature = config.get("temperature", self.temperature)
            
            # Import patterns
            if "category_patterns" in config_data:
                patterns_data = config_data["category_patterns"]
                new_patterns = {}
                for cat_str, patterns in patterns_data.items():
                    try:
                        category = FeedbackCategory(cat_str)
                        new_patterns[category] = patterns
                    except ValueError:
                        continue
                self.category_patterns.update(new_patterns)
            
            # Import few-shot examples
            if "few_shot_examples" in config_data:
                self.few_shot_examples = config_data["few_shot_examples"]
            
            # Import category weights
            if "category_weights" in config_data and config_data["category_weights"]:
                weights_data = config_data["category_weights"]
                new_weights = {}
                for cat_str, weight in weights_data.items():
                    try:
                        category = FeedbackCategory(cat_str)
                        new_weights[category] = float(weight)
                    except (ValueError, TypeError):
                        continue
                self.category_weights = new_weights
            
            self.logger.info("Model configuration imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import model configuration: {str(e)}")
            return False