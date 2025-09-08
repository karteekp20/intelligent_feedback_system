"""
NLP utilities for text processing and analysis.
"""

import re
import string
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

from ..utils.logger import get_logger


class NLPUtils:
    """Utility class for NLP operations."""
    
    def __init__(self):
        """Initialize NLP utilities and download required NLTK data."""
        self.logger = get_logger("nlp_utils")
        self._download_nltk_data()
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.logger.warning("NLTK stopwords not available, using basic set")
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
                'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                'with', 'through', 'during', 'before', 'after', 'above', 'below',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        # Technical and app-specific terms to preserve
        self.preserve_terms = {
            'app', 'bug', 'crash', 'error', 'login', 'sync', 'ui', 'ux', 'api',
            'feature', 'update', 'version', 'android', 'ios', 'iphone', 'ipad'
        }
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'averaged_perceptron_tagger'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                    self.logger.debug(f"Downloaded NLTK data: {data}")
                except Exception as e:
                    self.logger.warning(f"Failed to download NLTK data {data}: {e}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and normalizing.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve punctuation that might be meaningful
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Fallback tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [
                token for token in tokens 
                if token not in self.stop_words or token in self.preserve_terms
            ]
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from text using frequency analysis.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of (keyword, score) tuples sorted by importance
        """
        # Clean and tokenize
        clean_text = self.clean_text(text)
        tokens = self.tokenize(clean_text, remove_stopwords=True)
        
        if not tokens:
            return []
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Calculate TF scores
        total_tokens = len(tokens)
        tf_scores = {
            token: count / total_tokens 
            for token, count in token_counts.items()
        }
        
        # Boost scores for technical terms
        for token in tf_scores:
            if token in self.preserve_terms:
                tf_scores[token] *= 2.0
            
            # Boost longer tokens (likely more specific)
            if len(token) > 6:
                tf_scores[token] *= 1.5
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_keywords[:max_keywords]
    
    def extract_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """
        Extract meaningful phrases from text.
        
        Args:
            text: Input text
            min_length: Minimum phrase length in words
            max_length: Maximum phrase length in words
            
        Returns:
            List of extracted phrases
        """
        # Clean text
        clean_text = self.clean_text(text)
        
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(clean_text)
        except LookupError:
            sentences = re.split(r'[.!?]+', clean_text)
        
        phrases = []
        
        for sentence in sentences:
            tokens = self.tokenize(sentence, remove_stopwords=False)
            
            # Extract n-grams
            for n in range(min_length, max_length + 1):
                for i in range(len(tokens) - n + 1):
                    phrase_tokens = tokens[i:i+n]
                    
                    # Skip phrases that are mostly stopwords
                    meaningful_tokens = [
                        token for token in phrase_tokens 
                        if token not in self.stop_words
                    ]
                    
                    if len(meaningful_tokens) >= min(2, n):
                        phrase = ' '.join(phrase_tokens)
                        phrases.append(phrase)
        
        # Remove duplicates and return most common
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(20)]
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using multiple methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {"polarity": 0.0, "subjectivity": 0.0, "compound": 0.0}
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
        except Exception:
            polarity, subjectivity = 0.0, 0.0
        
        # VADER sentiment (if available)
        compound = 0.0
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            vader_scores = sia.polarity_scores(text)
            compound = vader_scores['compound']
        except Exception:
            # Fallback: simple keyword-based sentiment
            compound = self._simple_sentiment(text)
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "compound": compound
        }
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis."""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'like',
            'perfect', 'wonderful', 'fantastic', 'brilliant', 'outstanding'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'horrible', 'worst', 'disgusting',
            'disappointed', 'frustrated', 'angry', 'broken', 'useless'
        }
        
        tokens = self.tokenize(text.lower())
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_sentiment_words
    
    def extract_technical_details(self, text: str) -> Dict[str, List[str]]:
        """
        Extract technical details like version numbers, error codes, etc.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with categorized technical details
        """
        details = {
            "versions": [],
            "error_codes": [],
            "platforms": [],
            "devices": [],
            "steps": []
        }
        
        # Version patterns
        version_patterns = [
            r'\bv?(\d+\.\d+(?:\.\d+)?(?:\.\d+)?)\b',
            r'\bversion\s+(\d+\.\d+(?:\.\d+)?)\b'
        ]
        
        for pattern in version_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            details["versions"].extend(matches)
        
        # Error code patterns
        error_patterns = [
            r'\berror\s+(?:code\s+)?(\d+)\b',
            r'\b(E\d{3,4})\b',
            r'\b(\d{3,4}\s+error)\b'
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            details["error_codes"].extend(matches)
        
        # Platform detection
        platform_keywords = {
            'ios': ['ios', 'iphone', 'ipad', 'apple'],
            'android': ['android', 'google play', 'samsung', 'pixel'],
            'web': ['browser', 'chrome', 'firefox', 'safari', 'edge'],
            'windows': ['windows', 'pc', 'desktop']
        }
        
        text_lower = text.lower()
        for platform, keywords in platform_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                details["platforms"].append(platform)
        
        # Device mentions
        device_patterns = [
            r'\b(iphone\s+\d+(?:\s+\w+)?)\b',
            r'\b(ipad\s+\w+)\b',
            r'\b(galaxy\s+\w+)\b',
            r'\b(pixel\s+\d+)\b'
        ]
        
        for pattern in device_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            details["devices"].extend(matches)
        
        # Steps to reproduce
        step_indicators = [
            'steps:', 'reproduce:', 'how to:', 'to reproduce:',
            '1.', '2.', '3.', 'first', 'then', 'next', 'after'
        ]
        
        if any(indicator in text.lower() for indicator in step_indicators):
            # Try to extract numbered steps
            step_patterns = [
                r'(\d+\.\s+[^.]+\.)',
                r'(step\s+\d+[:\-]\s+[^.]+\.)'
            ]
            
            for pattern in step_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                details["steps"].extend(matches)
        
        # Clean up results
        for key in details:
            details[key] = list(set(details[key]))  # Remove duplicates
        
        return details
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize both texts
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            blob = TextBlob(text)
            return blob.detect_language()
        except Exception:
            # Default to English if detection fails
            return 'en'
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Get basic statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "unique_words": 0,
                "avg_word_length": 0
            }
        
        # Character count
        char_count = len(text)
        
        # Word count
        words = self.tokenize(text, remove_stopwords=False)
        word_count = len(words)
        unique_words = len(set(words))
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence count
        try:
            sentences = sent_tokenize(text)
            sentence_count = len(sentences)
        except LookupError:
            sentence_count = len(re.split(r'[.!?]+', text))
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "unique_words": unique_words,
            "avg_word_length": round(avg_word_length, 2)
        }