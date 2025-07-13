# src/preprocessing/email_preprocessor.py
"""
Email preprocessing module for feature extraction and text cleaning
"""

import re
import nltk
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)


class EmailPreprocessor:
    """Preprocesses emails and extracts features for phishing detection"""
    
    def __init__(self):
        """Initialize preprocessor with keywords and patterns"""
        self.stop_words = set(stopwords.words('english'))
        
        # Phishing indicators
        self.urgency_keywords = [
            'urgent', 'immediate', 'expire', 'suspend', 'verify',
            'confirm', 'update', 'act now', 'limited time', 'expire soon',
            'deadline', 'within hours', 'immediately', 'asap'
        ]
        
        self.financial_keywords = [
            'bank', 'account', 'credit', 'payment', 'money',
            'transfer', 'transaction', 'reward', 'prize', 'cash',
            'dollar', 'pounds', 'euros', 'bitcoin', 'cryptocurrency',
            'invoice', 'billing', 'refund', 'tax'
        ]
        
        self.credential_keywords = [
            'password', 'username', 'login', 'signin', 'credential',
            'pin', 'ssn', 'social security', 'passport', 'identification',
            'verify identity', 'confirm identity'
        ]
        
        self.action_keywords = [
            'click', 'click here', 'click below', 'follow link',
            'visit', 'go to', 'log in', 'sign in', 'download',
            'open attachment', 'view document'
        ]
        
        # Suspicious patterns
        self.suspicious_patterns = [
            r'click\s+here',
            r'verify\s+your\s+account',
            r'suspend(ed)?\s+account',
            r'confirm\s+your\s+identity',
            r'update\s+your\s+information',
            r'unauthorized\s+access',
            r'security\s+alert',
            r'winner|congratulations',
            r'tax\s+refund',
            r'limited\s+time\s+offer'
        ]
        
        # URL shorteners
        self.url_shorteners = [
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'short.link',
            'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'bit.do'
        ]
        
        logger.info("Initialized email preprocessor")
    
    def preprocess_email(self, email_text: str, urls: List[str] = None) -> Dict:
        """
        Preprocess email and extract all features
        
        Args:
            email_text: Raw email text
            urls: List of URLs found in email
            
        Returns:
            Dictionary containing cleaned text and features
        """
        # Clean text
        cleaned_text = self.clean_text(email_text)
        
        # Extract features
        text_features = self.extract_text_features(cleaned_text)
        keyword_features = self.extract_keyword_features(cleaned_text)
        pattern_features = self.extract_pattern_features(cleaned_text)
        
        # Extract URL features if provided
        url_features = {}
        if urls:
            url_features = self.extract_url_features(urls)
        
        # Combine all features
        features = {
            **text_features,
            **keyword_features,
            **pattern_features,
            **url_features
        }
        
        return {
            'cleaned_text': cleaned_text,
            'features': features,
            'feature_vector': self._create_feature_vector(features)
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean email text
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase for processing
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove email headers
        text = re.sub(r'^(from|to|subject|date|cc|bcc):\s*.*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs (but keep them for separate analysis)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract text-based features"""
        features = {}
        
        # Basic metrics
        features['text_length'] = len(text)
        features['word_count'] = len(word_tokenize(text))
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Average word and sentence length
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text) if len(text) > 100 else 0
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text) if len(text) > 100 else 0
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
        
        # Punctuation density
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        features['punctuation_density'] = punctuation_count / len(text) if len(text) > 0 else 0
        
        # Capital letter ratio (in original text)
        features['capital_ratio'] = sum(1 for char in text if char.isupper()) / len(text) if len(text) > 0 else 0
        
        # Digit ratio
        features['digit_ratio'] = sum(1 for char in text if char.isdigit()) / len(text) if len(text) > 0 else 0
        
        return features
    
    def extract_keyword_features(self, text: str) -> Dict[str, float]:
        """Extract keyword-based features"""
        features = {}
        text_lower = text.lower()
        
        # Urgency score
        urgency_count = sum(1 for keyword in self.urgency_keywords 
                          if keyword.lower() in text_lower)
        features['urgency_score'] = urgency_count / len(self.urgency_keywords)
        features['urgency_count'] = urgency_count
        
        # Financial score
        financial_count = sum(1 for keyword in self.financial_keywords 
                            if keyword.lower() in text_lower)
        features['financial_score'] = financial_count / len(self.financial_keywords)
        features['financial_count'] = financial_count
        
        # Credential score
        credential_count = sum(1 for keyword in self.credential_keywords 
                             if keyword.lower() in text_lower)
        features['credential_score'] = credential_count / len(self.credential_keywords)
        features['credential_count'] = credential_count
        
        # Action score
        action_count = sum(1 for keyword in self.action_keywords 
                         if keyword.lower() in text_lower)
        features['action_score'] = action_count / len(self.action_keywords)
        features['action_count'] = action_count
        
        # Combined suspicious keyword score
        total_suspicious = urgency_count + financial_count + credential_count + action_count
        total_keywords = len(self.urgency_keywords) + len(self.financial_keywords) + \
                        len(self.credential_keywords) + len(self.action_keywords)
        features['overall_keyword_score'] = total_suspicious / total_keywords if total_keywords > 0 else 0
        
        return features
    
    def extract_pattern_features(self, text: str) -> Dict[str, float]:
        """Extract pattern-based features"""
        features = {}
        
        # Count suspicious patterns
        pattern_matches = 0
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pattern_matches += len(matches)
        
        features['suspicious_pattern_count'] = pattern_matches
        features['suspicious_pattern_density'] = pattern_matches / len(text.split()) if text else 0
        
        # Check for common phishing phrases
        features['has_verify_account'] = 1 if re.search(r'verify\s+your\s+account', text, re.IGNORECASE) else 0
        features['has_suspended_account'] = 1 if re.search(r'suspend(ed)?\s+account', text, re.IGNORECASE) else 0
        features['has_click_here'] = 1 if re.search(r'click\s+here', text, re.IGNORECASE) else 0
        features['has_limited_time'] = 1 if re.search(r'limited\s+time', text, re.IGNORECASE) else 0
        features['has_act_now'] = 1 if re.search(r'act\s+now', text, re.IGNORECASE) else 0
        
        # Check for money/prize mentions
        features['mentions_money'] = 1 if re.search(r'\$\d+|\d+\s*dollar|prize|reward|cash', text, re.IGNORECASE) else 0
        
        # Check for threat/urgency language
        features['has_threat_language'] = 1 if re.search(
            r'suspend|terminate|expire|deadline|urgent|immediate', text, re.IGNORECASE
        ) else 0
        
        return features
    
    def extract_url_features(self, urls: List[str]) -> Dict[str, float]:
        """Extract URL-based features"""
        features = {}
        
        if not urls:
            features['url_count'] = 0
            features['has_url_shortener'] = 0
            features['avg_url_length'] = 0
            features['has_ip_address'] = 0
            features['has_https'] = 0
            features['suspicious_tld'] = 0
            return features
        
        features['url_count'] = len(urls)
        
        url_lengths = []
        has_shortener = 0
        has_ip = 0
        has_https = 0
        suspicious_tld_count = 0
        
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download', '.review']
        
        for url in urls:
            url_lengths.append(len(url))
            
            # Parse URL
            try:
                parsed = urlparse(url)
                
                # Check for URL shorteners
                if any(shortener in parsed.netloc for shortener in self.url_shorteners):
                    has_shortener = 1
                
                # Check for IP addresses
                if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc):
                    has_ip = 1
                
                # Check for HTTPS
                if parsed.scheme == 'https':
                    has_https += 1
                
                # Check for suspicious TLDs
                if any(tld in parsed.netloc for tld in suspicious_tlds):
                    suspicious_tld_count += 1
                
                # Subdomain count
                subdomain_count = parsed.netloc.count('.')
                features[f'url_{urls.index(url)}_subdomain_count'] = subdomain_count
                
            except:
                continue
        
        features['avg_url_length'] = np.mean(url_lengths) if url_lengths else 0
        features['max_url_length'] = max(url_lengths) if url_lengths else 0
        features['has_url_shortener'] = has_shortener
        features['has_ip_address'] = has_ip
        features['https_ratio'] = has_https / len(urls) if urls else 0
        features['suspicious_tld_ratio'] = suspicious_tld_count / len(urls) if urls else 0
        
        return features
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls
    
    def _create_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Create numerical feature vector from feature dictionary"""
        # Define feature order for consistency
        feature_names = [
            'text_length', 'word_count', 'sentence_count',
            'avg_word_length', 'avg_sentence_length',
            'flesch_reading_ease', 'flesch_kincaid_grade',
            'punctuation_density', 'capital_ratio', 'digit_ratio',
            'urgency_score', 'financial_score', 'credential_score',
            'action_score', 'overall_keyword_score',
            'suspicious_pattern_count', 'suspicious_pattern_density',
            'has_verify_account', 'has_suspended_account',
            'has_click_here', 'has_limited_time', 'has_act_now',
            'mentions_money', 'has_threat_language',
            'url_count', 'avg_url_length', 'has_url_shortener',
            'has_ip_address', 'https_ratio', 'suspicious_tld_ratio'
        ]
        
        # Create vector
        vector = []
        for name in feature_names:
            if name in features:
                vector.append(features[name])
            else:
                vector.append(0.0)
        
        return np.array(vector)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order"""
        return [
            'text_length', 'word_count', 'sentence_count',
            'avg_word_length', 'avg_sentence_length',
            'flesch_reading_ease', 'flesch_kincaid_grade',
            'punctuation_density', 'capital_ratio', 'digit_ratio',
            'urgency_score', 'financial_score', 'credential_score',
            'action_score', 'overall_keyword_score',
            'suspicious_pattern_count', 'suspicious_pattern_density',
            'has_verify_account', 'has_suspended_account',
            'has_click_here', 'has_limited_time', 'has_act_now',
            'mentions_money', 'has_threat_language',
            'url_count', 'avg_url_length', 'has_url_shortener',
            'has_ip_address', 'https_ratio', 'suspicious_tld_ratio'
        ]
