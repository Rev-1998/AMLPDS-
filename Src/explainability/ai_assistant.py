# src/explainability/ai_assistant.py
"""
AI Assistant module for explainable phishing detection results
"""

import shap
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from collections import Counter
import logging
import re

logger = logging.getLogger(__name__)


class AIAssistant:
    """Provides explainable AI capabilities for phishing detection"""
    
    def __init__(self, model, tokenizer):
        """
        Initialize AI Assistant
        
        Args:
            model: BERT model instance
            tokenizer: BERT tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Initialize SHAP explainer with a wrapper function
        self.explainer = None
        self._init_shap_explainer()
        
        # Phishing indicators for keyword analysis
        self.phishing_indicators = [
            'urgent', 'verify', 'suspend', 'expire', 'confirm',
            'click here', 'act now', 'limited time', 'congratulations',
            'winner', 'prize', 'free', 'guarantee', 'risk-free',
            'account', 'password', 'security', 'alert', 'update',
            'payment', 'refund', 'invoice', 'tax', 'bank'
        ]
        
        logger.info("Initialized AI Assistant for explainability")
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer with model wrapper"""
        def model_predict(texts):
            """Wrapper function for SHAP"""
            predictions = []
            
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    predictions.append(probs.cpu().numpy()[0])
            
            return np.array(predictions)
        
        # Create explainer with a small background dataset
        self.explainer = shap.Explainer(
            model_predict,
            masker=shap.maskers.Text(tokenizer=self.tokenizer),
            algorithm="auto"
        )
    
    def generate_explanation(self, text: str, prediction: str, 
                           confidence: float) -> Dict:
        """
        Generate comprehensive explanation for prediction
        
        Args:
            text: Email text
            prediction: Model prediction (phishing/legitimate)
            confidence: Prediction confidence
            
        Returns:
            Dictionary containing explanation details
        """
        # Generate SHAP explanation
        shap_explanation = self.generate_shap_explanation(text)
        
        # Identify suspicious keywords
        suspicious_keywords = self.identify_suspicious_keywords(text)
        
        # Extract top influential words
        top_words = self._extract_top_words(shap_explanation, n=10)
        
        # Generate summary
        summary = self._generate_summary(prediction, confidence, 
                                       suspicious_keywords, top_words)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(prediction, confidence,
                                                       suspicious_keywords)
        
        # Analyze patterns
        pattern_analysis = self._analyze_patterns(text)
        
        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'summary': summary,
            'top_influential_words': top_words,
            'suspicious_keywords': suspicious_keywords,
            'pattern_analysis': pattern_analysis,
            'recommendations': recommendations,
            'risk_level': self._assess_risk_level(prediction, confidence),
            'shap_values': shap_explanation['shap_values'] if shap_explanation else None
        }
        
        return explanation
    
    def generate_shap_explanation(self, text: str) -> Dict:
        """Generate SHAP explanation for the prediction"""
        try:
            # Get SHAP values
            shap_values = self.explainer([text])
            
            # Get word-level explanations
            words = text.split()
            
            # Create explanation dictionary
            explanations = []
            
            if hasattr(shap_values, 'values') and shap_values.values is not None:
                # Get values for phishing class (index 1)
                values = shap_values.values[0][:, 1] if len(shap_values.values[0].shape) > 1 else shap_values.values[0]
                
                # Map SHAP values to words
                tokens = self.tokenizer.tokenize(text)
                
                # Group subword tokens back to words
                word_importance = self._aggregate_token_importance(tokens, values, words)
                
                for word, importance in word_importance.items():
                    explanations.append({
                        'word': word,
                        'importance': float(importance),
                        'contribution': 'positive' if importance > 0 else 'negative'
                    })
            
            return {
                'explanations': explanations,
                'shap_values': values.tolist() if 'values' in locals() else None
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(text)
    
    def identify_suspicious_keywords(self, text: str) -> List[Dict]:
        """Identify suspicious keywords in the text"""
        text_lower = text.lower()
        found_indicators = []
        
        for indicator in self.phishing_indicators:
            if indicator in text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(indicator, start)
                    if pos == -1:
                        break
                    
                    # Get context
                    context_start = max(0, pos - 20)
                    context_end = min(len(text), pos + len(indicator) + 20)
                    context = text[context_start:context_end]
                    
                    found_indicators.append({
                        'keyword': indicator,
                        'position': pos,
                        'context': context,
                        'severity': self._get_keyword_severity(indicator)
                    })
                    
                    start = pos + 1
        
        # Sort by severity
        found_indicators.sort(key=lambda x: x['severity'], reverse=True)
        
        return found_indicators
    
    def _extract_top_words(self, shap_explanation: Dict, n: int = 10) -> List[Dict]:
        """Extract top influential words from SHAP explanation"""
        if not shap_explanation or 'explanations' not in shap_explanation:
            return []
        
        explanations = shap_explanation['explanations']
        
        # Sort by absolute importance
        sorted_explanations = sorted(
            explanations,
            key=lambda x: abs(x['importance']),
            reverse=True
        )[:n]
        
        return sorted_explanations
    
    def _generate_summary(self, prediction: str, confidence: float,
                         suspicious_keywords: List[Dict],
                         top_words: List[Dict]) -> str:
        """Generate human-readable summary"""
        if prediction == 'phishing':
            summary = f"This email is classified as PHISHING with {confidence:.1%} confidence. "
            
            if suspicious_keywords:
                summary += f"Found {len(suspicious_keywords)} suspicious indicators including: "
                unique_keywords = list(set([kw['keyword'] for kw in suspicious_keywords[:5]]))
                summary += ", ".join(f'"{kw}"' for kw in unique_keywords[:3])
                
                if len(unique_keywords) > 3:
                    summary += f" and {len(unique_keywords) - 3} more"
                summary += ". "
            
            if top_words:
                contributing_words = [w for w in top_words if w['contribution'] == 'positive'][:3]
                if contributing_words:
                    summary += "Key factors: " + ", ".join(f'"{w["word"]}"' for w in contributing_words) + "."
                    
        else:
            summary = f"This email appears LEGITIMATE with {confidence:.1%} confidence. "
            
            if suspicious_keywords:
                summary += f"Although {len(suspicious_keywords)} potential indicators were found, "
                summary += "the overall content and context suggest it's legitimate. "
            
            if top_words:
                contributing_words = [w for w in top_words if w['contribution'] == 'negative'][:3]
                if contributing_words:
                    summary += "Legitimate factors: " + ", ".join(f'"{w["word"]}"' for w in contributing_words) + "."
        
        return summary
    
    def _generate_recommendations(self, prediction: str, confidence: float,
                                suspicious_keywords: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if prediction == 'phishing':
            recommendations.extend([
                "Do not click any links in this email",
                "Do not provide personal information",
                "Do not download any attachments",
                "Verify sender identity through official channels",
                "Report this email to your IT security team",
                "Delete this email from your inbox"
            ])
            
            # Add specific recommendations based on keywords
            keyword_types = [kw['keyword'] for kw in suspicious_keywords]
            
            if any('password' in kw or 'account' in kw for kw in keyword_types):
                recommendations.append("Never share passwords via email")
            
            if any('payment' in kw or 'bank' in kw for kw in keyword_types):
                recommendations.append("Contact your bank directly to verify any financial requests")
                
        else:
            if confidence < 0.8:
                recommendations.append("Exercise caution - confidence is relatively low")
                
            recommendations.extend([
                "Verify sender address matches official domain",
                "Check links before clicking (hover to see destination)",
                "Be cautious with personal information requests",
                "If unsure, contact sender through known channels"
            ])
        
        return recommendations
    
    def _analyze_patterns(self, text: str) -> Dict:
        """Analyze email patterns"""
        patterns = {
            'urgency_language': bool(re.search(r'urgent|immediate|expire|act now', text, re.I)),
            'action_required': bool(re.search(r'click|verify|confirm|update', text, re.I)),
            'threat_language': bool(re.search(r'suspend|terminate|close|disable', text, re.I)),
            'financial_language': bool(re.search(r'payment|refund|prize|lottery|bank', text, re.I)),
            'credential_request': bool(re.search(r'password|username|pin|ssn', text, re.I)),
            'generic_greeting': bool(re.search(r'dear (customer|user|member|sir/madam)', text, re.I)),
            'spelling_errors': self._detect_spelling_errors(text),
            'excessive_punctuation': bool(re.search(r'[!]{2,}|[?]{2,}', text))
        }
        
        return patterns
    
    def _assess_risk_level(self, prediction: str, confidence: float) -> str:
        """Assess overall risk level"""
        if prediction == 'phishing':
            if confidence > 0.9:
                return 'CRITICAL'
            elif confidence > 0.7:
                return 'HIGH'
            else:
                return 'MEDIUM'
        else:
            if confidence > 0.9:
                return 'LOW'
            elif confidence > 0.7:
                return 'MEDIUM'
            else:
                return 'UNCERTAIN'
    
    def _get_keyword_severity(self, keyword: str) -> int:
        """Get severity score for keyword (1-5)"""
        high_severity = ['password', 'suspend', 'terminate', 'expire', 'urgent']
        medium_severity = ['verify', 'confirm', 'update', 'payment', 'account']
        
        if keyword in high_severity:
            return 5
        elif keyword in medium_severity:
            return 3
        else:
            return 1
    
    def _aggregate_token_importance(self, tokens: List[str], values: np.ndarray,
                                   words: List[str]) -> Dict[str, float]:
        """Aggregate subword token importance to word level"""
        word_importance = {}
        current_word = ""
        current_importance = 0
        
        for token, value in zip(tokens, values):
            if token.startswith("##"):
                # Continuation of current word
                current_importance += value
            else:
                # New word
                if current_word:
                    word_importance[current_word] = current_importance
                current_word = token
                current_importance = value
        
        # Don't forget the last word
        if current_word:
            word_importance[current_word] = current_importance
        
        return word_importance
    
    def _fallback_explanation(self, text: str) -> Dict:
        """Fallback explanation when SHAP fails"""
        # Simple keyword-based importance
        words = text.split()
        explanations = []
        
        for word in words:
            importance = 0
            if word.lower() in self.phishing_indicators:
                importance = 0.5
            elif word.lower() in ['the', 'a', 'an', 'is', 'are']:
                importance = -0.1
            
            if importance != 0:
                explanations.append({
                    'word': word,
                    'importance': importance,
                    'contribution': 'positive' if importance > 0 else 'negative'
                })
        
        return {'explanations': explanations, 'shap_values': None}
    
    def _detect_spelling_errors(self, text: str) -> bool:
        """Simple heuristic for detecting potential spelling errors"""
        # This is a simplified check - in production, use a proper spell checker
        common_misspellings = [
            'recieve', 'occured', 'seperate', 'definately', 'occassion',
            'aquire', 'wierd', 'concious', 'occuring', 'dissapear'
        ]
        
        text_lower = text.lower()
        return any(misspelling in text_lower for misspelling in common_misspellings)
