# tests/test_preprocessing.py
"""
Tests for preprocessing module
"""

import pytest
from src.preprocessing.email_preprocessor import EmailPreprocessor


class TestEmailPreprocessor:
    """Test email preprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return EmailPreprocessor()
    
    def test_clean_text(self, preprocessor):
        """Test text cleaning"""
        text = """
        <html>
        <body>
        From: sender@example.com
        To: recipient@example.com
        Subject: Test Email
        
        Visit http://example.com for more info!
        Contact us at support@example.com
        </body>
        </html>
        """
        
        cleaned = preprocessor.clean_text(text)
        
        assert '<html>' not in cleaned
        assert 'From:' not in cleaned.lower()
        assert 'URL' in cleaned
        assert 'EMAIL' in cleaned
    
    def test_extract_text_features(self, preprocessor):
        """Test text feature extraction"""
        text = "This is a test email. It contains two sentences!"
        features = preprocessor.extract_text_features(text)
        
        assert 'text_length' in features
        assert 'word_count' in features
        assert 'sentence_count' in features
        assert features['sentence_count'] == 2
        assert features['word_count'] > 0
    
    def test_extract_keyword_features(self, preprocessor):
        """Test keyword feature extraction"""
        text = "URGENT: Please verify your account immediately to avoid suspension"
        features = preprocessor.extract_keyword_features(text)
        
        assert 'urgency_score' in features
        assert 'credential_score' in features
        assert features['urgency_score'] > 0
        assert features['urgency_count'] >= 2  # 'urgent' and 'verify'
    
    def test_extract_pattern_features(self, preprocessor):
        """Test pattern feature extraction"""
        text = "Click here to verify your account and claim your prize!"
        features = preprocessor.extract_pattern_features(text)
        
        assert 'suspicious_pattern_count' in features
        assert 'has_click_here' in features
        assert 'has_verify_account' in features
        assert features['has_click_here'] == 1
        assert features['has_verify_account'] == 1
    
    def test_extract_url_features(self, preprocessor):
        """Test URL feature extraction"""
        urls = [
            "https://example.com",
            "http://192.168.1.1/phishing",
            "https://bit.ly/shortlink"
        ]
        
        features = preprocessor.extract_url_features(urls)
        
        assert features['url_count'] == 3
        assert features['has_ip_address'] == 1
        assert features['has_url_shortener'] == 1
        assert features['https_ratio'] == 2/3
    
    def test_extract_urls(self, preprocessor):
        """Test URL extraction from text"""
        text = """
        Visit our website at https://example.com
        Or click here: http://phishing-site.com/verify
        """
        
        urls = preprocessor.extract_urls(text)
        
        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "http://phishing-site.com/verify" in urls
    
    def test_preprocess_email(self, preprocessor):
        """Test complete email preprocessing"""
        email_text = """
        Dear Customer,
        
        URGENT: Your account will be suspended!
        Click here to verify: http://suspicious-link.com
        
        Best regards,
        Support Team
        """
        
        result = preprocessor.preprocess_email(email_text)
        
        assert 'cleaned_text' in result
        assert 'features' in result
        assert 'feature_vector' in result
        assert isinstance(result['feature_vector'], np.ndarray)
