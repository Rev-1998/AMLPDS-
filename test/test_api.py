# tests/test_api.py
"""
Tests for API endpoints
"""

import pytest
import json
from unittest.mock import patch, Mock
from flask_jwt_extended import create_access_token

from src.web.app import create_app


class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        app = create_app('testing')
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def auth_headers(self, app):
        """Create authentication headers"""
        with app.app_context():
            token = create_access_token(identity='testuser')
            return {'Authorization': f'Bearer {token}'}
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        data = json.loads(response.data)
        
        assert response.status_code in [200, 503]
        assert 'status' in data
        assert 'timestamp' in data
    
    @patch('src.web.api.classifier')
    @patch('src.web.api.preprocessor')
    @patch('src.web.api.ai_assistant')
    def test_predict_success(self, mock_assistant, mock_preprocessor, 
                           mock_classifier, client, auth_headers):
        """Test successful prediction"""
        # Mock responses
        mock_preprocessor.preprocess_email.return_value = {
            'cleaned_text': 'cleaned email text',
            'features': {}
        }
        
        mock_classifier.predict.return_value = {
            'prediction': 'phishing',
            'confidence': 0.95
        }
        
        mock_assistant.generate_explanation.return_value = {
            'risk_level': 'HIGH',
            'summary': 'This is phishing',
            'suspicious_keywords': [],
            'recommendations': ['Do not click links'],
            'pattern_analysis': {}
        }
        
        # Make request
        response = client.post('/api/predict',
                             headers=auth_headers,
                             json={'email_text': 'Test email'})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['prediction'] == 'phishing'
        assert data['confidence'] == 0.95
    
    def test_predict_no_auth(self, client):
        """Test prediction without authentication"""
        response = client.post('/api/predict',
                             json={'email_text': 'Test email'})
        
        assert response.status_code == 401
    
    def test_predict_no_data(self, client, auth_headers):
        """Test prediction with no data"""
        response = client.post('/api/predict',
                             headers=auth_headers,
                             json={})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
