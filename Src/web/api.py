# src/web/api.py
"""
API endpoints for phishing detection
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime
import traceback

# Import models and utilities
from ..models.bert_classifier import BERTPhishingClassifier
from ..models.adversarial_trainer import AdversarialTrainer
from ..preprocessing.email_preprocessor import EmailPreprocessor
from ..explainability.ai_assistant import AIAssistant

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Initialize components (in production, use dependency injection)
classifier = None
preprocessor = None
ai_assistant = None


def init_models():
    """Initialize ML models"""
    global classifier, preprocessor, ai_assistant
    
    try:
        # Initialize classifier
        model_path = current_app.config.get('MODEL_PATH')
        classifier = BERTPhishingClassifier()
        
        # Load pre-trained model if exists
        if os.path.exists(model_path):
            classifier.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"No pre-trained model found at {model_path}")
        
        # Initialize preprocessor
        preprocessor = EmailPreprocessor()
        
        # Initialize AI assistant
        ai_assistant = AIAssistant(classifier.model, classifier.tokenizer)
        
        logger.info("Models initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


# Initialize models on first request
@api_bp.before_app_first_request
def before_first_request():
    init_models()


@api_bp.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    """Predict if email is phishing"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email_text = data.get('email_text', '')
        urls = data.get('urls', [])
        
        if not email_text:
            return jsonify({'error': 'Email text required'}), 400
        
        # Preprocess email
        preprocessed = preprocessor.preprocess_email(email_text, urls)
        
        # Make prediction
        result = classifier.predict(preprocessed['cleaned_text'])
        
        # Generate explanation
        explanation = ai_assistant.generate_explanation(
            preprocessed['cleaned_text'],
            result['prediction'],
            result['confidence']
        )
        
        # Log prediction
        username = get_jwt_identity()
        logger.info(f"Prediction made by {username}: {result['prediction']} "
                   f"(confidence: {result['confidence']:.2%})")
        
        response = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'risk_level': explanation['risk_level'],
            'explanation': {
                'summary': explanation['summary'],
                'suspicious_keywords': explanation['suspicious_keywords'],
                'recommendations': explanation['recommendations'],
                'pattern_analysis': explanation['pattern_analysis']
            },
            'features': preprocessed['features'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500


@api_bp.route('/predict/batch', methods=['POST'])
@jwt_required()
def predict_batch():
    """Batch prediction for multiple emails"""
    try:
        data = request.get_json()
        
        if not data or 'emails' not in data:
            return jsonify({'error': 'No emails provided'}), 400
        
        emails = data['emails']
        if not isinstance(emails, list):
            return jsonify({'error': 'Emails must be a list'}), 400
        
        results = []
        
        for email in emails:
            if isinstance(email, dict):
                email_text = email.get('text', '')
                email_id = email.get('id', None)
            else:
                email_text = str(email)
                email_id = None
            
            # Preprocess
            preprocessed = preprocessor.preprocess_email(email_text)
            
            # Predict
            result = classifier.predict(preprocessed['cleaned_text'])
            
            results.append({
                'id': email_id,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
        
        username = get_jwt_identity()
        logger.info(f"Batch prediction by {username}: {len(results)} emails processed")
        
        return jsonify({
            'results': results,
            'total': len(results),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'message': str(e)}), 500


@api_bp.route('/analyze/file', methods=['POST'])
@jwt_required()
def analyze_file():
    """Analyze uploaded email file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'.txt', '.eml', '.msg'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'File type {file_ext} not supported'}), 400
        
        # Read file content
        content = file.read().decode('utf-8', errors='ignore')
        
        # Extract URLs
        urls = preprocessor.extract_urls(content)
        
        # Preprocess
        preprocessed = preprocessor.preprocess_email(content, urls)
        
        # Predict
        result = classifier.predict(preprocessed['cleaned_text'])
        
        # Generate explanation
        explanation = ai_assistant.generate_explanation(
            preprocessed['cleaned_text'],
            result['prediction'],
            result['confidence']
        )
        
        username = get_jwt_identity()
        logger.info(f"File analysis by {username}: {secure_filename(file.filename)}")
        
        return jsonify({
            'filename': secure_filename(file.filename),
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'risk_level': explanation['risk_level'],
            'explanation': explanation,
            'urls_found': len(urls),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"File analysis error: {str(e)}")
        return jsonify({'error': 'File analysis failed', 'message': str(e)}), 500


@api_bp.route('/report', methods=['POST'])
@jwt_required()
def report_phishing():
    """Report a phishing email"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email_text = data.get('email_text', '')
        reported_as = data.get('reported_as', 'phishing')
        notes = data.get('notes', '')
        
        if not email_text:
            return jsonify({'error': 'Email text required'}), 400
        
        # In production, save to database
        report = {
            'user': get_jwt_identity(),
            'email_text': email_text,
            'reported_as': reported_as,
            'notes': notes,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Phishing report submitted by {report['user']}")
        
        return jsonify({
            'message': 'Report submitted successfully',
            'report_id': 'REPORT-' + datetime.utcnow().strftime('%Y%m%d%H%M%S')
        }), 201
        
    except Exception as e:
        logger.error(f"Report submission error: {str(e)}")
        return jsonify({'error': 'Report submission failed', 'message': str(e)}), 500


@api_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    """Get system statistics"""
    try:
        # In production, fetch from database
        stats = {
            'total_scans': 1234,
            'phishing_detected': 456,
            'legitimate_emails': 778,
            'detection_rate': 0.37,
            'active_users': 42,
            'last_24h_scans': 234,
            'average_confidence': 0.892,
            'system_uptime': '99.9%'
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': 'Failed to get stats'}), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if models are loaded
        models_loaded = all([classifier is not None, 
                           preprocessor is not None, 
                           ai_assistant is not None])
        
        health = {
            'status': 'healthy' if models_loaded else 'unhealthy',
            'models_loaded': models_loaded,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(health), 200 if models_loaded else 503
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503


@api_bp.route('/model/info', methods=['GET'])
@jwt_required()
def model_info():
    """Get model information"""
    try:
        info = {
            'model_name': classifier.model_name if classifier else 'Not loaded',
            'num_labels': classifier.num_labels if classifier else None,
            'device': str(classifier.device) if classifier else 'Unknown',
            'preprocessing': {
                'urgency_keywords': len(preprocessor.urgency_keywords) if preprocessor else 0,
                'financial_keywords': len(preprocessor.financial_keywords) if preprocessor else 0,
                'patterns': len(preprocessor.suspicious_patterns) if preprocessor else 0
            }
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': 'Failed to get model info'}), 500
