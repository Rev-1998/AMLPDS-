# src/web/auth.py
"""
Authentication routes and handlers
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)

# In-memory user store (replace with database in production)
users = {
    'admin': {
        'password': generate_password_hash('changeme'),
        'email': 'admin@amlpds.com',
        'role': 'admin',
        'created_at': datetime.utcnow()
    }
}


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler"""
    if request.method == 'GET':
        return render_template('login.html')
    
    # Handle POST request
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Username and password required'}), 400
    
    # Check user exists
    user = users.get(username)
    if not user or not check_password_hash(user['password'], password):
        logger.warning(f"Failed login attempt for username: {username}")
        return jsonify({'message': 'Invalid credentials'}), 401
    
    # Create tokens
    access_token = create_access_token(
        identity=username,
        additional_claims={'role': user['role']}
    )
    refresh_token = create_refresh_token(identity=username)
    
    logger.info(f"Successful login for user: {username}")
    
    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': {
            'username': username,
            'email': user['email'],
            'role': user['role']
        }
    }), 200


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and handler"""
    if request.method == 'GET':
        return render_template('register.html')
    
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    
    # Validate input
    if not all([username, password, email]):
        return jsonify({'message': 'All fields required'}), 400
    
    # Check username length
    if len(username) < 3 or len(username) > 20:
        return jsonify({'message': 'Username must be 3-20 characters'}), 400
    
    # Check password strength
    if len(password) < 8:
        return jsonify({'message': 'Password must be at least 8 characters'}), 400
    
    if not re.search(r'[A-Z]', password):
        return jsonify({'message': 'Password must contain uppercase letter'}), 400
    
    if not re.search(r'[a-z]', password):
        return jsonify({'message': 'Password must contain lowercase letter'}), 400
    
    if not re.search(r'[0-9]', password):
        return jsonify({'message': 'Password must contain number'}), 400
    
    # Validate email
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        return jsonify({'message': 'Invalid email format'}), 400
    
    # Check if user exists
    if username in users:
        return jsonify({'message': 'Username already exists'}), 409
    
    # Create user
    users[username] = {
        'password': generate_password_hash(password),
        'email': email,
        'role': 'user',
        'created_at': datetime.utcnow()
    }
    
    logger.info(f"New user registered: {username}")
    
    # Create tokens
    access_token = create_access_token(identity=username)
    refresh_token = create_refresh_token(identity=username)
    
    return jsonify({
        'message': 'Registration successful',
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': {
            'username': username,
            'email': email,
            'role': 'user'
        }
    }), 201


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    identity = get_jwt_identity()
    access_token = create_access_token(identity=identity)
    
    return jsonify({'access_token': access_token}), 200


@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    """Get user profile"""
    username = get_jwt_identity()
    user = users.get(username)
    
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    return jsonify({
        'username': username,
        'email': user['email'],
        'role': user['role'],
        'created_at': user['created_at'].isoformat()
    }), 200


@auth_bp.route('/change-password', methods=['POST'])
@jwt_required()
def change_password():
    """Change user password"""
    username = get_jwt_identity()
    data = request.get_json()
    
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not current_password or not new_password:
        return jsonify({'message': 'Current and new password required'}), 400
    
    user = users.get(username)
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    # Verify current password
    if not check_password_hash(user['password'], current_password):
        return jsonify({'message': 'Invalid current password'}), 401
    
    # Validate new password
    if len(new_password) < 8:
        return jsonify({'message': 'Password must be at least 8 characters'}), 400
    
    # Update password
    users[username]['password'] = generate_password_hash(new_password)
    
    logger.info(f"Password changed for user: {username}")
    
    return jsonify({'message': 'Password changed successfully'}), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout handler"""
    # In a production system, you would blacklist the token here
    username = get_jwt_identity()
    logger.info(f"User logged out: {username}")
    
    return jsonify({'message': 'Logged out successfully'}), 200
