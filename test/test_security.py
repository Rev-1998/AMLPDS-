# tests/test_security.py
"""
Tests for security features
"""

import pytest
from werkzeug.security import generate_password_hash, check_password_hash

from src.web.auth import users


class TestSecurity:
    """Test security features"""
    
    def test_password_hashing(self):
        """Test password hashing"""
        password = "SecurePass123!"
        hashed = generate_password_hash(password)
        
        assert password != hashed
        assert check_password_hash(hashed, password)
        assert not check_password_hash(hashed, "WrongPassword")
    
    def test_password_requirements(self):
        """Test password validation"""
        # These would be actual validation functions in production
        def validate_password(password):
            if len(password) < 8:
                return False
            if not any(c.isupper() for c in password):
                return False
            if not any(c.islower() for c in password):
                return False
            if not any(c.isdigit() for c in password):
                return False
            return True
        
        assert validate_password("SecurePass123") == True
        assert validate_password("weak") == False
        assert validate_password("ALLCAPS123") == False
        assert validate_password("nocaps123") == False
        assert validate_password("NoNumbers") == False
