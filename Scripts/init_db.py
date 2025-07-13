Scripts/init_db.py
#!/usr/bin/env python3
"""
Initialize database with tables and sample data
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default='user')
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)


class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    email_text = Column(Text, nullable=False)
    prediction = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)


class Report(Base):
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    email_text = Column(Text, nullable=False)
    reported_as = Column(String(20), nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_database(database_url):
    """Initialize database with tables"""
    engine = create_engine(database_url)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Add sample admin user
    from werkzeug.security import generate_password_hash
    
    admin = User(
        username='admin',
        email='admin@amlpds.com',
        password_hash=generate_password_hash('changeme'),
        role='admin'
    )
    
    session.add(admin)
    session.commit()
    
    print("Database initialized successfully!")


if __name__ == '__main__':
    database_url = os.getenv('DATABASE_URL', 'sqlite:///amlpds.db')
    init_database(database_url)
