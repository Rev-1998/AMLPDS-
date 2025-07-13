# src/utils/logger.py
"""
Logging configuration for AMLPDS
"""

import logging
import logging.config
import os
from datetime import datetime


def setup_logging(log_level=None, log_file=None):
    """Setup logging configuration"""
    
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    if log_file is None:
        log_file = os.getenv('LOG_FILE', 'logs/amlpds.log')
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'detailed',
                'filename': log_file,
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            'src': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'werkzeug': {
                'level': 'WARNING',
                'handlers': ['console', 'file']
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console', 'file']
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name):
    """Get a logger instance"""
    return logging.getLogger(name)


# src/utils/__init__.py
"""Utility modules for AMLPDS"""

from .config import Config, config
from .logger import setup_logging, get_logger

__all__ = ['Config', 'config', 'setup_logging', 'get_logger']
