# src/cli.py
"""
Command-line interface for AMLPDS
"""

import click
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bert_classifier import BERTPhishingClassifier
from src.preprocessing.email_preprocessor import EmailPreprocessor
from src.explainability.ai_assistant import AIAssistant
from src.utils.logger import setup_logging
