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

# Setup logging
setup_logging()


@click.group()
def cli():
    """AMLPDS Command Line Interface"""
    pass


@cli.command()
@click.option('--file', '-f', required=True, help='Path to email file')
@click.option('--model-path', default='data/models/bert_phishing_classifier.pt', 
              help='Path to trained model')
@click.option('--output', '-o', help='Output file for results')
def classify(file, model_path, output):
    """Classify a single email"""
    click.echo("Loading model...")
    
    # Initialize components
    classifier = BERTPhishingClassifier()
    if os.path.exists(model_path):
        classifier.load_model(model_path)
    
    preprocessor = EmailPreprocessor()
    ai_assistant = AIAssistant(classifier.model, classifier.tokenizer)
    
    # Read email
    try:
        with open(file, 'r', encoding='utf-8') as f:
            email_text = f.read()
    except Exception as e:
        click.echo(f"Error reading file: {e}", err=True)
        return
    
    # Preprocess
    click.echo("Preprocessing email...")
    preprocessed = preprocessor.preprocess_email(email_text)
    
    # Predict
    click.echo("Analyzing...")
    result = classifier.predict(preprocessed['cleaned_text'])
    
    # Generate explanation
    explanation = ai_assistant.generate_explanation(
        preprocessed['cleaned_text'],
        result['prediction'],
        result['confidence']
    )
    
    # Display results
    click.echo("\n" + "="*50)
    click.echo(f"Prediction: {result['prediction'].upper()}")
    click.echo(f"Confidence: {result['confidence']:.2%}")
    click.echo(f"Risk Level: {explanation['risk_level']}")
    click.echo("="*50 + "\n")
    
    click.echo("Summary:")
    click.echo(explanation['summary'])
    
    if explanation['suspicious_keywords']:
        click.echo("\nSuspicious Keywords:")
        for kw in explanation['suspicious_keywords'][:5]:
            click.echo(f"  - {kw['keyword']}")
    
    click.echo("\nRecommendations:")
    for rec in explanation['recommendations'][:3]:
        click.echo(f"  • {rec}")
    
    # Save results if output specified
    if output:
        results_data = {
            'file': file,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'risk_level': explanation['risk_level'],
            'explanation': explanation
        }
        
        with open(output, 'w') as f:
            json.dump(results_data, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.option('--input', '-i', required=True, help='CSV file with emails')
@click.option('--output', '-o', required=True, help='Output CSV file')
@click.option('--model-path', default='data/models/bert_phishing_classifier.pt',
              help='Path to trained model')
@click.option('--batch-size', default=16, help='Batch size for processing')
def batch(input, output, model_path, batch_size):
    """Batch classification of emails"""
    import pandas as pd
    
    click.echo(f"Loading emails from {input}...")
    
    try:
        df = pd.read_csv(input)
        if 'email_text' not in df.columns and 'text' not in df.columns:
            click.echo("Error: CSV must have 'email_text' or 'text' column", err=True)
            return
        
        text_column = 'email_text' if 'email_text' in df.columns else 'text'
        emails = df[text_column].tolist()
    except Exception as e:
        click.echo(f"Error reading CSV: {e}", err=True)
        return
    
    click.echo(f"Found {len(emails)} emails")
    
    # Initialize components
    click.echo("Loading model...")
    classifier = BERTPhishingClassifier()
    if os.path.exists(model_path):
        classifier.load_model(model_path)
    
    preprocessor = EmailPreprocessor()
    
    # Process emails
    results = []
    with click.progressbar(emails, label='Processing emails') as bar:
        for email in bar:
            # Preprocess
            preprocessed = preprocessor.preprocess_email(email)
            
            # Predict
            result = classifier.predict(preprocessed['cleaned_text'])
            
            results.append({
                'email_text': email[:100] + '...' if len(email) > 100 else email,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output, index=False)
    
    # Display summary
    phishing_count = sum(1 for r in results if r['prediction'] == 'phishing')
    click.echo(f"\nResults saved to {output}")
    click.echo(f"Total emails: {len(results)}")
    click.echo(f"Phishing detected: {phishing_count} ({phishing_count/len(results)*100:.1f}%)")
    click.echo(f"Legitimate: {len(results) - phishing_count} ({(len(results) - phishing_count)/len(results)*100:.1f}%)")


@cli.command()
@click.option('--input', '-i', required=True, help='Email text file')
@click.option('--attack', default='textfooler', help='Attack type')
@click.option('--model-path', default='data/models/bert_phishing_classifier.pt',
              help='Path to trained model')
def adversarial(input, attack, model_path):
    """Generate adversarial example for an email"""
    from src.models.adversarial_trainer import AdversarialTrainer
    
    click.echo("Loading model...")
    classifier = BERTPhishingClassifier()
    if os.path.exists(model_path):
        classifier.load_model(model_path)
    
    # Read email
    try:
        with open(input, 'r', encoding='utf-8') as f:
            email_text = f.read()
    except Exception as e:
        click.echo(f"Error reading file: {e}", err=True)
        return
    
    # Get original prediction
    original_result = classifier.predict(email_text)
    original_label = 1 if original_result['prediction'] == 'phishing' else 0
    
    click.echo(f"Original prediction: {original_result['prediction']} "
               f"({original_result['confidence']:.2%})")
    
    # Generate adversarial example
    click.echo(f"Generating adversarial example using {attack}...")
    trainer = AdversarialTrainer(classifier.model, classifier.tokenizer)
    
    adv_examples = trainer.generate_adversarial_examples(
        [email_text], [original_label], num_examples=1
    )
    
    if adv_examples and adv_examples[0]['success']:
        adv_example = adv_examples[0]
        click.echo("\nAdversarial example generated successfully!")
        click.echo(f"Perturbation: {adv_example['perturbation_percentage']:.1f}% of words changed")
        
        # Show changes
        click.echo("\nModified text:")
        click.echo("-" * 50)
        click.echo(adv_example['adversarial_text'])
        click.echo("-" * 50)
        
        # New prediction
        new_result = classifier.predict(adv_example['adversarial_text'])
        click.echo(f"\nNew prediction: {new_result['prediction']} "
                   f"({new_result['confidence']:.2%})")
    else:
        click.echo("Failed to generate adversarial example")


@cli.command()
@click.option('--model-path', default='data/models/bert_phishing_classifier.pt',
              help='Path to model')
def info(model_path):
    """Display model information"""
    if not os.path.exists(model_path):
        click.echo(f"Model not found at {model_path}", err=True)
        return
    
    # Load model info
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    
    click.echo("Model Information:")
    click.echo("-" * 30)
    click.echo(f"Model type: {checkpoint.get('tokenizer_name', 'Unknown')}")
    click.echo(f"Number of labels: {checkpoint.get('num_labels', 'Unknown')}")
    click.echo(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    
    # Count parameters
    if 'model_state_dict' in checkpoint:
        total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values() 
                          if p.dtype == torch.float32)
        click.echo(f"Total parameters: {total_params:,}")


def main():
    """Main entry point"""
    cli()


if __name__ == '__main__':
    main()


# scripts/generate_ssl_cert.py
#!/usr/bin/env python3
"""
Generate self-signed SSL certificate for development
"""

import os
import subprocess
import sys


def generate_ssl_certificate(cert_path='ssl/cert.pem', key_path='ssl/key.pem'):
    """Generate self-signed SSL certificate"""
    
    # Create SSL directory
    ssl_dir = os.path.dirname(cert_path)
    os.makedirs(ssl_dir, exist_ok=True)
    
    # Generate certificate
    cmd = [
        'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
        '-keyout', key_path,
        '-out', cert_path,
        '-days', '365',
        '-nodes',
        '-subj', '/C=US/ST=State/L=City/O=AMLPDS/CN=localhost'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"SSL certificate generated successfully!")
        print(f"Certificate: {cert_path}")
        print(f"Private key: {key_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating certificate: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("OpenSSL not found. Please install OpenSSL.")
        sys.exit(1)


if __name__ == '__main__':
    generate_ssl_certificate()


# run_dev.sh
#!/bin/bash
#
# Development startup script for AMLPDS
#

echo "Starting AMLPDS in development mode..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{raw,processed,models} logs static/{css,js,img} templates

# Set environment variables
export FLASK_APP=src.web.app
export FLASK_ENV=development
export PYTHONPATH=$PWD

# Generate SSL certificate if not exists
if [ ! -f "ssl/cert.pem" ]; then
    echo "Generating SSL certificate..."
    python scripts/generate_ssl_cert.py
fi

# Run the application
echo "Starting Flask development server..."
flask run --host=0.0.0.0 --port=5000 --cert=ssl/cert.pem --key=ssl/key.pem


# run_prod.sh
#!/bin/bash
#
# Production startup script for AMLPDS
#

echo "Starting AMLPDS in production mode..."

# Set environment variables
export FLASK_APP=src.web.app
export FLASK_ENV=production
export PYTHONPATH=$PWD

# Run with Gunicorn
echo "Starting Gunicorn server..."
gunicorn -w 4 \
    -b 0.0.0.0:5000 \
    --timeout 120 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --certfile ssl/cert.pem \
    --keyfile ssl/key.pem \
    src.web.app:app


# Makefile
# Makefile for AMLPDS

.PHONY: help install test run clean docker-build docker-run lint format

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make test         Run tests"
	@echo "  make run          Run development server"
	@echo "  make clean        Clean up temporary files"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run with Docker Compose"
	@echo "  make lint         Run code linting"
	@echo "  make format       Format code with black"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

test:
	pytest tests/ -v --cov=src --cov-report=html

run:
	./run_dev.sh

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

docker-build:
	docker-compose build

docker-run:
	docker-compose up

lint:
	flake8 src/ --max-line-length=100
	pylint src/

format:
	black src/ tests/ scripts/

freeze:
	pip freeze > requirements.txt
