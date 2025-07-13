 scripts/train_model.py
#!/usr/bin/env python3
"""
Script to train BERT model with adversarial training for phishing detection
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bert_classifier import BERTPhishingClassifier, PhishingDataset
from src.models.adversarial_trainer import AdversarialTrainer
from src.preprocessing.email_preprocessor import EmailPreprocessor
from src.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def load_data(data_path):
    """Load and prepare training data"""
    logger.info(f"Loading data from {data_path}")
    
    # Load phishing emails
    phishing_df = pd.read_csv(os.path.join(data_path, 'phishing_emails.csv'))
    phishing_texts = phishing_df['email_text'].tolist()
    phishing_labels = [1] * len(phishing_texts)
    
    # Load legitimate emails
    legitimate_df = pd.read_csv(os.path.join(data_path, 'legitimate_emails.csv'))
    legitimate_texts = legitimate_df['email_text'].tolist()
    legitimate_labels = [0] * len(legitimate_texts)
    
    # Combine data
    all_texts = phishing_texts + legitimate_texts
    all_labels = phishing_labels + legitimate_labels
    
    # Shuffle data
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    logger.info(f"Loaded {len(all_texts)} emails ({sum(all_labels)} phishing, {len(all_labels) - sum(all_labels)} legitimate)")
    
    return all_texts, all_labels


def preprocess_data(texts, preprocessor):
    """Preprocess email texts"""
    logger.info("Preprocessing emails...")
    processed_texts = []
    
    for text in tqdm(texts, desc="Preprocessing"):
        result = preprocessor.preprocess_email(text)
        processed_texts.append(result['cleaned_text'])
    
    return processed_texts


def create_data_loaders(texts, labels, tokenizer, train_ratio=0.8, batch_size=16):
    """Create train and validation data loaders"""
    # Create dataset
    dataset = PhishingDataset(texts, labels, tokenizer)
    
    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_standard(classifier, train_loader, val_loader, args):
    """Standard training without adversarial examples"""
    logger.info("Starting standard training...")
    
    classifier.train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_path
    )


def train_adversarial(classifier, train_texts, train_labels, val_texts, val_labels, args):
    """Adversarial training with TextFooler"""
    logger.info("Starting adversarial training...")
    
    # Initialize adversarial trainer
    trainer = AdversarialTrainer(classifier.model, classifier.tokenizer)
    
    # Perform adversarial training
    trainer.adversarial_training(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_adv=args.lambda_adv,
        save_path=args.output_path
    )


def main():
    parser = argparse.ArgumentParser(description='Train BERT phishing classifier')
    parser.add_argument('--data-path', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--output-path', type=str, default='data/models/bert_phishing_classifier.pt',
                        help='Path to save trained model')
    parser.add_argument('--adversarial', action='store_true',
                        help='Use adversarial training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--lambda-adv', type=float, default=0.5,
                        help='Adversarial loss weight')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                        help='BERT model name')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        # Load data
        texts, labels = load_data(args.data_path)
        
        # Initialize preprocessor
        preprocessor = EmailPreprocessor()
        
        # Preprocess texts
        processed_texts = preprocess_data(texts, preprocessor)
        
        # Initialize classifier
        classifier = BERTPhishingClassifier(model_name=args.model_name)
        
        # Split data
        split_idx = int(0.8 * len(processed_texts))
        train_texts = processed_texts[:split_idx]
        train_labels = labels[:split_idx]
        val_texts = processed_texts[split_idx:]
        val_labels = labels[split_idx:]
        
        if args.adversarial:
            # Adversarial training
            train_adversarial(classifier, train_texts, train_labels, 
                            val_texts, val_labels, args)
        else:
            # Standard training
            train_loader, val_loader = create_data_loaders(
                processed_texts, labels, classifier.tokenizer, 
                batch_size=args.batch_size
            )
            train_standard(classifier, train_loader, val_loader, args)
        
        logger.info(f"Training completed! Model saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
