# scripts/preprocess_data.py
#!/usr/bin/env python3
"""
Script to preprocess raw email data
"""

import os
import sys
import argparse
import logging
import pandas as pd
import re
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.email_preprocessor import EmailPreprocessor
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_phishtank_data(file_path):
    """Load PhishTank data"""
    logger.info(f"Loading PhishTank data from {file_path}")
    
    # PhishTank CSV format may vary, adjust columns as needed
    df = pd.read_csv(file_path)
    
    # Extract email content (adjust based on actual format)
    emails = []
    for _, row in df.iterrows():
        # Combine available text fields
        email_text = str(row.get('url', '')) + ' ' + str(row.get('target', ''))
        emails.append({
            'email_text': email_text,
            'label': 1  # Phishing
        })
    
    return pd.DataFrame(emails)


def load_enron_data(directory_path, sample_size=10000):
    """Load Enron email data"""
    logger.info(f"Loading Enron data from {directory_path}")
    
    emails = []
    count = 0
    
    # Walk through Enron directory structure
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if count >= sample_size:
                break
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    emails.append({
                        'email_text': content,
                        'label': 0  # Legitimate
                    })
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
    
    return pd.DataFrame(emails)


def clean_and_split_data(df, train_ratio=0.8, val_ratio=0.1):
    """Clean data and split into train/val/test sets"""
    logger.info("Cleaning and splitting data...")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['email_text'])
    
    # Remove empty emails
    df = df[df['email_text'].str.strip() != '']
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def save_processed_data(train_df, val_df, test_df, output_dir):
    """Save processed data"""
    logger.info(f"Saving processed data to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save separate files for phishing and legitimate emails
    train_phishing = train_df[train_df['label'] == 1]
    train_legitimate = train_df[train_df['label'] == 0]
    
    train_phishing.to_csv(os.path.join(output_dir, 'phishing_emails.csv'), index=False)
    train_legitimate.to_csv(os.path.join(output_dir, 'legitimate_emails.csv'), index=False)
    
    # Save validation and test sets
    val_df.to_csv(os.path.join(output_dir, 'val_emails.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_emails.csv'), index=False)
    
    # Save combined training set
    train_df.to_csv(os.path.join(output_dir, 'train_emails.csv'), index=False)
    
    logger.info("Data saved successfully")


def main():
    parser = argparse.ArgumentParser(description='Preprocess email data')
    parser.add_argument('--phishtank-path', type=str, 
                        default='data/raw/phishtank/phishtank.csv',
                        help='Path to PhishTank data')
    parser.add_argument('--enron-path', type=str,
                        default='data/raw/enron',
                        help='Path to Enron data directory')
    parser.add_argument('--output-dir', type=str,
                        default='data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Number of samples to use from each dataset')
    
    args = parser.parse_args()
    
    try:
        # Load data
        phishing_df = load_phishtank_data(args.phishtank_path)
        legitimate_df = load_enron_data(args.enron_path, args.sample_size)
        
        # Sample data if needed
        if len(phishing_df) > args.sample_size:
            phishing_df = phishing_df.sample(n=args.sample_size, random_state=42)
        
        # Combine data
        all_data = pd.concat([phishing_df, legitimate_df], ignore_index=True)
        
        logger.info(f"Total emails: {len(all_data)} "
                   f"(Phishing: {sum(all_data['label'] == 1)}, "
                   f"Legitimate: {sum(all_data['label'] == 0)})")
        
        # Clean and split data
        train_df, val_df, test_df = clean_and_split_data(all_data)
        
        # Save processed data
        save_processed_data(train_df, val_df, test_df, args.output_dir)
        
        logger.info("Data preprocessing completed!")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
