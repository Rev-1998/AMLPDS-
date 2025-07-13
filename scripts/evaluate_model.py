# scripts/evaluate_model.py
#!/usr/bin/env python3
"""
Script to evaluate trained model performance
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bert_classifier import BERTPhishingClassifier
from src.models.adversarial_trainer import AdversarialTrainer
from src.preprocessing.email_preprocessor import EmailPreprocessor
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_test_data(data_path):
    """Load test data"""
    logger.info(f"Loading test data from {data_path}")
    
    # Load test set
    test_df = pd.read_csv(os.path.join(data_path, 'test_emails.csv'))
    texts = test_df['email_text'].tolist()
    labels = test_df['label'].tolist()
    
    logger.info(f"Loaded {len(texts)} test emails")
    
    return texts, labels


def evaluate_clean_performance(classifier, texts, labels):
    """Evaluate model on clean test data"""
    logger.info("Evaluating clean performance...")
    
    predictions = []
    probabilities = []
    
    # Make predictions
    for text in texts:
        result = classifier.predict(text, return_probabilities=True)
        predictions.append(1 if result['prediction'] == 'phishing' else 0)
        probabilities.append(result['probabilities'])
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # Calculate AUC-ROC
    probs_positive = [p[1] for p in probabilities]
    auc_roc = roc_auc_score(labels, probs_positive)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(labels),
        'correct_predictions': sum(1 for p, l in zip(predictions, labels) if p == l)
    }
    
    return results, predictions, probabilities


def evaluate_adversarial_robustness(classifier, texts, labels, num_samples=100):
    """Evaluate model robustness against adversarial attacks"""
    logger.info("Evaluating adversarial robustness...")
    
    # Sample subset for adversarial evaluation (for speed)
    if len(texts) > num_samples:
        indices = np.random.choice(len(texts), num_samples, replace=False)
        sample_texts = [texts[i] for i in indices]
        sample_labels = [labels[i] for i in indices]
    else:
        sample_texts = texts
        sample_labels = labels
    
    # Initialize adversarial trainer
    trainer = AdversarialTrainer(classifier.model, classifier.tokenizer)
    
    # Evaluate robustness
    robustness_results = trainer.evaluate_robustness(sample_texts, sample_labels)
    
    return robustness_results


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def generate_report(clean_results, adversarial_results, output_path):
    """Generate evaluation report"""
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'clean_performance': clean_results,
        'adversarial_robustness': adversarial_results
    }
    
    # Save JSON report
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate text report
    with open(output_path, 'w') as f:
        f.write("AMLPDS Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {report['evaluation_date']}\n\n")
        
        f.write("Clean Data Performance:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy: {clean_results['accuracy']:.4f}\n")
        f.write(f"Precision: {clean_results['precision']:.4f}\n")
        f.write(f"Recall: {clean_results['recall']:.4f}\n")
        f.write(f"F1-Score: {clean_results['f1_score']:.4f}\n")
        f.write(f"AUC-ROC: {clean_results['auc_roc']:.4f}\n")
        f.write(f"Total Samples: {clean_results['total_samples']}\n")
        f.write(f"Correct Predictions: {clean_results['correct_predictions']}\n\n")
        
        f.write("Adversarial Robustness:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Clean Accuracy: {adversarial_results['clean_accuracy']:.4f}\n")
        f.write(f"Robust Accuracy: {adversarial_results['textfooler_robust_accuracy']:.4f}\n")
        f.write(f"Attack Success Rate: {adversarial_results['textfooler_attack_success_rate']:.4f}\n")
        f.write(f"Avg Perturbation: {adversarial_results['textfooler_avg_perturbation']:.2f}%\n")
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT phishing classifier')
    parser.add_argument('--model-path', type=str, default='data/models/bert_phishing_classifier.pt',
                        help='Path to trained model')
    parser.add_argument('--data-path', type=str, default='data/processed',
                        help='Path to test data')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--test-adversarial', action='store_true',
                        help='Test adversarial robustness')
    parser.add_argument('--num-adversarial-samples', type=int, default=100,
                        help='Number of samples for adversarial testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model_path}")
        classifier = BERTPhishingClassifier()
        classifier.load_model(args.model_path)
        
        # Load test data
        texts, labels = load_test_data(args.data_path)
        
        # Initialize preprocessor
        preprocessor = EmailPreprocessor()
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            result = preprocessor.preprocess_email(text)
            processed_texts.append(result['cleaned_text'])
        
        # Evaluate clean performance
        clean_results, predictions, probabilities = evaluate_clean_performance(
            classifier, processed_texts, labels
        )
        
        # Plot confusion matrix
        cm = np.array(clean_results['confusion_matrix'])
        plot_confusion_matrix(cm, os.path.join(args.output_dir, 'confusion_matrix.png'))
        
        # Evaluate adversarial robustness
        adversarial_results = {}
        if args.test_adversarial:
            adversarial_results = evaluate_adversarial_robustness(
                classifier, processed_texts, labels, args.num_adversarial_samples
            )
        
        # Generate report
        report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
        generate_report(clean_results, adversarial_results, report_path)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
