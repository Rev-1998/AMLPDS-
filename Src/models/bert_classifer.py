# src/models/bert_classifier.py
"""
BERT-based phishing email classifier with adversarial robustness
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BERTPhishingClassifier:
    """BERT-based classifier for phishing detection"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 2):
        """
        Initialize BERT classifier
        
        Args:
            model_name: Pre-trained BERT model name
            num_labels: Number of classification labels (2 for binary)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Set model to evaluation mode by default
        self.model.eval()
        
        logger.info(f"Initialized BERT classifier on {self.device}")
    
    def predict(self, text: str, return_probabilities: bool = False) -> Dict[str, any]:
        """
        Predict if email is phishing or legitimate
        
        Args:
            text: Email text to classify
            return_probabilities: Whether to return probability scores
            
        Returns:
            Dictionary with prediction and confidence
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get prediction and confidence
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities.max().item()
        
        result = {
            'prediction': 'phishing' if prediction == 1 else 'legitimate',
            'confidence': confidence,
            'label': prediction
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities.cpu().numpy().tolist()[0]
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, any]]:
        """
        Predict multiple emails in batches
        
        Args:
            texts: List of email texts
            batch_size: Batch size for processing
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                batch_predictions = torch.argmax(logits, dim=1)
                batch_probabilities = torch.softmax(logits, dim=1)
                
                for j, (pred, probs) in enumerate(zip(batch_predictions, batch_probabilities)):
                    predictions.append({
                        'prediction': 'phishing' if pred.item() == 1 else 'legitimate',
                        'confidence': probs.max().item(),
                        'label': pred.item()
                    })
        
        return predictions
    
    def train_model(self, train_dataloader, val_dataloader, epochs: int = 3, 
                    learning_rate: float = 2e-5, save_path: str = None):
        """
        Train the BERT model
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Path to save the trained model
        """
        from transformers import AdamW, get_linear_schedule_with_warmup
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * epochs
        
        # Create learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
            
            # Validation
            val_accuracy = self.evaluate(val_dataloader)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataloader):.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
    
    def evaluate(self, dataloader) -> float:
        """
        Evaluate model on a dataset
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Accuracy score
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        self.model.train()
        return correct / total
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_name': self.model_name,
            'num_labels': self.num_labels
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Model loaded from {path}")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get BERT embeddings for text"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()


class PhishingDataset(torch.utils.data.Dataset):
    """Custom dataset for phishing emails"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
