# src/models/adversarial_trainer.py
"""
Adversarial training module using TextAttack framework
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from textattack import Attack, AttackArgs, Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """Handles adversarial training for phishing detection model"""
    
    def __init__(self, model, tokenizer):
        """
        Initialize adversarial trainer
        
        Args:
            model: BERT model instance
            tokenizer: BERT tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Create model wrapper for TextAttack
        self.model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        
        # Initialize TextFooler attack
        self.attack = TextFoolerJin2019.build(self.model_wrapper)
        
        logger.info("Initialized adversarial trainer with TextFooler attack")
    
    def generate_adversarial_examples(self, texts: List[str], labels: List[int], 
                                    num_examples: int = None) -> List[Dict]:
        """
        Generate adversarial examples using TextFooler
        
        Args:
            texts: Original email texts
            labels: Original labels (0 or 1)
            num_examples: Number of examples to generate (None for all)
            
        Returns:
            List of adversarial examples with metadata
        """
        adversarial_examples = []
        
        # Create dataset
        dataset = [(text, label) for text, label in zip(texts, labels)]
        if num_examples:
            dataset = dataset[:num_examples]
        
        # Generate adversarial examples
        for i, (text, label) in enumerate(tqdm(dataset, desc="Generating adversarial examples")):
            try:
                # Create attack input
                result = self.attack.attack(text, label)
                
                if result.perturbed_result.attacked_text != text:
                    adversarial_examples.append({
                        'original_text': text,
                        'adversarial_text': result.perturbed_result.attacked_text,
                        'original_label': label,
                        'predicted_label': result.perturbed_result.output,
                        'success': result.goal_status == 0,  # 0 means successful attack
                        'num_queries': result.num_queries,
                        'perturbation_percentage': self._calculate_perturbation(
                            text, result.perturbed_result.attacked_text
                        )
                    })
                else:
                    # Attack failed, keep original
                    adversarial_examples.append({
                        'original_text': text,
                        'adversarial_text': text,
                        'original_label': label,
                        'predicted_label': label,
                        'success': False,
                        'num_queries': 0,
                        'perturbation_percentage': 0.0
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to generate adversarial example {i}: {e}")
                # Keep original on failure
                adversarial_examples.append({
                    'original_text': text,
                    'adversarial_text': text,
                    'original_label': label,
                    'predicted_label': label,
                    'success': False,
                    'num_queries': 0,
                    'perturbation_percentage': 0.0
                })
        
        return adversarial_examples
    
    def adversarial_training_step(self, clean_texts: List[str], labels: List[int], 
                                 lambda_adv: float = 0.5) -> torch.Tensor:
        """
        Perform one adversarial training step
        
        Args:
            clean_texts: Original training texts
            labels: Training labels
            lambda_adv: Weight for adversarial loss
            
        Returns:
            Combined loss tensor
        """
        # Generate adversarial examples
        adv_examples = self.generate_adversarial_examples(clean_texts, labels)
        adv_texts = [ex['adversarial_text'] for ex in adv_examples]
        
        # Compute clean loss
        clean_loss = self._compute_loss(clean_texts, labels)
        
        # Compute adversarial loss
        adv_loss = self._compute_loss(adv_texts, labels)
        
        # Combined loss
        total_loss = clean_loss + lambda_adv * adv_loss
        
        return total_loss
    
    def adversarial_training(self, train_texts: List[str], train_labels: List[int],
                           val_texts: List[str], val_labels: List[int],
                           epochs: int = 3, batch_size: int = 8,
                           learning_rate: float = 2e-5, lambda_adv: float = 0.5,
                           save_path: str = None):
        """
        Full adversarial training loop
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            lambda_adv: Adversarial loss weight
            save_path: Path to save model
        """
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Calculate total steps
        steps_per_epoch = len(train_texts) // batch_size
        total_steps = steps_per_epoch * epochs
        
        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(len(train_texts))
            train_texts = [train_texts[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]
            
            # Process batches
            for i in tqdm(range(0, len(train_texts), batch_size), desc=f"Epoch {epoch+1}"):
                batch_texts = train_texts[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                # Generate adversarial examples for batch
                adv_examples = self.generate_adversarial_examples(batch_texts, batch_labels)
                adv_texts = [ex['adversarial_text'] for ex in adv_examples]
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Compute losses
                clean_loss = self._compute_loss(batch_texts, batch_labels)
                adv_loss = self._compute_loss(adv_texts, batch_labels)
                
                # Combined loss
                loss = clean_loss + lambda_adv * adv_loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
            
            # Validation
            val_accuracy = self._evaluate(val_texts, val_labels)
            adv_val_examples = self.generate_adversarial_examples(
                val_texts[:100], val_labels[:100]  # Subset for speed
            )
            adv_accuracy = self._evaluate(
                [ex['adversarial_text'] for ex in adv_val_examples],
                val_labels[:100]
            )
            
            avg_loss = total_loss / steps_per_epoch
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}, Adv Val Acc: {adv_accuracy:.4f}")
        
        # Save model
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
    
    def evaluate_robustness(self, test_texts: List[str], test_labels: List[int],
                          attack_types: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model robustness against various attacks
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            attack_types: List of attack types to test
            
        Returns:
            Dictionary of robustness metrics
        """
        if attack_types is None:
            attack_types = ['textfooler']
        
        results = {}
        
        # Clean accuracy
        clean_acc = self._evaluate(test_texts, test_labels)
        results['clean_accuracy'] = clean_acc
        
        # Adversarial accuracy
        for attack_type in attack_types:
            logger.info(f"Evaluating against {attack_type}")
            
            # Generate adversarial examples
            adv_examples = self.generate_adversarial_examples(test_texts, test_labels)
            
            # Calculate metrics
            successful_attacks = sum(1 for ex in adv_examples if ex['success'])
            attack_success_rate = successful_attacks / len(adv_examples)
            
            adv_texts = [ex['adversarial_text'] for ex in adv_examples]
            adv_accuracy = self._evaluate(adv_texts, test_labels)
            
            avg_perturbation = np.mean([ex['perturbation_percentage'] 
                                       for ex in adv_examples if ex['success']])
            
            results[f'{attack_type}_attack_success_rate'] = attack_success_rate
            results[f'{attack_type}_robust_accuracy'] = adv_accuracy
            results[f'{attack_type}_avg_perturbation'] = avg_perturbation
        
        return results
    
    def _compute_loss(self, texts: List[str], labels: List[int]) -> torch.Tensor:
        """Compute loss for a batch of texts"""
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=labels_tensor)
        
        return outputs.loss
    
    def _evaluate(self, texts: List[str], labels: List[int], batch_size: int = 32) -> float:
        """Evaluate accuracy on texts"""
        self.model.eval()
        correct = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Predict
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                # Count correct
                batch_labels_tensor = torch.tensor(batch_labels).to(self.device)
                correct += (predictions == batch_labels_tensor).sum().item()
        
        self.model.train()
        return correct / len(texts)
    
    def _calculate_perturbation(self, original: str, perturbed: str) -> float:
        """Calculate word-level perturbation percentage"""
        original_words = original.lower().split()
        perturbed_words = perturbed.lower().split()
        
        if len(original_words) == 0:
            return 0.0
        
        # Simple word-level comparison
        changes = sum(1 for o, p in zip(original_words, perturbed_words) if o != p)
        
        # Account for length differences
        length_diff = abs(len(original_words) - len(perturbed_words))
        total_changes = changes + length_diff
        
        return (total_changes / len(original_words)) * 100
