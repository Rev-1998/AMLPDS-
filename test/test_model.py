# tests/test_models.py
"""
Tests for ML models
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models.bert_classifier import BERTPhishingClassifier, PhishingDataset
from src.models.adversarial_trainer import AdversarialTrainer


class TestBERTClassifier:
    """Test BERT classifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return BERTPhishingClassifier(model_name='bert-base-uncased')
    
    def test_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier.model is not None
        assert classifier.tokenizer is not None
        assert classifier.num_labels == 2
        assert classifier.device in [torch.device('cuda'), torch.device('cpu')]
    
    def test_predict_single(self, classifier):
        """Test single prediction"""
        text = "Click here to verify your account immediately!"
        result = classifier.predict(text)
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'label' in result
        assert result['prediction'] in ['phishing', 'legitimate']
        assert 0 <= result['confidence'] <= 1
        assert result['label'] in [0, 1]
    
    def test_predict_with_probabilities(self, classifier):
        """Test prediction with probabilities"""
        text = "This is a normal email"
        result = classifier.predict(text, return_probabilities=True)
        
        assert 'probabilities' in result
        assert len(result['probabilities']) == 2
        assert sum(result['probabilities']) == pytest.approx(1.0, rel=1e-5)
    
    def test_predict_batch(self, classifier):
        """Test batch prediction"""
        texts = [
            "Urgent: Update your password",
            "Meeting scheduled for tomorrow",
            "You've won a prize!"
        ]
        results = classifier.predict_batch(texts, batch_size=2)
        
        assert len(results) == 3
        for result in results:
            assert 'prediction' in result
            assert 'confidence' in result
    
    def test_get_embeddings(self, classifier):
        """Test embedding extraction"""
        text = "Test email content"
        embeddings = classifier.get_embeddings(text)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1  # Batch size
        assert embeddings.shape[1] == 768  # BERT base hidden size


class TestPhishingDataset:
    """Test dataset class"""
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        texts = ["Email 1", "Email 2", "Email 3"]
        labels = [0, 1, 0]
        
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        dataset = PhishingDataset(texts, labels, tokenizer)
        
        assert len(dataset) == 3
        
        # Test single item
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        assert item['labels'].item() == 0


class TestAdversarialTrainer:
    """Test adversarial trainer"""
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance"""
        classifier = BERTPhishingClassifier()
        return AdversarialTrainer(classifier.model, classifier.tokenizer)
    
    def test_initialization(self, trainer):
        """Test trainer initialization"""
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.attack is not None
    
    @patch('textattack.Attack.attack')
    def test_generate_adversarial_examples(self, mock_attack, trainer):
        """Test adversarial example generation"""
        # Mock attack result
        mock_result = Mock()
        mock_result.perturbed_result.attacked_text = "Perturbed text"
        mock_result.perturbed_result.output = 1
        mock_result.goal_status = 0  # Success
        mock_result.num_queries = 10
        mock_attack.return_value = mock_result
        
        texts = ["Original text"]
        labels = [0]
        
        examples = trainer.generate_adversarial_examples(texts, labels)
        
        assert len(examples) == 1
        assert examples[0]['original_text'] == "Original text"
        assert examples[0]['adversarial_text'] == "Perturbed text"
        assert examples[0]['success'] == True
    
    def test_calculate_perturbation(self, trainer):
        """Test perturbation calculation"""
        original = "This is the original text"
        perturbed = "This was the modified text"
        
        perturbation = trainer._calculate_perturbation(original, perturbed)
        
        assert isinstance(perturbation, float)
        assert 0 <= perturbation <= 100
