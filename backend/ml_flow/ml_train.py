import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HindiNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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


class LSTMBERTModel(nn.Module):
    def __init__(self, bert_model_name: str = "ai4bharat/indic-bert", 
                 hidden_dim: int = 256, lstm_layers: int = 2, dropout: float = 0.3):
        super(LSTMBERTModel, self).__init__()
        
        # BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT parameters initially (can be unfrozen during fine-tuning)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # LSTM processing
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch_size, seq_len, 1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        output = self.dropout(attended_output)
        logits = self.classifier(output)
        
        return logits, attention_weights


class TFIDFClassifier:
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words=None  # We've already removed stopwords in preprocessing
        )
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        
    def fit(self, texts: List[str], labels: List[int]):
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        
        logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'classification_report': report
        }
    
    def save_models(self, save_path: str = "models/"):
        """Save all models"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save LSTM-BERT model
        torch.save(self.lstm_bert_model.state_dict(), f"{save_path}/lstm_bert_model.pth")
        
        # Save TF-IDF model
        with open(f"{save_path}/tfidf_model.pkl", 'wb') as f:
            pickle.dump(self.tfidf_model, f)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{save_path}/tokenizer")
        
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, save_path: str = "models/"):
        """Load all models"""
        # Load LSTM-BERT model
        self.lstm_bert_model.load_state_dict(
            torch.load(f"{save_path}/lstm_bert_model.pth", map_location=self.device)
        )
        
        # Load TF-IDF model
        with open(f"{save_path}/tfidf_model.pkl", 'rb') as f:
            self.tfidf_model = pickle.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(f"{save_path}/tokenizer")
        
        logger.info(f"Models loaded from {save_path}")
    
    def get_attention_weights(self, text: str) -> Dict:
        """Get attention weights for explainability"""
        dataset = HindiNewsDataset([text], [0], self.tokenizer)  # Dummy label
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.lstm_bert_model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits, attention_weights = self.lstm_bert_model(input_ids, attention_mask)
                
                # Get tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                
                # Get attention weights (remove padding)
                mask = attention_mask[0].cpu().numpy()
                valid_length = int(mask.sum())
                
                attention_weights = attention_weights[0][:valid_length].cpu().numpy().flatten()
                tokens = tokens[:valid_length]
                
                return {
                    'tokens': tokens,
                    'attention_weights': attention_weights.tolist(),
                    'prediction': int(torch.argmax(logits, dim=1).item()),
                    'confidence': float(torch.max(F.softmax(logits, dim=1)).item())
                }


class ModelTrainer:
    def __init__(self):
        self.ensemble_model = EnsembleModel()
    
    def train_complete_pipeline(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                              test_df: pd.DataFrame, config: Dict = None):
        """Train the complete ensemble pipeline"""
        
        default_config = {
            'lstm_bert_epochs': 5,
            'lstm_bert_batch_size': 16,
            'lstm_bert_lr': 2e-5,
            'save_models': True
        }
        
        if config:
            default_config.update(config)
        
        # Prepare data
        train_texts = train_df['cleaned_text'].tolist()
        train_labels = train_df['label'].tolist()
        
        val_texts = val_df['cleaned_text'].tolist()
        val_labels = val_df['label'].tolist()
        
        test_texts = test_df['cleaned_text'].tolist()
        test_labels = test_df['label'].tolist()
        
        logger.info("Starting model training pipeline...")
        
        # Train LSTM-BERT model
        logger.info("Training LSTM-BERT model...")
        self.ensemble_model.train_lstm_bert(
            train_texts, train_labels,
            val_texts, val_labels,
            epochs=default_config['lstm_bert_epochs'],
            batch_size=default_config['lstm_bert_batch_size'],
            learning_rate=default_config['lstm_bert_lr']
        )
        
        # Load best LSTM-BERT model
        self.ensemble_model.lstm_bert_model.load_state_dict(
            torch.load('best_lstm_bert_model.pth', map_location=self.ensemble_model.device)
        )
        
        # Train TF-IDF model
        logger.info("Training TF-IDF model...")
        self.ensemble_model.train_tfidf(train_texts, train_labels)
        
        # Evaluate ensemble
        logger.info("Evaluating ensemble model...")
        results = self.ensemble_model.evaluate_ensemble(test_texts, test_labels)
        
        # Save models
        if default_config['save_models']:
            self.ensemble_model.save_models()
        
        return results


if __name__ == "__main__":
    # Example usage
    try:
        # Load preprocessed data
        train_df = pd.read_csv('train_data.csv')
        val_df = pd.read_csv('val_data.csv')
        test_df = pd.read_csv('test_data.csv')
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Training configuration
        config = {
            'lstm_bert_epochs': 3,  # Reduced for quick testing
            'lstm_bert_batch_size': 8,  # Reduced for memory constraints
            'lstm_bert_lr': 2e-5,
            'save_models': True
        }
        
        # Train models
        results = trainer.train_complete_pipeline(train_df, val_df, test_df, config)
        
        print("Training completed!")
        print(f"Final Accuracy: {results['accuracy']:.4f}")
        
        # Test single prediction
        sample_text = "यह एक नकली खबर का उदाहरण है।"  # Example fake news text
        prediction = trainer.ensemble_model.predict_single(sample_text)
        
        print(f"\nSample Prediction:")
        print(f"Text: {sample_text}")
        print(f"Prediction: {'Fake' if prediction['prediction'] == 1 else 'Genuine'}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        
    except FileNotFoundError:
        print("Please run the data preprocessing script first to generate train/val/test data.")
    except Exception as e:
        print(f"Error during training: {e}")
        logger.error(f"Training error: {e}")(f"TF-IDF classifier trained with {X.shape[0]} samples and {X.shape[1]} features")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        # Vectorize texts
        X = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        return predictions, probabilities


class EnsembleModel:
    def __init__(self, bert_model_name: str = "ai4bharat/indic-bert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.lstm_bert_model = LSTMBERTModel(bert_model_name).to(self.device)
        self.tfidf_model = TFIDFClassifier()
        
        # Ensemble weights
        self.lstm_bert_weight = 0.7
        self.tfidf_weight = 0.3
        
    def train_lstm_bert(self, train_texts: List[str], train_labels: List[int],
                       val_texts: List[str], val_labels: List[int],
                       epochs: int = 5, batch_size: int = 16, learning_rate: float = 2e-5):
        
        # Create datasets
        train_dataset = HindiNewsDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = HindiNewsDataset(val_texts, val_labels, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss function
        optimizer = torch.optim.AdamW(self.lstm_bert_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.lstm_bert_model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                logits, _ = self.lstm_bert_model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            
            train_acc = train_correct / len(train_dataset)
            
            # Validation phase
            val_acc, val_loss = self._evaluate_lstm_bert(val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.lstm_bert_model.state_dict(), 'best_lstm_bert_model.pth')
                logger.info("Saved best model!")
    
    def _evaluate_lstm_bert(self, data_loader, criterion):
        self.lstm_bert_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits, _ = self.lstm_bert_model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                total += labels.size(0)
        
        return correct / total, total_loss / len(data_loader)
    
    def train_tfidf(self, train_texts: List[str], train_labels: List[int]):
        self.tfidf_model.fit(train_texts, train_labels)
        logger.info("TF-IDF model trained successfully!")
    
    def predict_single(self, text: str) -> Dict:
        """Predict on a single text sample"""
        return self.predict_batch([text])[0]
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict on a batch of texts using ensemble"""
        results = []
        
        # LSTM-BERT predictions
        lstm_bert_probs = self._predict_lstm_bert_batch(texts)
        
        # TF-IDF predictions
        _, tfidf_probs = self.tfidf_model.predict(texts)
        
        for i, text in enumerate(texts):
            # Ensemble predictions
            lstm_bert_prob = lstm_bert_probs[i]
            tfidf_prob = tfidf_probs[i]
            
            # Weighted ensemble
            ensemble_prob = (self.lstm_bert_weight * lstm_bert_prob + 
                           self.tfidf_weight * tfidf_prob)
            
            prediction = int(np.argmax(ensemble_prob))
            confidence = float(np.max(ensemble_prob))
            
            result = {
                'text': text,
                'prediction': prediction,  # 0: genuine, 1: fake
                'confidence': confidence,
                'lstm_bert_prob': lstm_bert_prob.tolist(),
                'tfidf_prob': tfidf_prob.tolist(),
                'ensemble_prob': ensemble_prob.tolist()
            }
            results.append(result)
        
        return results
    
    def _predict_lstm_bert_batch(self, texts: List[str]) -> np.ndarray:
        """Get LSTM-BERT predictions for a batch of texts"""
        dataset = HindiNewsDataset(texts, [0] * len(texts), self.tokenizer)  # Dummy labels
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.lstm_bert_model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits, _ = self.lstm_bert_model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def evaluate_ensemble(self, test_texts: List[str], test_labels: List[int]) -> Dict:
        """Evaluate the ensemble model"""
        predictions = self.predict_batch(test_texts)
        
        y_pred = [pred['prediction'] for pred in predictions]
        y_true = test_labels
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Genuine', 'Fake'])
        
        logger.info