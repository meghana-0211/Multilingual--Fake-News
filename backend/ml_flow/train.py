# scripts/train_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import json
from multilingual_preprocessor import MultilingualPreprocessor

import torch
import torch.nn as nn
from transformers import AutoModel

class LSTMBERTEnsemble(nn.Module):
    def __init__(self, 
                 bert_model='ai4bharat/indic-bert',
                 lstm_hidden=256,
                 lstm_layers=2,
                 dropout=0.3,
                 num_classes=2):
        super().__init__()
        
        # BERT component
        self.bert = AutoModel.from_pretrained(bert_model)
        bert_dim = self.bert.config.hidden_size
        
        # BiLSTM component
        self.lstm = nn.LSTM(
            bert_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification heads
        self.dropout = nn.Dropout(dropout)
        
        # BERT classifier
        self.bert_classifier = nn.Linear(bert_dim, num_classes)
        
        # LSTM classifier
        self.lstm_classifier = nn.Linear(lstm_hidden * 2, num_classes)
        
        # Ensemble fusion
        self.fusion = nn.Linear(num_classes * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # BERT forward
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token for classification
        bert_pooled = bert_output.pooler_output  # [batch, bert_dim]
        bert_logits = self.bert_classifier(self.dropout(bert_pooled))
        
        # Get sequence output for LSTM
        bert_sequence = bert_output.last_hidden_state  # [batch, seq_len, bert_dim]
        
        # BiLSTM forward
        lstm_out, (hidden, cell) = self.lstm(bert_sequence)  # [batch, seq_len, lstm_hidden*2]
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Mean pooling with attention
        attn_pooled = torch.mean(attn_out, dim=1)  # [batch, lstm_hidden*2]
        
        lstm_logits = self.lstm_classifier(self.dropout(attn_pooled))
        
        # Ensemble fusion
        combined = torch.cat([bert_logits, lstm_logits], dim=1)
        final_logits = self.fusion(combined)
        
        return {
            'logits': final_logits,
            'bert_logits': bert_logits,
            'lstm_logits': lstm_logits,
            'attention_weights': attn_weights,
            'bert_embeddings': bert_pooled
        }
    
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, languages, preprocessor):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding, _ = self.preprocessor.preprocess_batch(
            [self.texts[idx]], 
            [self.languages[idx]]
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    
    # Calculate per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, 
                                   target_names=['Fake', 'Real'],
                                   output_dict=True)
    
    return accuracy, report

def main():
    # Load data
    df = pd.read_csv('data/processed/multilingual_dataset.csv')
    
    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)
    
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = MultilingualPreprocessor()
    model = LSTMBERTEnsemble().to(device)
    
    # Datasets
    train_dataset = FakeNewsDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        train_df['language'].tolist(),
        preprocessor
    )
    val_dataset = FakeNewsDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        val_df['language'].tolist(),
        preprocessor
    )
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Scheduler
    num_epochs = 5
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc, val_report = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(json.dumps(val_report, indent=2))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'data/models/best_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()