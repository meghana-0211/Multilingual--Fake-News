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