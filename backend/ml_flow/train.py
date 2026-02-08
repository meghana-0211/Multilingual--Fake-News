"""
MODEL ARCHITECTURE
IndicBERT + BiLSTM with Attention for Multilingual Fake News Detection

Author: Your Name
Architecture: Ensemble of IndicBERT and BiLSTM with Multi-head Attention
Languages: Hindi, Gujarati, Marathi, Telugu
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BiLSTMWithAttention(nn.Module):
    """BiLSTM with Multi-head Attention layer"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 because bidirectional
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, input_dim]
            mask: [batch, seq_len] - attention mask
        
        Returns:
            output: [batch, hidden_dim * 2]
            attention_weights: [batch, seq_len, seq_len]
        """
        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        
        # Multi-head attention
        # Query, Key, Value are all the LSTM output
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask  # Mask padding positions
        )
        
        # Residual connection + Layer norm
        attn_out = self.layer_norm(attn_out + lstm_out)
        
        # Global average pooling over sequence
        if mask is not None:
            # Mask out padding before pooling
            mask_expanded = (~mask).unsqueeze(-1).float()  # [batch, seq_len, 1]
            attn_out = attn_out * mask_expanded
            pooled = attn_out.sum(dim=1) / mask_expanded.sum(dim=1)  # [batch, hidden_dim*2]
        else:
            pooled = attn_out.mean(dim=1)  # [batch, hidden_dim*2]
        
        return pooled, attn_weights


class IndicBERTBiLSTMEnsemble(nn.Module):
    """
    Ensemble model combining IndicBERT and BiLSTM with Attention
    
    Architecture:
    1. IndicBERT path: Pretrained transformer for contextual understanding
    2. BiLSTM path: Sequential processing with attention for pattern detection
    3. Ensemble fusion: Weighted combination of both paths
    """
    
    def __init__(
        self,
        bert_model_name='ai4bharat/indic-bert',
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        attention_heads=8,
        dropout=0.3,
        num_classes=2,
        freeze_bert_layers=0  # Number of BERT layers to freeze (0 = train all)
    ):
        super().__init__()
        
        # IndicBERT component
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_config = AutoConfig.from_pretrained(bert_model_name)
        bert_dim = self.bert_config.hidden_size  # 768 for base model
        
        # Optionally freeze early BERT layers
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # BiLSTM component
        self.bilstm = BiLSTMWithAttention(
            input_dim=bert_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification heads
        # BERT path classifier
        self.bert_classifier = nn.Sequential(
            nn.Linear(bert_dim, bert_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bert_dim // 2, num_classes)
        )
        
        # BiLSTM path classifier
        lstm_output_dim = lstm_hidden_dim * 2  # Bidirectional
        self.lstm_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
        
        # Ensemble fusion
        # Learnable weights for combining predictions
        self.ensemble_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # BERT=0.6, LSTM=0.4
        
        # Alternative: Use a small network to combine features
        self.fusion_layer = nn.Sequential(
            nn.Linear(bert_dim + lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.use_fusion_layer = True  # Set to True for better performance
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            dict with:
                - logits: [batch, num_classes]
                - bert_logits: [batch, num_classes]
                - lstm_logits: [batch, num_classes]
                - attention_weights: [batch, num_heads, seq_len, seq_len]
                - bert_embeddings: [batch, bert_dim]
                - lstm_embeddings: [batch, lstm_dim]
        """
        # =========================================
        # BERT Path
        # =========================================
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get [CLS] token representation for classification
        bert_pooled = bert_outputs.pooler_output  # [batch, bert_dim]
        bert_pooled = self.dropout(bert_pooled)
        
        # BERT classification
        bert_logits = self.bert_classifier(bert_pooled)
        
        # =========================================
        # BiLSTM Path
        # =========================================
        # Use BERT's last hidden state as input to LSTM
        bert_sequence = bert_outputs.last_hidden_state  # [batch, seq_len, bert_dim]
        
        # Create padding mask for LSTM (True where padding)
        padding_mask = (attention_mask == 0)
        
        # BiLSTM with attention
        lstm_pooled, attn_weights = self.bilstm(bert_sequence, padding_mask)
        lstm_pooled = self.dropout(lstm_pooled)
        
        # LSTM classification
        lstm_logits = self.lstm_classifier(lstm_pooled)
        
        # =========================================
        # Ensemble Fusion
        # =========================================
        if self.use_fusion_layer:
            # Concatenate BERT and LSTM features
            combined_features = torch.cat([bert_pooled, lstm_pooled], dim=1)
            final_logits = self.fusion_layer(combined_features)
        else:
            # Weighted average of logits
            weights = torch.softmax(self.ensemble_weights, dim=0)
            final_logits = weights[0] * bert_logits + weights[1] * lstm_logits
        
        return {
            'logits': final_logits,
            'bert_logits': bert_logits,
            'lstm_logits': lstm_logits,
            'attention_weights': attn_weights,
            'bert_embeddings': bert_pooled,
            'lstm_embeddings': lstm_pooled
        }


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def test_model():
    """Test model architecture"""
    print("="*60)
    print("TESTING MODEL ARCHITECTURE")
    print("="*60)
    
    # Create model
    model = IndicBERTBiLSTMEnsemble()
    
    # Count parameters
    trainable, total = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Size: ~{total * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    # Create dummy inputs
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    print(f"\nTesting forward pass...")
    print(f"  Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  BERT logits: {outputs['bert_logits'].shape}")
    print(f"  LSTM logits: {outputs['lstm_logits'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")
    print(f"  BERT embeddings: {outputs['bert_embeddings'].shape}")
    print(f"  LSTM embeddings: {outputs['lstm_embeddings'].shape}")
    
    print("\n✓ Model architecture test passed!")


if __name__ == '__main__':
    test_model()