import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import lime
from lime.lime_text import LimeTextExplainer
import shap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import logging

logger = logging.getLogger(__name__)

import torch
import numpy as np
from transformers import AutoTokenizer
# adjust import if needed

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
    
class EnsembleModelWrapper:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

        # Load model
        self.model = LSTMBERTEnsemble()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

    def predict_batch(self, texts):
        enc = self._encode(texts)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(outputs["logits"], dim=1)

        results = []
        for p in probs.cpu().numpy():
            results.append({
                "ensemble_prob": p,
                "prediction": int(np.argmax(p)),
                "confidence": float(np.max(p))
            })

        return results

    def predict_single(self, text):
        return self.predict_batch([text])[0]

    def get_attention_weights(self, text):
        enc = self._encode([text])
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(enc["input_ids"], enc["attention_mask"])

        # Check if attention_weights exists in outputs
        if "attention_weights" in outputs:
            attn = outputs["attention_weights"].mean(dim=1).squeeze(0).cpu().numpy()
        else:
            # If model doesn't return attention weights, create dummy ones
            attn = np.ones(len(enc["input_ids"][0])) / len(enc["input_ids"][0])

        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

        probs = torch.softmax(outputs["logits"], dim=1)[0]

        return {
            "tokens": tokens,
            "attention_weights": attn.tolist(),
            "prediction": int(torch.argmax(probs)),
            "confidence": float(torch.max(probs))
        }


class ExplainabilityModule:
    def __init__(self, ensemble_model, class_names: List[str] = ['Genuine', 'Fake']):
        self.ensemble_model = ensemble_model
        self.class_names = class_names
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=class_names,
            feature_selection='auto',
            split_expression=r'\s+',  # Split by whitespace
            bow=True,
        )
    
    def explain_with_lime(self, text: str, num_features: int = 10) -> Dict:
        """Generate LIME explanation for a text"""
        try:
            # Define prediction function for LIME
            def predict_fn(texts):
                predictions = self.ensemble_model.predict_batch(texts)
                probabilities = np.array([pred['ensemble_prob'] for pred in predictions])
                return probabilities
            
            # Generate explanation
            explanation = self.lime_explainer.explain_instance(
                text, 
                predict_fn,
                num_features=num_features,
                num_samples=1000
            )
            
            # Extract explanation data
            explanation_data = {
                'text': text,
                'prediction_class': explanation.predicted_class,
                'prediction_proba': explanation.predict_proba.tolist(),
                'local_exp': explanation.local_exp,
                'intercept': explanation.intercept,
                'score': explanation.score
            }
            
            # Get feature importance
            feature_importance = explanation.as_list()
            explanation_data['feature_importance'] = feature_importance
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"LIME explanation error: {e}")
            return {'error': str(e)}
    
    def explain_with_attention(self, text: str) -> Dict:
        """Get attention-based explanation from LSTM-BERT model"""
        try:
            attention_data = self.ensemble_model.get_attention_weights(text)
            
            # Process tokens and attention weights
            tokens = attention_data['tokens']
            weights = attention_data['attention_weights']
            
            # Remove special tokens and combine subwords
            processed_tokens, processed_weights = self._process_tokens_weights(tokens, weights)
            
            # Sort by importance
            token_importance = list(zip(processed_tokens, processed_weights))
            token_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return {
                'text': text,
                'tokens': processed_tokens,
                'attention_weights': processed_weights,
                'token_importance': token_importance[:15],  # Top 15 most important
                'prediction': attention_data['prediction'],
                'confidence': attention_data['confidence']
            }
            
        except Exception as e:
            logger.error(f"Attention explanation error: {e}")
            return {'error': str(e)}
    
    def _process_tokens_weights(self, tokens: List[str], weights: List[float]) -> Tuple[List[str], List[float]]:
        """Process BERT tokens and combine subwords"""
        processed_tokens = []
        processed_weights = []
        
        current_word = ""
        current_weight = 0.0
        word_count = 0
        
        for token, weight in zip(tokens, weights):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Handle subword tokens (starting with ##)
            if token.startswith('##'):
                current_word += token[2:]  # Remove ##
                current_weight += weight
                word_count += 1
            else:
                # Save previous word if exists
                if current_word:
                    processed_tokens.append(current_word)
                    processed_weights.append(current_weight / word_count if word_count > 0 else current_weight)
                
                # Start new word
                current_word = token
                current_weight = weight
                word_count = 1
        
        # Add last word
        if current_word:
            processed_tokens.append(current_word)
            processed_weights.append(current_weight / word_count if word_count > 0 else current_weight)
        
        return processed_tokens, processed_weights
    
    def generate_comprehensive_explanation(self, text: str) -> Dict:
        """Generate comprehensive explanation using multiple methods"""
        
        # Get base prediction
        prediction_result = self.ensemble_model.predict_single(text)
        
        # Get LIME explanation
        lime_explanation = self.explain_with_lime(text)
        
        # Get attention explanation
        attention_explanation = self.explain_with_attention(text)
        
        # Combine explanations
        comprehensive_explanation = {
            'text': text,
            'prediction': {
                'class': 'Fake' if prediction_result['prediction'] == 1 else 'Genuine',
                'confidence': prediction_result['confidence'],
                'ensemble_prob': prediction_result['ensemble_prob'].tolist()
            },
            'lime_explanation': lime_explanation,
            'attention_explanation': attention_explanation,
            'summary': self._create_explanation_summary(lime_explanation, attention_explanation)
        }
        
        return comprehensive_explanation
    
    def _create_explanation_summary(self, lime_exp: Dict, attention_exp: Dict) -> Dict:
        """Create a summary of explanations"""
        summary = {
            'key_indicators': [],
            'explanation_confidence': 0.0,
            'top_influential_words': []
        }
        
        try:
            # Get top words from LIME
            if 'feature_importance' in lime_exp:
                lime_words = [(word, score) for word, score in lime_exp['feature_importance'][:10]]
                summary['lime_top_words'] = lime_words
            
            # Get top words from attention
            if 'token_importance' in attention_exp:
                attention_words = attention_exp['token_importance'][:10]
                summary['attention_top_words'] = attention_words
            
            # Combine and rank all influential words
            all_words = {}
            
            # Add LIME words
            if 'feature_importance' in lime_exp:
                for word, score in lime_exp['feature_importance']:
                    all_words[word] = all_words.get(word, 0) + abs(score)
            
            # Add attention words
            if 'token_importance' in attention_exp:
                for word, score in attention_exp['token_importance']:
                    all_words[word] = all_words.get(word, 0) + abs(score) * 0.5  # Weight attention less
            
            # Sort combined words
            top_combined = sorted(all_words.items(), key=lambda x: x[1], reverse=True)[:10]
            summary['top_influential_words'] = top_combined
            
        except Exception as e:
            logger.error(f"Summary creation error: {e}")
        
        return summary
    
    def visualize_explanation(self, explanation_data: Dict, save_path: str = None) -> Dict:
        """Create visualizations for explanations"""
        visualizations = {}
        
        try:
            # 1. Feature importance bar plot
            if 'lime_explanation' in explanation_data and 'feature_importance' in explanation_data['lime_explanation']:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                words, scores = zip(*explanation_data['lime_explanation']['feature_importance'][:15])
                colors = ['green' if score > 0 else 'red' for score in scores]
                
                bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.7)
                ax.set_yticks(range(len(words)))
                ax.set_yticklabels(words)
                ax.set_xlabel('Feature Importance Score')
                ax.set_title('LIME Feature Importance Analysis')
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(f"{save_path}_lime_features.png", dpi=300, bbox_inches='tight')
                
                visualizations['lime_features'] = fig
            
            # 2. Attention heatmap
            if 'attention_explanation' in explanation_data:
                attention_data = explanation_data['attention_explanation']
                if 'tokens' in attention_data and 'attention_weights' in attention_data:
                    tokens = attention_data['tokens'][:50]  # Limit to first 50 tokens
                    weights = attention_data['attention_weights'][:50]
                    
                    fig, ax = plt.subplots(figsize=(15, 3))
                    
                    # Create heatmap
                    weights_2d = np.array(weights).reshape(1, -1)
                    im = ax.imshow(weights_2d, cmap='Reds', aspect='auto')
                    
                    ax.set_xticks(range(len(tokens)))
                    ax.set_xticklabels(tokens, rotation=45, ha='right')
                    ax.set_yticks([])
                    ax.set_title('Attention Weights Visualization')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
                    
                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(f"{save_path}_attention_heatmap.png", dpi=300, bbox_inches='tight')
                    
                    visualizations['attention_heatmap'] = fig
            
            # 3. Word cloud of important features
            if 'summary' in explanation_data and 'top_influential_words' in explanation_data['summary']:
                word_freq = dict(explanation_data['summary']['top_influential_words'])
                
                if word_freq:
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=50
                    ).generate_from_frequencies(word_freq)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Most Influential Words')
                    
                    if save_path:
                        plt.savefig(f"{save_path}_wordcloud.png", dpi=300, bbox_inches='tight')
                    
                    visualizations['wordcloud'] = fig
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def generate_text_explanation(self, explanation_data: Dict) -> str:
        """Generate human-readable text explanation"""
        try:
            text = explanation_data['text'][:100] + "..." if len(explanation_data['text']) > 100 else explanation_data['text']
            prediction = explanation_data['prediction']
            
            explanation = f"Analysis Results for: \"{text}\"\n"
            explanation += "=" * 50 + "\n\n"
            
            explanation += f"Prediction: {prediction['class']}\n"
            explanation += f"Confidence: {prediction['confidence']:.2%}\n\n"
            
            if 'summary' in explanation_data and 'top_influential_words' in explanation_data['summary']:
                explanation += "Key Influential Words:\n"
                for word, importance in explanation_data['summary']['top_influential_words'][:10]:
                    explanation += f"  • {word}: {importance:.3f}\n"
                explanation += "\n"
            
            if 'lime_explanation' in explanation_data and 'feature_importance' in explanation_data['lime_explanation']:
                explanation += "LIME Analysis - Words supporting the prediction:\n"
                for word, score in explanation_data['lime_explanation']['feature_importance'][:5]:
                    direction = "SUPPORTS" if score > 0 else "OPPOSES"
                    explanation += f"  • '{word}' {direction} the prediction (score: {score:.3f})\n"
                explanation += "\n"
            
            explanation += "Model Ensemble Probabilities:\n"
            probs = prediction['ensemble_prob']
            explanation += f"  • Genuine: {probs[0]:.4f}\n"
            explanation += f"  • Fake: {probs[1]:.4f}\n"
            
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"


class ExplanationServer:
    """Server component to handle explanation requests"""
    
    def __init__(self, ensemble_model):
        self.explainer = ExplainabilityModule(ensemble_model)
    
    def explain_article(self, text: str, explanation_type: str = "comprehensive") -> Dict:
        """Main endpoint for article explanation"""
        
        if explanation_type == "lime":
            return self.explainer.explain_with_lime(text)
        elif explanation_type == "attention":
            return self.explainer.explain_with_attention(text)
        elif explanation_type == "comprehensive":
            return self.explainer.generate_comprehensive_explanation(text)
        else:
            return {"error": "Unknown explanation type"}
    
    def get_explanation_with_visuals(self, text: str, save_path: str = None) -> Dict:
        """Get explanation with visualizations"""
        
        # Generate comprehensive explanation
        explanation = self.explainer.generate_comprehensive_explanation(text)
        
        # Create visualizations
        visualizations = self.explainer.visualize_explanation(explanation, save_path)
        
        # Generate text explanation
        text_explanation = self.explainer.generate_text_explanation(explanation)
        
        return {
            'explanation': explanation,
            'visualizations': visualizations,
            'text_explanation': text_explanation
        }


if __name__ == "__main__":
    # Example usage
    try:
        print("Loading model...")
        
        # FIX: Initialize the EnsembleModelWrapper correctly
        model_path = "/Users/Megha/finalyearproj/Hindi-Fake-News_detection/backend/ml_flow/multilingual-fake-news-complete.tar.gz"
        ensemble_model = EnsembleModelWrapper(model_path)
        
        print("Model loaded successfully!")
        
        # Initialize explainer with the model object (not the path)
        explainer = ExplanationServer(ensemble_model)
        
        print("Explainer initialized!")
        
        # Test explanation
        sample_text = "यह एक बहुत ही संदिग्ध खबर है जो फर्जी लग रही है।"
        
        print(f"\nAnalyzing text: {sample_text}")
        
        result = explainer.get_explanation_with_visuals(sample_text, "explanation_output")
        
        print("\n" + "="*60)
        print("Explanation generated successfully!")
        print("="*60)
        print("\nText Explanation:")
        print(result['text_explanation'])
        
        if 'visualizations' in result and result['visualizations']:
            print(f"\nVisualizations saved with prefix: explanation_output")
            
    except Exception as e:
        print(f"Error in explanation demo: {e}")
        import traceback
        traceback.print_exc()

