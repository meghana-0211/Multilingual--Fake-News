"""
EXPLAINABILITY MODULE
LIME + Attention-based explanations for fake news predictions

Author: Your Name
Methods: LIME (Local Interpretable Model-agnostic Explanations) + Attention Visualization
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper for trained model to use with explainability tools"""
    
    def __init__(self, model_path: str, model_class, device=None):
        """
        Args:
            model_path: Path to saved model checkpoint (.pth file)
            model_class: The model class (IndicBERTBiLSTMEnsemble)
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
        
        # Initialize model
        self.model = model_class()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✓ Model loaded successfully")
    
    def predict_single(self, text: str, return_all_outputs=False) -> Dict:
        """
        Predict on single text
        
        Returns:
            dict with prediction, confidence, probabilities
        """
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs['logits'], dim=1)[0]
            prediction = torch.argmax(probs).item()
        
        result = {
            'prediction': prediction,
            'label': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(probs[prediction]),
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            }
        }
        
        if return_all_outputs:
            result['attention_weights'] = outputs['attention_weights'].cpu()
            result['bert_logits'] = outputs['bert_logits'].cpu()
            result['lstm_logits'] = outputs['lstm_logits'].cpu()
        
        return result
    
    def predict_batch(self, texts: List[str]) -> np.ndarray:
        """
        Predict on batch of texts (for LIME)
        
        Returns:
            numpy array of shape [batch, 2] with probabilities
        """
        encodings = self.tokenizer(
            texts,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs['logits'], dim=1)
        
        return probs.cpu().numpy()
    
    def get_attention_weights(self, text: str) -> Dict:
        """Get attention weights for visualization"""
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Get attention weights (averaged across heads)
        attn_weights = outputs['attention_weights'].mean(dim=1)[0]  # [seq_len, seq_len]
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get actual token positions (non-padding)
        actual_length = attention_mask[0].sum().item()
        tokens = tokens[:actual_length]
        attn_weights = attn_weights[:actual_length, :actual_length].cpu().numpy()
        
        return {
            'tokens': tokens,
            'attention_weights': attn_weights,
            'prediction': torch.argmax(torch.softmax(outputs['logits'], dim=1)[0]).item()
        }


class ExplainabilityEngine:
    """Main explainability engine using LIME and attention"""
    
    def __init__(self, model_wrapper: ModelWrapper):
        self.model = model_wrapper
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=['Real', 'Fake'],
            feature_selection='auto',
            split_expression=r'\s+',
            bow=True
        )
        
        logger.info("✓ Explainability engine initialized")
    
    def explain_lime(self, text: str, num_features: int = 10, num_samples: int = 1000) -> Dict:
        """
        Generate LIME explanation
        
        Args:
            text: Input text to explain
            num_features: Number of top features to show
            num_samples: Number of samples for LIME (higher = more accurate but slower)
        
        Returns:
            Dictionary with explanation details
        """
        logger.info("Generating LIME explanation...")
        
        # Get base prediction
        prediction_result = self.model.predict_single(text)
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            text,
            self.model.predict_batch,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract feature importance
        feature_importance = explanation.as_list()
        
        return {
            'prediction': prediction_result,
            'feature_importance': feature_importance,
            'lime_score': explanation.score,
            'intercept': explanation.intercept
        }
    
    def explain_attention(self, text: str, top_k: int = 15) -> Dict:
        """
        Generate attention-based explanation
        
        Args:
            text: Input text to explain
            top_k: Number of top attended tokens to return
        
        Returns:
            Dictionary with attention explanation
        """
        logger.info("Generating attention explanation...")
        
        attention_data = self.model.get_attention_weights(text)
        
        # Calculate importance: average attention received by each token
        attn_matrix = attention_data['attention_weights']
        token_importance = attn_matrix.mean(axis=0)  # Average across all source positions
        
        # Combine tokens with their importance
        tokens = attention_data['tokens']
        
        # Process tokens (merge subwords, remove special tokens)
        processed_tokens = []
        processed_importance = []
        
        current_word = ""
        current_importance = 0.0
        word_count = 0
        
        for i, (token, importance) in enumerate(zip(tokens, token_importance)):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Handle subword tokens (##)
            if token.startswith('##'):
                current_word += token[2:]
                current_importance += importance
                word_count += 1
            else:
                # Save previous word
                if current_word:
                    processed_tokens.append(current_word)
                    processed_importance.append(current_importance / word_count if word_count > 0 else current_importance)
                
                # Start new word
                current_word = token
                current_importance = importance
                word_count = 1
        
        # Add last word
        if current_word:
            processed_tokens.append(current_word)
            processed_importance.append(current_importance / word_count if word_count > 0 else current_importance)
        
        # Sort by importance
        token_importance_pairs = list(zip(processed_tokens, processed_importance))
        token_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'top_tokens': token_importance_pairs[:top_k],
            'all_tokens': token_importance_pairs,
            'attention_matrix': attn_matrix,
            'prediction': attention_data['prediction']
        }
    
    def explain_comprehensive(self, text: str) -> Dict:
        """
        Generate comprehensive explanation using both LIME and attention
        
        Args:
            text: Input text to explain
        
        Returns:
            Complete explanation with visualizations
        """
        logger.info(f"Generating comprehensive explanation for text: {text[:100]}...")
        
        # Get both explanations
        lime_exp = self.explain_lime(text)
        attention_exp = self.explain_attention(text)
        
        # Combine influential words from both methods
        combined_importance = {}
        
        # Add LIME features
        for word, score in lime_exp['feature_importance']:
            combined_importance[word] = combined_importance.get(word, 0) + abs(score)
        
        # Add attention features (with lower weight)
        for word, score in attention_exp['top_tokens']:
            combined_importance[word] = combined_importance.get(word, 0) + abs(score) * 0.3
        
        # Sort combined
        top_combined = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return {
            'prediction': lime_exp['prediction'],
            'lime_explanation': lime_exp,
            'attention_explanation': attention_exp,
            'combined_top_words': top_combined,
            'text': text
        }
    
    def visualize_lime(self, lime_explanation: Dict, save_path: str = None):
        """Create bar plot of LIME feature importance"""
        features = lime_explanation['feature_importance'][:15]
        
        if not features:
            logger.warning("No features to visualize")
            return
        
        words, scores = zip(*features)
        colors = ['green' if score > 0 else 'red' for score in scores]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(words)), scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('Feature Importance (positive = supports prediction)')
        ax.set_title(f'LIME Explanation - Predicted: {lime_explanation["prediction"]["label"]}')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved LIME visualization: {save_path}")
        
        return fig
    
    def visualize_attention(self, attention_explanation: Dict, save_path: str = None):
        """Create heatmap of attention weights"""
        attn_matrix = attention_explanation['attention_matrix']
        tokens = [item[0] for item in attention_explanation['all_tokens'][:30]]
        
        # Limit to first 30 tokens for readability
        attn_subset = attn_matrix[:30, :30]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            attn_subset,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Reds',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax
        )
        ax.set_title('Attention Weights Heatmap')
        ax.set_xlabel('Attended Tokens')
        ax.set_ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved attention visualization: {save_path}")
        
        return fig
    
    def generate_html_report(self, explanation: Dict, save_path: str = 'explanation.html'):
        """Generate interactive HTML report"""
        
        pred = explanation['prediction']
        lime_features = explanation['lime_explanation']['feature_importance'][:10]
        attn_features = explanation['attention_explanation']['top_tokens'][:10]
        combined_features = explanation['combined_top_words'][:10]
        
        # Highlight text
        text = explanation['text']
        highlighted_words = set([word for word, _ in combined_features])
        
        # Simple word highlighting
        words = text.split()
        highlighted_text = []
        for word in words:
            if word in highlighted_words:
                highlighted_text.append(f'<mark style="background-color: #ffeb3b;">{word}</mark>')
            else:
                highlighted_text.append(word)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fake News Explanation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                .prediction {{ font-size: 24px; font-weight: bold; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .fake {{ background: #ffebee; color: #c62828; }}
                .real {{ background: #e8f5e9; color: #2e7d32; }}
                .confidence {{ font-size: 18px; margin: 10px 0; }}
                .text-box {{ background: #fafafa; padding: 20px; border-radius: 8px; margin: 20px 0; line-height: 1.8; }}
                mark {{ background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f0f0f0; font-weight: bold; }}
                .section {{ margin: 30px 0; }}
                .section-title {{ font-size: 20px; font-weight: bold; color: #555; margin-bottom: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Fake News Detection Explanation</h1>
                
                <div class="prediction {'fake' if pred['prediction'] == 1 else 'real'}">
                    Prediction: {pred['label']}
                </div>
                
                <div class="confidence">
                    Confidence: {pred['confidence']:.2%}
                </div>
                
                <div class="section">
                    <div class="section-title">📄 Article Text (Key Words Highlighted)</div>
                    <div class="text-box">
                        {' '.join(highlighted_text)}
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">💡 Top Influential Words (Combined LIME + Attention)</div>
                    <table>
                        <tr><th>Word</th><th>Importance Score</th></tr>
                        {''.join([f'<tr><td>{word}</td><td>{score:.4f}</td></tr>' for word, score in combined_features])}
                    </table>
                </div>
                
                <div class="section">
                    <div class="section-title">🔬 LIME Feature Analysis</div>
                    <table>
                        <tr><th>Word/Phrase</th><th>Impact</th></tr>
                        {''.join([f'<tr><td>{word}</td><td style="color: {"green" if score > 0 else "red"};">{score:+.4f}</td></tr>' for word, score in lime_features])}
                    </table>
                </div>
                
                <div class="section">
                    <div class="section-title">👁️ Attention-Based Analysis</div>
                    <table>
                        <tr><th>Token</th><th>Attention Weight</th></tr>
                        {''.join([f'<tr><td>{word}</td><td>{score:.4f}</td></tr>' for word, score in attn_features])}
                    </table>
                </div>
                
                <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #888; font-size: 12px;">
                    Generated by Multilingual Fake News Detection System
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"✓ Saved HTML report: {save_path}")
        
        return save_path


# Example usage
if __name__ == '__main__':
    # This will be used after training
    # For now, just test the structure
    
    print("Explainability module ready!")
    print("\nTo use after training:")
    print("1. Load your trained model")
    print("2. Create ModelWrapper with your model")
    print("3. Create ExplainabilityEngine")
    print("4. Call explain_comprehensive(text)")