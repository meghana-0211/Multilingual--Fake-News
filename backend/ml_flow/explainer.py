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
            mode='classification'
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
                'prediction_proba': explanation.predict_proba,
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
                'lstm_bert_prob': prediction_result['lstm_bert_prob'],
                'tfidf_prob': prediction_result['tfidf_prob'],
                'ensemble_prob': prediction_result['ensemble_prob']
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
            
            explanation += "Model Components:\n"
            explanation += f"  • LSTM-BERT probability: {prediction['lstm_bert_prob']}\n"
            explanation += f"  • TF-IDF probability: {prediction['tfidf_prob']}\n"
            explanation += f"  • Final ensemble probability: {prediction['ensemble_prob']}\n"
            
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
        from ensemble_model import EnsembleModel
        
        # Load trained model
        ensemble_model = EnsembleModel()
        ensemble_model.load_models("models/")
        
        # Initialize explainer
        explainer = ExplanationServer(ensemble_model)
        
        # Test explanation
        sample_text = "यह एक बहुत ही संदिग्ध खबर है जो फर्जी लग रही है।"
        
        result = explainer.get_explanation_with_visuals(sample_text, "explanation_output")
        
        print("Explanation generated successfully!")
        print("\nText Explanation:")
        print(result['text_explanation'])
        
    except Exception as e:
        print(f"Error in explanation demo: {e}")