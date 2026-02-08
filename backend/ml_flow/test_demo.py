"""
WORKING TEST SCRIPT
Fixed to handle your trained model checkpoint
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your model
from train import IndicBERTBiLSTMEnsemble


class ModelWrapper:
    """Fixed wrapper that handles checkpoint compatibility"""
    
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
        logger.info("✓ Tokenizer loaded")
        
        # Initialize model
        self.model = IndicBERTBiLSTMEnsemble()
        
        # Load checkpoint with compatibility mode
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load state dict (strict=False ignores missing keys like ensemble_weights)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("✓ Model weights loaded (compatibility mode)")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("✓ Model ready for inference")
    
    def predict_single(self, text: str) -> Dict:
        """Predict on single text"""
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
        
        return {
            'prediction': prediction,
            'label': 'Fake' if prediction == 1 else 'Real',
            'confidence': float(probs[prediction]),
            'probabilities': {
                'real': float(probs[0]),
                'fake': float(probs[1])
            }
        }
    
    def predict_batch(self, texts: List[str]) -> np.ndarray:
        """Predict on batch (for LIME)"""
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


class SimpleExplainer:
    """Simplified explainer using only LIME"""
    
    def __init__(self, model_wrapper):
        self.model = model_wrapper
        self.lime_explainer = LimeTextExplainer(
            class_names=['Real', 'Fake'],
            split_expression=r'\s+'
        )
        logger.info("✓ Explainer initialized")
    
    def explain(self, text: str, num_features: int = 10) -> Dict:
        """Generate LIME explanation"""
        logger.info("Generating explanation...")
        
        # Get prediction
        pred = self.model.predict_single(text)
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            text,
            self.model.predict_batch,
            num_features=num_features,
            num_samples=500  # Reduced for speed
        )
        
        # Extract features
        features = explanation.as_list()
        
        return {
            'prediction': pred,
            'features': features,
            'text': text
        }
    
    def visualize(self, explanation: Dict, save_path: str = None):
        """Create simple bar plot"""
        features = explanation['features'][:15]
        
        if not features:
            logger.warning("No features to visualize")
            return
        
        words, scores = zip(*features)
        colors = ['green' if s > 0 else 'red' for s in scores]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(words)), scores, color=colors, alpha=0.7)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Importance Score')
        plt.title(f'Top Influential Words - Predicted: {explanation["prediction"]["label"]}')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved visualization: {save_path}")
        
        return plt.gcf()
    
    def create_report(self, explanation: Dict, save_path: str = 'report.html'):
        """Generate HTML report"""
        pred = explanation['prediction']
        features = explanation['features'][:10]
        text = explanation['text']
        
        # Highlight important words
        important_words = set([w for w, _ in features])
        words = text.split()
        highlighted = []
        for word in words:
            if word in important_words:
                highlighted.append(f'<mark style="background-color: #ffeb3b;">{word}</mark>')
            else:
                highlighted.append(word)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fake News Analysis</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; 
                     border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
        .prediction {{ font-size: 24px; font-weight: bold; padding: 20px; 
                      border-radius: 8px; margin: 20px 0; }}
        .fake {{ background: #ffebee; color: #c62828; border-left: 5px solid #c62828; }}
        .real {{ background: #e8f5e9; color: #2e7d32; border-left: 5px solid #2e7d32; }}
        .confidence {{ font-size: 18px; margin: 10px 0; color: #666; }}
        .text-box {{ background: #fafafa; padding: 20px; border-radius: 8px; 
                    margin: 20px 0; line-height: 1.8; border: 1px solid #ddd; }}
        mark {{ background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f0f0f0; font-weight: bold; }}
        .section {{ margin: 30px 0; }}
        .section-title {{ font-size: 20px; font-weight: bold; color: #555; 
                         margin-bottom: 15px; border-left: 4px solid #2196F3; padding-left: 10px; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; 
                  color: #888; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Fake News Detection Report</h1>
        
        <div class="prediction {'fake' if pred['prediction'] == 1 else 'real'}">
            Prediction: {pred['label']}
        </div>
        
        <div class="confidence">
            Confidence: {pred['confidence']:.1%} • 
            Fake: {pred['probabilities']['fake']:.1%} • 
            Real: {pred['probabilities']['real']:.1%}
        </div>
        
        <div class="section">
            <div class="section-title">📄 Article Text (Important Words Highlighted)</div>
            <div class="text-box">
                {' '.join(highlighted)}
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">💡 Top 10 Influential Words</div>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Word/Phrase</th>
                    <th>Impact</th>
                    <th>Direction</th>
                </tr>
                {''.join([
                    f'''<tr>
                        <td>{i+1}</td>
                        <td><strong>{word}</strong></td>
                        <td class="{'positive' if score > 0 else 'negative'}">{score:+.4f}</td>
                        <td>{'Supports' if score > 0 else 'Opposes'} prediction</td>
                    </tr>'''
                    for i, (word, score) in enumerate(features)
                ])}
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Interpretation</div>
            <p><strong>What do these scores mean?</strong></p>
            <ul>
                <li><strong>Positive scores (+):</strong> Words that support the model's prediction</li>
                <li><strong>Negative scores (-):</strong> Words that oppose the model's prediction</li>
                <li><strong>Higher magnitude:</strong> Stronger influence on the decision</li>
            </ul>
        </div>
        
        <div class="footer">
            Generated by Multilingual Fake News Detection System<br>
            Model: IndicBERT + BiLSTM with Attention | Method: LIME Explainability
        </div>
    </div>
</body>
</html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"✓ Saved HTML report: {save_path}")
        return save_path


def main():
    """Main demo script"""
    
    print("\n" + "="*60)
    print("MULTILINGUAL FAKE NEWS DETECTION - DEMO")
    print("="*60)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = ModelWrapper(
        model_path='/Users/Megha/finalyearproj/Hindi-Fake-News_detection/backend/ml_flow/best_model.pth',
        device='cpu'
    )
    
    # Create explainer
    print("\n[2/4] Initializing explainer...")
    explainer = SimpleExplainer(model)
    
    # Test examples
    examples = {
        'Hindi Fake': """
कोरोना वायरस से बचने के लिए गर्म पानी पीना काफी है। 
विशेषज्ञों का दावा है कि यह वायरस गर्म पानी से मर जाता है।
यह महत्वपूर्ण जानकारी व्हाट्सएप पर वायरल हो रही है।
        """.strip(),
        
        'Hindi Real': """
भारत सरकार ने आज नई शिक्षा नीति 2020 की घोषणा की।
शिक्षा मंत्री ने प्रेस कॉन्फ्रेंस में इसके मुख्य बिंदुओं की जानकारी दी।
        """.strip()
    }
    
    print("\n[3/4] Analyzing examples...\n")
    
    for i, (name, text) in enumerate(examples.items(), 1):
        print(f"\n--- Example {i}: {name} ---")
        print(f"Text: {text[:80]}...")
        
        # Get explanation
        result = explainer.explain(text)
        
        # Print results
        print(f"\n✓ Prediction: {result['prediction']['label']}")
        print(f"✓ Confidence: {result['prediction']['confidence']:.1%}")
        
        print(f"\nTop 5 Influential Words:")
        for word, score in result['features'][:5]:
            print(f"  • {word:20} {score:+.4f}")
        
        # Save outputs
        output_prefix = f"output_{name.lower().replace(' ', '_')}"
        explainer.visualize(result, f"{output_prefix}.png")
        explainer.create_report(result, f"{output_prefix}.html")
    
    print("\n[4/4] Complete!")
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • output_hindi_fake.png")
    print("  • output_hindi_fake.html")
    print("  • output_hindi_real.png")
    print("  • output_hindi_real.html")
    print("\nOpen the .html files in your browser to see full reports!")


if __name__ == '__main__':
    main()