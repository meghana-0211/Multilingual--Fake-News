# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import hashlib
import torch

from models.ml_model import LSTMBERTEnsemble
from models.preprocessor import MultilingualPreprocessor
from models.explainer import ExplainabilityEngine
from blockchain.web3_client import BlockchainClient
from database.models import db, User, Feedback
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
CORS(app)
jwt = JWTManager(app)
db.init_app(app)

# Initialize ML components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
preprocessor = MultilingualPreprocessor()
model = LSTMBERTEnsemble().to(device)

# Load trained model
checkpoint = torch.load('data/models/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Explainability
explainer = ExplainabilityEngine(model, preprocessor)

# Blockchain client
blockchain = BlockchainClient()
blockchain.load_contract('ArticleRegistry', Config.ARTICLE_REGISTRY_ADDRESS)
blockchain.load_contract('AnnotationRegistry', Config.ANNOTATION_REGISTRY_ADDRESS)
blockchain.set_account(Config.PRIVATE_KEY)

# Routes
@app.route('/api/analyze', methods=['POST'])
def analyze_article():
    """Main endpoint for fake news detection"""
    data = request.json
    text = data.get('text')
    language = data.get('language', 'hindi')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess
    encoding, normalized_text = preprocessor.preprocess_batch([text], [language])
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        
        fake_prob = probs[0][0].item()
        real_prob = probs[0][1].item()
        prediction = 'fake' if fake_prob > real_prob else 'real'
        confidence = max(fake_prob, real_prob)
    
    # Generate explanation
    explanation = explainer.explain(text, language)
    
    # Calculate content hash
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    
    # Check blockchain verification
    verification = blockchain.verify_article(content_hash)
    annotations = blockchain.get_annotations(content_hash)
    
    response = {
        'prediction': prediction,
        'confidence': float(confidence),
        'scores': {
            'fake': float(fake_prob),
            'real': float(real_prob)
        },
        'explanation': explanation,
        'contentHash': content_hash,
        'blockchain': {
            'verified': verification['exists'],
            'publisher': verification.get('publisher'),
            'timestamp': verification.get('timestamp'),
            'annotations': annotations
        }
    }
    
    return jsonify(response)

@app.route('/api/submit-feedback', methods=['POST'])
@jwt_required()
def submit_feedback():
    """Submit user feedback for model improvement"""
    current_user = get_jwt_identity()
    data = request.json
    
    feedback = Feedback(
        user_id=current_user,
        article_text=data['text'],
        predicted_label=data['predictedLabel'],
        correct_label=data['correctLabel'],
        language=data['language'],
        confidence=data.get('confidence')
    )
    
    db.session.add(feedback)
    db.session.commit()
    
    return jsonify({'message': 'Feedback submitted successfully'}), 201

@app.route('/api/register-article', methods=['POST'])
@jwt_required()
def register_article():
    """Register article on blockchain (publishers only)"""
    data = request.json
    
    content_hash = hashlib.sha256(data['text'].encode()).hexdigest()
    language = data['language']
    
    try:
        receipt = blockchain.register_article(content_hash, language)
        return jsonify({
            'success': True,
            'transactionHash': receipt.transactionHash.hex(),
            'contentHash': content_hash
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-annotation', methods=['POST'])
@jwt_required()
def add_annotation():
    """Add fact-check annotation (fact-checkers only)"""
    data = request.json
    
    content_hash = hashlib.sha256(data['text'].encode()).hexdigest()
    
    try:
        receipt = blockchain.add_annotation(
            content_hash,
            data['flagType'],
            data['ipfsHash'],
            data['confidence']
        )
        return jsonify({
            'success': True,
            'transactionHash': receipt.transactionHash.hex()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User authentication"""
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    
    if user and user.check_password(data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify({
            'accessToken': access_token,
            'user': {
                'id': user.id,
                'email': user.email,
                'role': user.role
            }
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)