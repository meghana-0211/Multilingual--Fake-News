"""
Bridge between ML Model and Blockchain

This connects your fake news detector to blockchain
"""

import hashlib
from web3_client import BlockchainClient, hash_article
from config import *


class MLBlockchainBridge:
    """
    Connects ML predictions to blockchain
    
    When your model detects fake news:
    1. Hash the article
    2. Register on blockchain
    3. Add annotation with confidence
    4. Everything is now permanent!
    """
    
    def __init__(self):
        """Initialize blockchain connection"""
        
        # Connect to blockchain
        self.client = BlockchainClient(
            provider_url=BLOCKCHAIN_URL,
            publisher_registry_address=PUBLISHER_REGISTRY,
            article_registry_address=ARTICLE_REGISTRY,
            annotation_registry_address=ANNOTATION_REGISTRY
        )
        
        # Set account (for sending transactions)
        self.client.set_account(PRIVATE_KEY)
        
        print("✓ Connected to blockchain")
        print(f"✓ Using account: {self.client.account.address}")
    
    def register_prediction(self, article_text, prediction, confidence, language='hindi'):
        """
        Record ML prediction on blockchain
        
        Args:
            article_text: The article text
            prediction: 0 (real) or 1 (fake)
            confidence: 0.0 to 1.0
            language: 'hindi', 'gujarati', 'marathi', or 'telugu'
        
        Returns:
            dict with transaction details
        
        Example:
            bridge = MLBlockchainBridge()
            result = bridge.register_prediction(
                article_text="कोरोना वायरस...",
                prediction=1,  # fake
                confidence=0.94,
                language='hindi'
            )
            print(f"Registered! Hash: {result['article_hash']}")
        """
        
        # Step 1: Hash the article
        article_hash = hash_article(article_text)
        hash_hex = article_hash.hex()
        
        print(f"\n📝 Processing article...")
        print(f"   Hash: {hash_hex[:16]}...")
        print(f"   Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Step 2: Register article
        try:
            receipt1 = self.client.register_article(article_hash, language)
            print(f"   ✓ Article registered on blockchain")
        except Exception as e:
            if "already registered" in str(e).lower():
                print(f"   ⚠ Article already exists (that's ok)")
            else:
                raise
        
        # Step 3: Add annotation with ML prediction
        # Map prediction to flag type:
        # 0 (real) → CORRECT (4)
        # 1 (fake) → FALSE (1)
        flag_type = 1 if prediction == 1 else 4
        
        # Convert confidence to 0-100 scale
        confidence_int = int(confidence * 100)
        
        receipt2 = self.client.add_annotation(
            article_hash,
            flag_type=flag_type,
            ipfs_hash="",  # Can add IPFS link later
            confidence=confidence_int
        )
        
        print(f"   ✓ Prediction recorded on blockchain")
        
        return {
            'article_hash': hash_hex,
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'blockchain_verified': True,
            'transaction_hash': receipt2['transactionHash'].hex()
        }
    
    def verify_article(self, article_text):
        """
        Check if article is on blockchain
        
        Args:
            article_text: Article to check
        
        Returns:
            dict with verification details
        """
        article_hash = hash_article(article_text)
        
        # Check if registered
        info = self.client.verify_article(article_hash)
        
        if not info['exists']:
            return {
                'registered': False,
                'message': 'Article not found on blockchain'
            }
        
        # Get annotations
        annotations = self.client.get_annotations(article_hash)
        
        return {
            'registered': True,
            'publisher': info['publisher'],
            'timestamp': info['timestamp'],
            'annotations_count': len(annotations),
            'annotations': annotations
        }


# ===========================================
# SIMPLE USAGE EXAMPLE
# ===========================================

def demo():
    """
    Demo: How to use with your ML model
    """
    
    print("="*60)
    print("BLOCKCHAIN INTEGRATION DEMO")
    print("="*60)
    
    # Initialize bridge
    bridge = MLBlockchainBridge()
    
    # Example article
    article = """
    कोरोना वायरस से बचने के लिए गर्म पानी पीना काफी है।
    विशेषज्ञों का दावा है कि यह वायरस गर्म पानी से मर जाता है।
    """
    
    # Simulate ML prediction
    prediction = 1  # fake
    confidence = 0.94
    
    # Register on blockchain
    result = bridge.register_prediction(
        article_text=article,
        prediction=prediction,
        confidence=confidence,
        language='hindi'
    )
    
    print("\n" + "="*60)
    print("✓ SUCCESS!")
    print("="*60)
    print(f"Article hash: {result['article_hash']}")
    print(f"Transaction: {result['transaction_hash']}")
    print("\nThis prediction is now permanent on blockchain!")
    print("It can never be deleted or changed!")
    
    # Verify it worked
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    verification = bridge.verify_article(article)

    print(f"Registered: {verification['registered']}")

    if verification['registered']:
        print(f"Annotations: {verification['annotations_count']}")
    else:
        print("No annotations found — article may not be registered yet.")


if __name__ == '__main__':
    demo()