from ml_blockchain_bridge import MLBlockchainBridge
from web3_client import hash_article

bridge = MLBlockchainBridge()

# Same article from demo
article = """
कोरोना वायरस से बचने के लिए गर्म पानी पीना काफी है।
विशेषज्ञों का दावा है कि यह वायरस गर्म पानी से मर जाता है।
"""

# Verify
result = bridge.verify_article(article)

print("="*60)
print("VERIFICATION TEST")
print("="*60)
print(f"Registered: {result['registered']}")
print(f"Annotations: {result.get('annotations_count', 0)}")

if result['registered']:
    print("\n✅ SUCCESS! Article found on blockchain!")
    print(f"Publisher: {result['publisher']}")
    print(f"Timestamp: {result['timestamp']}")
else:
    print("\n❌ FAILED! Article not found")