"""
Setup Script - Register accounts as publisher and fact-checker
"""

from web3_client import BlockchainClient
from config import *

def setup_accounts():
    """Register admin account as both publisher and fact-checker"""
    
    print("="*60)
    print("BLOCKCHAIN ACCOUNT SETUP")
    print("="*60)
    
    # Connect
    client = BlockchainClient(
        provider_url=BLOCKCHAIN_URL,
        publisher_registry_address=PUBLISHER_REGISTRY,
        article_registry_address=ARTICLE_REGISTRY,
        annotation_registry_address=ANNOTATION_REGISTRY
    )
    
    client.set_account(PRIVATE_KEY)
    
    print(f"\n✓ Connected to blockchain")
    print(f"✓ Using account: {client.account.address}")
    
    # Load contracts
    pub_contract = client.load_contract('PublisherRegistry')
    ann_contract = client.load_contract('AnnotationRegistry')
    
    # Check who is admin
    pub_admin = pub_contract.functions.admin().call()
    ann_admin = ann_contract.functions.admin().call()
    
    print(f"\nPublisherRegistry admin: {pub_admin}")
    print(f"AnnotationRegistry admin: {ann_admin}")
    print(f"Your account: {client.account.address}")
    
    if pub_admin.lower() != client.account.address.lower():
        print("\n⚠️  WARNING: You're not the admin!")
        print("You need to use the account that deployed the contracts")
        return
    
    # Register as publisher
    print("\n1️⃣  Registering as publisher...")
    try:
        receipt = client.register_publisher(
            client.account.address,
            "ML System Publisher"
        )
        print("   ✓ Registered as publisher")
    except Exception as e:
        if "already registered" in str(e).lower():
            print("   ✓ Already registered as publisher")
        else:
            print(f"   ✗ Error: {e}")
            raise
    
    # Register as fact-checker
    print("\n2️⃣  Registering as fact-checker...")
    try:
        txn = ann_contract.functions.verifyFactChecker(
            client.web3.to_checksum_address(client.account.address)
        ).build_transaction({
            'from': client.account.address,
            'nonce': client.web3.eth.get_transaction_count(client.account.address),
            'gas': 200000,
            'gasPrice': client.web3.eth.gas_price
        })
        
        signed = client.web3.eth.account.sign_transaction(txn, client.account.key)
        tx_hash = client.web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = client.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        print("   ✓ Registered as fact-checker")
    except Exception as e:
        if "already" in str(e).lower():
            print("   ✓ Already registered as fact-checker")
        else:
            print(f"   ✗ Error: {e}")
            raise
    
    # Verify registrations
    print("\n3️⃣  Verifying registrations...")
    
    is_verified = pub_contract.functions.isVerified(
        client.web3.to_checksum_address(client.account.address)
    ).call()
    print(f"   Publisher verified: {is_verified}")
    
    is_fact_checker = ann_contract.functions.verifiedFactCheckers(
        client.web3.to_checksum_address(client.account.address)
    ).call()
    print(f"   Fact-checker verified: {is_fact_checker}")
    
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60)
    print("\nYou can now run ml_blockchain_bridge.py")
    

if __name__ == '__main__':
    setup_accounts()