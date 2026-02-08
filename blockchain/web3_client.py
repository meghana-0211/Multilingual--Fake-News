"""
WEB3 CLIENT - Python ↔ Blockchain Bridge

This file connects your Python code to the blockchain.
Think: requests library but for blockchain

What it does:
- Connects to Ganache (local blockchain)
- Loads smart contracts
- Sends transactions (write to blockchain)
- Reads data (read from blockchain)
"""

from web3 import Web3
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlockchainClient:
    """
    Client to interact with blockchain contracts
    
    Think: Database client but for blockchain
    """
    
    def __init__(
        self,
        provider_url='http://127.0.0.1:8545',  # Ganache default
        publisher_registry_address=None,
        article_registry_address=None,
        annotation_registry_address=None
    ):
        """
        Initialize blockchain connection
        
        Args:
            provider_url: Where is blockchain running? (Ganache)
            *_address: Contract addresses from deployment
        """
        # Connect to blockchain
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        
        # Check connection
        if not self.web3.is_connected():
            raise Exception("Failed to connect to blockchain!")
        
        logger.info(f"✓ Connected to blockchain at {provider_url}")
        
        # Store contract addresses
        self.addresses = {
            'publisher_registry': publisher_registry_address,
            'article_registry': article_registry_address,
            'annotation_registry': annotation_registry_address
        }
        
        # Contracts will be loaded when needed
        self.contracts = {}
        
        # Default account (for sending transactions)
        self.account = None
    
    def set_account(self, private_key):
        """
        Set account for sending transactions
        
        Args:
            private_key: Private key from Ganache
        
        Example:
            # Get private key from Ganache output
            client.set_account('0xac0974bec...')
        """
        self.account = self.web3.eth.account.from_key(private_key)
        logger.info(f"✓ Account set: {self.account.address}")
    
    def load_contract(self, contract_name):
        """
        Load a smart contract
        
        How it works:
        1. Read ABI (contract interface) from build folder
        2. Get contract address
        3. Create contract object
        
        Args:
            contract_name: 'PublisherRegistry', 'ArticleRegistry', or 'AnnotationRegistry'
        
        Returns:
            Contract object
        """
        # Find ABI file
        # Truffle saves compiled contracts in build/contracts/
    # Resolve absolute path to this file's directory
        base_dir = Path(__file__).resolve().parent

        # Build path to Truffle artifacts
        abi_path = base_dir / "build" / "contracts" / f"{contract_name}.json"

        # Debug log (helps if path ever breaks again)
        logger.info(f"Looking for ABI at: {abi_path}")
        
        if not abi_path.exists():
            raise FileNotFoundError(
                f"Contract ABI not found: {abi_path}\n"
                f"Did you run 'truffle migrate'?"
            )
        
        # Load ABI (Application Binary Interface)
        # Think: API documentation for the contract
        with open(abi_path) as f:
            contract_json = json.load(f)
            abi = contract_json['abi']
        
        # Get contract address
        address_key = contract_name.lower().replace('registry', '_registry')
        address = self.addresses.get(address_key)
        
        if not address:
            raise ValueError(f"Address not set for {contract_name}")
        
        # Create contract object
        contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=abi
        )
        
        # Cache for reuse
        self.contracts[contract_name] = contract
        logger.info(f"✓ Loaded {contract_name} at {address}")
        
        return contract
    
    # ========================================
    # PUBLISHER REGISTRY FUNCTIONS
    # ========================================
    
    def register_publisher(self, publisher_address, publisher_name):
        """
        Register a new verified publisher
        
        Args:
            publisher_address: Ethereum address of publisher
            publisher_name: Name (e.g., "BBC News")
        
        Returns:
            Transaction receipt
        
        Example:
            client.register_publisher(
                '0x70997970C51812dc3A010C7d01b50e0d17dc79C8',
                'BBC News'
            )
        """
        contract = self.contracts.get('PublisherRegistry') or self.load_contract('PublisherRegistry')
        
        # Build transaction
        txn = contract.functions.registerPublisher(
            Web3.to_checksum_address(publisher_address),
            publisher_name
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.web3.eth.gas_price
        })
        
        # Sign transaction
        signed = self.web3.eth.account.sign_transaction(txn, self.account.key)
        
        # Send transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        
        # Wait for confirmation
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        logger.info(f"✓ Registered publisher: {publisher_name}")
        return receipt
    
    def is_verified_publisher(self, publisher_address):
        """
        Check if address is a verified publisher
        
        Args:
            publisher_address: Address to check
        
        Returns:
            bool: True if verified
        """
        contract = self.contracts.get('PublisherRegistry') or self.load_contract('PublisherRegistry')
        
        is_verified = contract.functions.isVerified(
            Web3.to_checksum_address(publisher_address)
        ).call()
        
        return is_verified
    
    # ========================================
    # ARTICLE REGISTRY FUNCTIONS
    # ========================================
    
    def register_article(self, content_hash, language):
        """
        Register an article on blockchain
        
        Flow:
        1. Your ML model analyzes article
        2. You call this function
        3. Article hash stored on blockchain
        4. Can prove "this article existed on this date"
        
        Args:
            content_hash: SHA-256 hash of article text (bytes32)
            language: 'hindi', 'gujarati', 'marathi', or 'telugu'
        
        Returns:
            Transaction receipt
        
        Example:
            import hashlib
            article = "कोरोना वायरस..."
            hash_bytes = hashlib.sha256(article.encode()).digest()
            client.register_article(hash_bytes, 'hindi')
        """
        contract = self.contracts.get('ArticleRegistry') or self.load_contract('ArticleRegistry')
        
        # Ensure hash is bytes32
        if isinstance(content_hash, str):
            content_hash = bytes.fromhex(content_hash.replace('0x', ''))
        
        # Build transaction
        txn = contract.functions.registerArticle(
            content_hash,
            language
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.web3.eth.gas_price
        })
        
        # Sign and send
        signed = self.web3.eth.account.sign_transaction(txn, self.account.key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        logger.info(f"✓ Article registered on blockchain")
        return receipt
    
    def verify_article(self, content_hash):
        """
        Check if article exists on blockchain
        
        Args:
            content_hash: SHA-256 hash (bytes32)
        
        Returns:
            dict with:
                - exists: bool
                - publisher: address
                - timestamp: int (Unix timestamp)
        
        Example:
            info = client.verify_article(hash_bytes)
            if info['exists']:
                print(f"Article registered on {info['timestamp']}")
        """
        contract = self.contracts.get('ArticleRegistry') or self.load_contract('ArticleRegistry')
        
        if isinstance(content_hash, str):
            content_hash = bytes.fromhex(content_hash.replace('0x', ''))
        
        exists, publisher, timestamp = contract.functions.verifyArticle(
            content_hash
        ).call()
        
        return {
            'exists': exists,
            'publisher': publisher,
            'timestamp': timestamp
        }
    
    # ========================================
    # ANNOTATION REGISTRY FUNCTIONS
    # ========================================
    
    def add_annotation(self, article_hash, flag_type, ipfs_hash, confidence):
        """
        Add fact-check annotation
        
        Args:
            article_hash: Article hash (bytes32)
            flag_type: 0=MISLEADING, 1=FALSE, 2=SATIRE, 3=UNVERIFIED, 4=CORRECT
            ipfs_hash: Link to detailed correction
            confidence: 0-100
        
        Example:
            # Your model says article is fake with 95% confidence
            client.add_annotation(
                article_hash,
                flag_type=1,  # FALSE
                ipfs_hash="",  # Can be empty for now
                confidence=95
            )
        """
        contract = self.contracts.get('AnnotationRegistry') or self.load_contract('AnnotationRegistry')
        
        if isinstance(article_hash, str):
            article_hash = bytes.fromhex(article_hash.replace('0x', ''))
        
        txn = contract.functions.addAnnotation(
            article_hash,
            flag_type,
            ipfs_hash,
            confidence
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': 300000,
            'gasPrice': self.web3.eth.gas_price
        })
        
        signed = self.web3.eth.account.sign_transaction(txn, self.account.key)
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        logger.info(f"✓ Annotation added")
        return receipt
    
    def get_annotations(self, article_hash):
        """
        Get all annotations for an article
        
        Args:
            article_hash: Article hash (bytes32)
        
        Returns:
            list of dict with annotation details
        """
        contract = self.contracts.get('AnnotationRegistry') or self.load_contract('AnnotationRegistry')
        
        if isinstance(article_hash, str):
            article_hash = bytes.fromhex(article_hash.replace('0x', ''))
        
        annotations = contract.functions.getAnnotations(article_hash).call()
        
        # Convert to readable format
        result = []
        for ann in annotations:
            result.append({
                'article_hash': ann[0].hex(),
                'fact_checker': ann[1],
                'flag_type': ann[2],
                'ipfs_hash': ann[3],
                'timestamp': ann[4],
                'confidence': ann[5]
            })
        
        return result


# ===========================================
# HELPER FUNCTIONS
# ===========================================

def hash_article(text):
    """
    Create blockchain-compatible hash of article
    
    Args:
        text: Article text (string)
    
    Returns:
        bytes32 hash
    
    Example:
        hash_bytes = hash_article("कोरोना वायरस...")
        client.register_article(hash_bytes, 'hindi')
    """
    import hashlib
    normalized = " ".join(text.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).digest()