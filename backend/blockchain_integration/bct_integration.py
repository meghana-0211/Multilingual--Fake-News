import hashlib
import json
from typing import Dict, List, Optional, Tuple
from web3 import Web3, HTTPProvider
from eth_account import Account
import ipfshttpclient
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ArticleRecord:
    hash: str
    publisher: str
    timestamp: int
    verified: bool


@dataclass
class AnnotationRecord:
    article_hash: str
    annotator: str
    annotation_type: str  # 'flag', 'correction', 'verification'
    ipfs_hash: str
    timestamp: int


class SmartContractManager:
    def __init__(self, web3_provider_url: str, contract_addresses: Dict[str, str], 
                 private_key: str = None):
        self.w3 = Web3(HTTPProvider(web3_provider_url))
        self.contract_addresses = contract_addresses
        self.private_key = private_key
        
        if private_key:
            self.account = Account.from_key(private_key)
            self.w3.eth.default_account = self.account.address
        
        # Load contract ABIs and create contract instances
        self.contracts = {}
        self._initialize_contracts()
    
    def _initialize_contracts(self):
        """Initialize smart contract instances"""
        # Publisher Registry ABI
        publisher_abi = [
            {
                "inputs": [],
                "name": "getPublisherCount",
                "outputs": [{"type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"type": "address"}],
                "name": "isVerifiedPublisher",
                "outputs": [{"type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"type": "address"}, {"type": "string"}],
                "name": "addPublisher",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"type": "address"}],
                "name": "getPublisherInfo",
                "outputs": [{"type": "string"}, {"type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Article Registry ABI
        article_abi = [
            {
                "inputs": [{"type": "string"}, {"type": "string"}],
                "name": "registerArticle",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"type": "string"}],
                "name": "getArticleInfo",
                "outputs": [{"type": "address"}, {"type": "uint256"}, {"type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"type": "string"}],
                "name": "isArticleRegistered",
                "outputs": [{"type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Annotation Registry ABI
        annotation_abi = [
            {
                "inputs": [{"type": "string"}, {"type": "string"}, {"type": "string"}],
                "name": "addAnnotation",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"type": "string"}],
                "name": "getAnnotations",
                "outputs": [{"type": "tuple[]", "components": [
                    {"type": "address", "name": "annotator"},
                    {"type": "string", "name": "annotationType"},
                    {"type": "string", "name": "ipfsHash"},
                    {"type": "uint256", "name": "timestamp"}
                ]}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"type": "string"}],
                "name": "getAnnotationCount",
                "outputs": [{"type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Create contract instances
        try:
            self.contracts['publisher'] = self.w3.eth.contract(
                address=self.contract_addresses['publisher'],
                abi=publisher_abi
            )
            
            self.contracts['article'] = self.w3.eth.contract(
                address=self.contract_addresses['article'],
                abi=article_abi
            )
            
            self.contracts['annotation'] = self.w3.eth.contract(
                address=self.contract_addresses['annotation'],
                abi=annotation_abi
            )
            
            logger.info("Smart contracts initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing contracts: {e}")
    
    def register_article(self, article_hash: str, publisher_info: str) -> Tuple[bool, str]:
        """Register an article on the blockchain"""
        try:
            if not self.private_key:
                return False, "Private key required for transactions"
            
            # Build transaction
            function = self.contracts['article'].functions.registerArticle(article_hash, publisher_info)
            
            # Estimate gas
            gas_estimate = function.estimateGas({'from': self.account.address})
            
            # Build transaction
            transaction = function.buildTransaction({
                'from': self.account.address,
                'gas': gas_estimate,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if tx_receipt.status == 1:
                return True, tx_hash.hex()
            else:
                return False, "Transaction failed"
                
        except Exception as e:
            logger.error(f"Error adding annotation: {e}")
            return False, str(e)
    
    def get_annotations(self, article_hash: str) -> List[AnnotationRecord]:
        """Get all annotations for an article"""
        try:
            annotations = self.contracts['annotation'].functions.getAnnotations(article_hash).call()
            
            result = []
            for annotation in annotations:
                result.append(AnnotationRecord(
                    article_hash=article_hash,
                    annotator=annotation[0],
                    annotation_type=annotation[1],
                    ipfs_hash=annotation[2],
                    timestamp=annotation[3]
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting annotations: {e}")
            return []
    
    def is_verified_publisher(self, publisher_address: str) -> bool:
        """Check if a publisher is verified"""
        try:
            return self.contracts['publisher'].functions.isVerifiedPublisher(publisher_address).call()
        except Exception as e:
            logger.error(f"Error checking publisher verification: {e}")
            return False


class IPFSManager:
    def __init__(self, ipfs_api_url: str = '/ip4/127.0.0.1/tcp/5001'):
        try:
            self.client = ipfshttpclient.connect(ipfs_api_url)
            logger.info("Connected to IPFS")
        except Exception as e:
            logger.error(f"Error connecting to IPFS: {e}")
            self.client = None
    
    def store_data(self, data: Dict) -> Optional[str]:
        """Store data in IPFS and return hash"""
        if not self.client:
            return None
        
        try:
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            result = self.client.add_json(data)
            logger.info(f"Data stored in IPFS with hash: {result}")
            return result
        except Exception as e:
            logger.error(f"Error storing data in IPFS: {e}")
            return None
    
    def retrieve_data(self, ipfs_hash: str) -> Optional[Dict]:
        """Retrieve data from IPFS"""
        if not self.client:
            return None
        
        try:
            data = self.client.get_json(ipfs_hash)
            return data
        except Exception as e:
            logger.error(f"Error retrieving data from IPFS: {e}")
            return None
    
    def store_file(self, file_path: str) -> Optional[str]:
        """Store a file in IPFS"""
        if not self.client:
            return None
        
        try:
            result = self.client.add(file_path)
            return result['Hash']
        except Exception as e:
            logger.error(f"Error storing file in IPFS: {e}")
            return None


class ArticleHasher:
    @staticmethod
    def calculate_hash(text: str) -> str:
        """Calculate SHA-256 hash of article text"""
        # Normalize text before hashing
        normalized_text = text.strip().lower()
        # Remove extra whitespace
        normalized_text = ' '.join(normalized_text.split())
        
        # Calculate hash
        hash_object = hashlib.sha256(normalized_text.encode('utf-8'))
        return hash_object.hexdigest()
    
    @staticmethod
    def verify_hash(text: str, expected_hash: str) -> bool:
        """Verify if text matches the expected hash"""
        calculated_hash = ArticleHasher.calculate_hash(text)
        return calculated_hash == expected_hash


class BlockchainVerificationService:
    def __init__(self, web3_provider_url: str, contract_addresses: Dict[str, str], 
                 ipfs_api_url: str = '/ip4/127.0.0.1/tcp/5001', private_key: str = None):
        
        self.contract_manager = SmartContractManager(web3_provider_url, contract_addresses, private_key)
        self.ipfs_manager = IPFSManager(ipfs_api_url)
        self.hasher = ArticleHasher()
    
    def verify_article(self, text: str) -> Dict:
        """Comprehensive article verification"""
        
        # Calculate article hash
        article_hash = self.hasher.calculate_hash(text)
        
        # Get article info from blockchain
        article_info = self.contract_manager.get_article_info(article_hash)
        
        # Get annotations
        annotations = self.contract_manager.get_annotations(article_hash)
        
        # Process annotations
        processed_annotations = []
        for annotation in annotations:
            annotation_data = self.ipfs_manager.retrieve_data(annotation.ipfs_hash)
            
            processed_annotations.append({
                'annotator': annotation.annotator,
                'type': annotation.annotation_type,
                'timestamp': datetime.fromtimestamp(annotation.timestamp).isoformat(),
                'data': annotation_data,
                'ipfs_hash': annotation.ipfs_hash
            })
        
        verification_result = {
            'article_hash': article_hash,
            'verified_source': False,
            'publisher_info': None,
            'registration_timestamp': None,
            'annotations': processed_annotations,
            'annotation_count': len(annotations),
            'flags': [],
            'corrections': [],
            'verifications': []
        }
        
        # Process article info
        if article_info:
            verification_result['verified_source'] = True
            verification_result['publisher_address'] = article_info.publisher
            verification_result['registration_timestamp'] = datetime.fromtimestamp(article_info.timestamp).isoformat()
            
            # Check if publisher is verified
            is_verified = self.contract_manager.is_verified_publisher(article_info.publisher)
            verification_result['publisher_verified'] = is_verified
        
        # Categorize annotations
        for annotation in processed_annotations:
            if annotation['type'] == 'flag':
                verification_result['flags'].append(annotation)
            elif annotation['type'] == 'correction':
                verification_result['corrections'].append(annotation)
            elif annotation['type'] == 'verification':
                verification_result['verifications'].append(annotation)
        
        return verification_result
    
    def submit_flag(self, article_text: str, flag_data: Dict, annotator_address: str = None) -> Tuple[bool, str]:
        """Submit a flag for an article"""
        
        if not annotator_address:
            annotator_address = self.contract_manager.account.address if self.contract_manager.account else None
        
        if not annotator_address:
            return False, "Annotator address required"
        
        # Calculate article hash
        article_hash = self.hasher.calculate_hash(article_text)
        
        # Prepare flag data
        flag_info = {
            'flag_type': flag_data.get('flag_type', 'general'),
            'description': flag_data.get('description', ''),
            'evidence': flag_data.get('evidence', ''),
            'annotator': annotator_address,
            'timestamp': datetime.now().isoformat(),
            'article_hash': article_hash
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs_manager.store_data(flag_info)
        if not ipfs_hash:
            return False, "Failed to store flag data in IPFS"
        
        # Add annotation to blockchain
        success, tx_hash = self.contract_manager.add_annotation(article_hash, 'flag', ipfs_hash)
        
        if success:
            return True, f"Flag submitted successfully. Transaction: {tx_hash}"
        else:
            return False, f"Failed to submit flag: {tx_hash}"
    
    def submit_correction(self, article_text: str, correction_data: Dict, annotator_address: str = None) -> Tuple[bool, str]:
        """Submit a correction for an article"""
        
        if not annotator_address:
            annotator_address = self.contract_manager.account.address if self.contract_manager.account else None
        
        if not annotator_address:
            return False, "Annotator address required"
        
        # Calculate article hash
        article_hash = self.hasher.calculate_hash(article_text)
        
        # Prepare correction data
        correction_info = {
            'correction_type': correction_data.get('correction_type', 'factual'),
            'original_claim': correction_data.get('original_claim', ''),
            'corrected_information': correction_data.get('corrected_information', ''),
            'sources': correction_data.get('sources', []),
            'annotator': annotator_address,
            'timestamp': datetime.now().isoformat(),
            'article_hash': article_hash
        }
        
        # Store in IPFS
        ipfs_hash = self.ipfs_manager.store_data(correction_info)
        if not ipfs_hash:
            return False, "Failed to store correction data in IPFS"
        
        # Add annotation to blockchain
        success, tx_hash = self.contract_manager.add_annotation(article_hash, 'correction', ipfs_hash)
        
        if success:
            return True, f"Correction submitted successfully. Transaction: {tx_hash}"
        else:
            return False, f"Failed to submit correction: {tx_hash}"
    
    def register_publisher_article(self, article_text: str, publisher_info: str) -> Tuple[bool, str]:
        """Register an article as coming from a verified publisher"""
        
        # Calculate article hash
        article_hash = self.hasher.calculate_hash(article_text)
        
        # Register on blockchain
        success, tx_hash = self.contract_manager.register_article(article_hash, publisher_info)
        
        if success:
            return True, f"Article registered successfully. Transaction: {tx_hash}"
        else:
            return False, f"Failed to register article: {tx_hash}"


class BlockchainAPI:
    """API wrapper for blockchain operations"""
    
    def __init__(self, config: Dict):
        self.verification_service = BlockchainVerificationService(
            web3_provider_url=config['web3_provider_url'],
            contract_addresses=config['contract_addresses'],
            ipfs_api_url=config.get('ipfs_api_url', '/ip4/127.0.0.1/tcp/5001'),
            private_key=config.get('private_key')
        )
    
    def verify_article_endpoint(self, text: str) -> Dict:
        """API endpoint for article verification"""
        try:
            result = self.verification_service.verify_article(text)
            result['success'] = True
            return result
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                'success': False,
                'error': str(e),
                'article_hash': self.verification_service.hasher.calculate_hash(text)
            }
    
    def submit_flag_endpoint(self, article_text: str, flag_data: Dict) -> Dict:
        """API endpoint for submitting flags"""
        try:
            success, message = self.verification_service.submit_flag(article_text, flag_data)
            return {
                'success': success,
                'message': message,
                'article_hash': self.verification_service.hasher.calculate_hash(article_text)
            }
        except Exception as e:
            logger.error(f"Flag submission error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def submit_correction_endpoint(self, article_text: str, correction_data: Dict) -> Dict:
        """API endpoint for submitting corrections"""
        try:
            success, message = self.verification_service.submit_correction(article_text, correction_data)
            return {
                'success': success,
                'message': message,
                'article_hash': self.verification_service.hasher.calculate_hash(article_text)
            }
        except Exception as e:
            logger.error(f"Correction submission error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Example configuration
    config = {
        'web3_provider_url': 'http://127.0.0.1:8545',  # Local Ganache
        'contract_addresses': {
            'publisher': '0x...',  # Replace with actual contract addresses
            'article': '0x...',
            'annotation': '0x...'
        },
        'ipfs_api_url': '/ip4/127.0.0.1/tcp/5001',
        'private_key': '0x...'  # Replace with actual private key
    }
    
    # Example usage

    # Initialize blockchain API
    blockchain_api = BlockchainAPI(config)
    
    # Test article verification
    sample_text = "यह एक नमूना हिंदी समाचार है जिसकी जांच की जा रही है।"
    
    verification_result = blockchain_api.verify_article_endpoint(sample_text)
    print("Verification Result:")
    print(json.dumps(verification_result, indent=2, ensure_ascii=False))
    
    # Test flag submission
    flag_data = {
        'flag_type': 'misleading',
        'description': 'This article contains misleading information',
        'evidence': 'Contrary evidence found at: https://example.com/fact-check'
    }
    
    flag_result = blockchain_api.submit_flag_endpoint(sample_text, flag_data)
    print("\nFlag Submission Result:")
    print(json.dumps(flag_result, indent=2, ensure_ascii=False))
    

    
    def get_article_info(self, article_hash: str) -> Optional[ArticleRecord]:
        """Get article information from blockchain"""
        try:
            result = self.contracts['article'].functions.getArticleInfo(article_hash).call()
            
            if result[2]:  # If article exists
                return ArticleRecord(
                    hash=article_hash,
                    publisher=result[0],
                    timestamp=result[1],
                    verified=True
                )
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting article info: {e}")
            return None
    
    def add_annotation(self, article_hash: str, annotation_type: str, ipfs_hash: str) -> Tuple[bool, str]:
        """Add an annotation to an article"""

        if not self.private_key:
            return False, "Private key required for transactions"
        
        function = self.contracts['annotation'].functions.addAnnotation(
            article_hash, annotation_type, ipfs_hash
        )
        
        # Estimate gas
        gas_estimate = function.estimateGas({'from': self.account.address})
        
        # Build transaction
        transaction = function.buildTransaction({
            'from': self.account.address,
            'gas': gas_estimate,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(self.account.address)})