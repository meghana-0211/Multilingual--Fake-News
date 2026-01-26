# backend/blockchain/web3_client.py
from web3 import Web3
import json
from pathlib import Path

class BlockchainClient:
    def __init__(self, provider_url='http://127.0.0.1:8545'):
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        self.account = None
        self.contracts = {}
        
    def load_contract(self, contract_name, address):
        """Load compiled contract ABI and create contract instance"""
        abi_path = Path(f'blockchain/build/contracts/{contract_name}.json')
        with open(abi_path, 'r') as f:
            contract_json = json.load(f)
            abi = contract_json['abi']
        
        contract = self.web3.eth.contract(address=address, abi=abi)
        self.contracts[contract_name] = contract
        return contract
    
    def set_account(self, private_key):
        """Set account for transactions"""
        self.account = self.web3.eth.account.from_key(private_key)
    
    def register_article(self, content_hash, language):
        """Register article on blockchain"""
        contract = self.contracts['ArticleRegistry']
        
        # Convert hash to bytes32
        hash_bytes = bytes.fromhex(content_hash)
        
        # Build transaction
        txn = contract.functions.registerArticle(
            hash_bytes,
            language
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': 200000,
            'gasPrice': self.web3.eth.gas_price
        })
        
        # Sign and send
        signed_txn = self.web3.eth.account.sign_transaction(txn, self.account.key)
        txn_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for receipt
        receipt = self.web3.eth.wait_for_transaction_receipt(txn_hash)
        return receipt
    
    def verify_article(self, content_hash):
        """Check if article is registered"""
        contract = self.contracts['ArticleRegistry']
        hash_bytes = bytes.fromhex(content_hash)
        
        exists, publisher, timestamp = contract.functions.verifyArticle(hash_bytes).call()
        
        return {
            'exists': exists,
            'publisher': publisher,
            'timestamp': timestamp
        }
    
    def add_annotation(self, content_hash, flag_type, ipfs_hash, confidence):
        """Add fact-check annotation"""
        contract = self.contracts['AnnotationRegistry']
        hash_bytes = bytes.fromhex(content_hash)
        
        txn = contract.functions.addAnnotation(
            hash_bytes,
            flag_type,
            ipfs_hash,
            confidence
        ).build_transaction({
            'from': self.account.address,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
            'gas': 300000,
            'gasPrice': self.web3.eth.gas_price
        })
        
        signed_txn = self.web3.eth.account.sign_transaction(txn, self.account.key)
        txn_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(txn_hash)
        
        return receipt
    
    def get_annotations(self, content_hash):
        """Retrieve all annotations for an article"""
        contract = self.contracts['AnnotationRegistry']
        hash_bytes = bytes.fromhex(content_hash)
        
        annotations = contract.functions.getAnnotations(hash_bytes).call()
        
        return [{
            'articleHash': ann[0].hex(),
            'factChecker': ann[1],
            'flagType': ann[2],
            'ipfsHash': ann[3],
            'timestamp': ann[4],
            'confidence': ann[5]
        } for ann in annotations]