"""
Blockchain Configuration
Save your contract addresses here
"""

# Ganache URL (your local blockchain)
BLOCKCHAIN_URL = 'http://127.0.0.1:8545'

# Contract Addresses (REPLACE WITH YOUR ADDRESSES!)
PUBLISHER_REGISTRY = '0xFC628dd79137395F3C9744e33b1c5DE554D94882'
ARTICLE_REGISTRY = '0x5f8e26fAcC23FA4cbd87b8d9Dbbd33D5047abDE1'
ANNOTATION_REGISTRY = '0x21a59654176f2689d12E828B77a783072CD26680'
# Your account private key (from Ganache)
# REPLACE WITH YOUR PRIVATE KEY FROM GANACHE!
PRIVATE_KEY = '0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d'

# Path to contract ABIs
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONTRACTS_PATH = os.path.join(PROJECT_ROOT, 'blockchain', 'build', 'contracts')