"""
Blockchain Configuration
Save your contract addresses here
"""

# Ganache URL (your local blockchain)
BLOCKCHAIN_URL = 'http://127.0.0.1:8545'

# Contract Addresses (REPLACE WITH YOUR ADDRESSES!)
PUBLISHER_REGISTRY = '0x254dffcd3277C0b1660F6d42EFbB754edaBAbC2B'
ARTICLE_REGISTRY = '0xC89Ce4735882C9F0f0FE26686c53074E09B0D550'
ANNOTATION_REGISTRY = '0xD833215cBcc3f914bD1C9ece3EE7BF8B14f841bb'
# Your account private key (from Ganache)
# REPLACE WITH YOUR PRIVATE KEY FROM GANACHE!
PRIVATE_KEY = '0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d'

# Path to contract ABIs
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONTRACTS_PATH = os.path.join(PROJECT_ROOT, 'blockchain', 'build', 'contracts')