# backend/models/preprocessor.py
import re
import torch
from transformers import AutoTokenizer
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize

class MultilingualPreprocessor:
    def __init__(self, model_name='ai4bharat/indic-bert'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.normalizer_factory = indic_normalize.IndicNormalizerFactory()
        
        # Language-specific normalizers
        self.normalizers = {
            'hindi': self.normalizer_factory.get_normalizer('hi'),
            'gujarati': self.normalizer_factory.get_normalizer('gu'),
            'marathi': self.normalizer_factory.get_normalizer('mr'),
            'telugu': self.normalizer_factory.get_normalizer('te')
        }
    
    def normalize_text(self, text, language):
        """Apply language-specific normalization"""
        normalizer = self.normalizers.get(language)
        if normalizer:
            text = normalizer.normalize(text)
        
        # Common cleaning
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s।॥]', '', text)  # Keep only letters, spaces, dandas
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text, max_length=512):
        """Tokenize with language-aware handling"""
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding
    
    def preprocess_batch(self, texts, languages, max_length=512):
        """Preprocess batch of multilingual texts"""
        normalized = [
            self.normalize_text(text, lang)
            for text, lang in zip(texts, languages)
        ]
        
        encodings = self.tokenizer(
            normalized,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return encodings, normalized