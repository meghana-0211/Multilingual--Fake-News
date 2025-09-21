import pandas as pd
import numpy as np
import re
import unicodedata
import zipfile
import os
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

class HindiTextPreprocessor:
    def __init__(self, model_name="ai4bharat/indic-bert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hindi_stopwords = self._load_hindi_stopwords()
        self.english_stopwords = self._load_english_stopwords()
    
    def _load_hindi_stopwords(self) -> set:
        """Load Hindi stopwords"""
        hindi_stopwords = {
            'और', 'है', 'हैं', 'का', 'के', 'की', 'को', 'में', 'से', 'पर', 'यह', 'वह', 
            'एक', 'दो', 'तीन', 'होना', 'करना', 'आना', 'जाना', 'देना', 'लेना', 'कहना',
            'आप', 'हम', 'तुम', 'मैं', 'वे', 'इस', 'उस', 'जो', 'कि', 'तो', 'अगर',
            'लेकिन', 'क्योंकि', 'इसलिए', 'फिर', 'अब', 'तब', 'यहाँ', 'वहाँ', 'कहाँ',
            'कब', 'क्यों', 'कैसे', 'क्या', 'कौन', 'किस', 'किसे', 'किससे'
        }
        return hindi_stopwords
    
    def _load_english_stopwords(self) -> set:
        """Load English stopwords for Hinglish content"""
        english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'
        }
        return english_stopwords
    
    def normalize_devanagari(self, text: str) -> str:
        """Normalize Devanagari script variations"""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Common Devanagari character normalizations
        normalizations = {
            'क़': 'क', 'ख़': 'ख', 'ग़': 'ग', 'ज़': 'ज', 'ड़': 'ड', 
            'ढ़': 'ढ', 'फ़': 'फ', 'य़': 'य', 'ऱ': 'र', 'ऴ': 'ल'
        }
        
        for old, new in normalizations.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Devanagari, Latin letters, and numbers
        text = re.sub(r'[^\u0900-\u097F\u0A00-\u0A7F\u0B00-\u0B7F\u0C00-\u0C7F\u0D00-\u0D7F\u0980-\u09FF\w\s।.,!?]', '', text)
        
        # Normalize Devanagari
        text = self.normalize_devanagari(text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """Remove Hindi and English stopwords"""
        words = text.split()
        filtered_words = [
            word for word in words 
            if word.lower() not in self.hindi_stopwords 
            and word.lower() not in self.english_stopwords
            and len(word) > 1
        ]
        return ' '.join(filtered_words)
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> str:
        """Complete text preprocessing pipeline"""
        text = self.clean_text(text)
        if remove_stopwords:
            text = self.remove_stopwords(text)
        return text
    
    def tokenize_for_bert(self, text: str, max_length: int = 512) -> Dict:
        """Tokenize text for BERT model"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts"""
        return [self.preprocess_text(text) for text in texts]


class HindiFakeNewsDatasetLoader:
    def __init__(self, zip_file_path: str = None, extracted_folder_path: str = None):
        """
        Initialize with either zip file path or extracted folder path
        
        Args:
            zip_file_path: Path to the zip file containing text files
            extracted_folder_path: Path to folder containing extracted text files
        """
        self.zip_file_path = zip_file_path
        self.extracted_folder_path = extracted_folder_path
        self.preprocessor = HindiTextPreprocessor()
    
    def extract_zip_file(self, extract_to: str = "extracted_data") -> str:
        """Extract zip file and return extraction path"""
        if not self.zip_file_path:
            raise ValueError("Zip file path not provided")
            
        print(f"Extracting {self.zip_file_path} to {extract_to}")
        
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        self.extracted_folder_path = extract_to
        print(f"Extraction completed to {extract_to}")
        return extract_to
    
    def load_text_files(self, folder_path: str = None) -> pd.DataFrame:
        """Load all text files from folder and create DataFrame"""
        if folder_path is None:
            folder_path = self.extracted_folder_path
            
        if not folder_path or not os.path.exists(folder_path):
            raise ValueError(f"Folder path {folder_path} does not exist")
        
        texts = []
        filenames = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        # Try different encodings
                        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                                    if content.strip():  # Only add non-empty files
                                        texts.append(content)
                                        filenames.append(file)
                                break
                            except UnicodeDecodeError:
                                continue
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
                        continue
        
        print(f"Loaded {len(texts)} text files")
        
        # Create DataFrame with fake news label (assuming all files are fake news)
        df = pd.DataFrame({
            'filename': filenames,
            'text': texts,
            'label': 1  # 1 for fake news
        })
        
        return df
    
    def create_balanced_dataset(self, fake_df: pd.DataFrame, genuine_texts: List[str] = None) -> pd.DataFrame:
        """
        Create a balanced dataset by adding genuine news or duplicating/augmenting existing data
        
        Args:
            fake_df: DataFrame containing fake news
            genuine_texts: List of genuine news texts (optional)
        """
        if genuine_texts:
            # If genuine texts are provided
            genuine_df = pd.DataFrame({
                'filename': [f'genuine_{i}.txt' for i in range(len(genuine_texts))],
                'text': genuine_texts,
                'label': 0  # 0 for genuine news
            })
            
            # Balance the dataset
            min_count = min(len(fake_df), len(genuine_df))
            fake_sample = fake_df.sample(min_count, random_state=42)
            genuine_sample = genuine_df.sample(min_count, random_state=42)
            
            balanced_df = pd.concat([fake_sample, genuine_sample]).reset_index(drop=True)
            
        else:
            # If no genuine texts, create a synthetic balanced dataset
            # by treating half as genuine (this is just for demonstration)
            print("Warning: No genuine texts provided. Creating synthetic balanced dataset for demonstration.")
            
            # Shuffle and split the fake news into two groups
            shuffled_df = fake_df.sample(frac=1, random_state=42).reset_index(drop=True)
            mid_point = len(shuffled_df) // 2
            
            # Treat first half as fake, second half as genuine (for demo purposes)
            fake_part = shuffled_df.iloc[:mid_point].copy()
            genuine_part = shuffled_df.iloc[mid_point:].copy()
            genuine_part['label'] = 0  # Change label to genuine
            
            balanced_df = pd.concat([fake_part, genuine_part]).reset_index(drop=True)
        
        # Shuffle the final dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Created balanced dataset with {len(balanced_df)} samples")
        print(f"Genuine: {(balanced_df['label'] == 0).sum()}, Fake: {(balanced_df['label'] == 1).sum()}")
        
        return balanced_df
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the entire dataset"""
        print("Preprocessing text data...")
        
        # Clean and preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocessor.preprocess_text)
        
        # Remove empty texts after preprocessing
        initial_count = len(df)
        df = df[df['cleaned_text'].str.len() > 10].reset_index(drop=True)
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} texts with insufficient content after preprocessing")
        
        print(f"Final dataset shape: {df.shape}")
        return df
    
    def create_train_val_test_split(self, df: pd.DataFrame, 
                                   test_size: float = 0.15, 
                                   val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train, validation, and test sets"""
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=42, 
            stratify=train_val_df['label']
        )
        
        print(f"Train size: {len(train_df)}")
        print(f"Validation size: {len(val_df)}")
        print(f"Test size: {len(test_df)}")
        
        return train_df, val_df, test_df


class FeatureExtractor:
    def __init__(self):
        self.preprocessor = HindiTextPreprocessor()
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Punctuation statistics
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['period_count'] = text.count('.')
        features['hindi_period_count'] = text.count('।')  # Hindi period
        
        # Capital letters (for English parts in Hinglish)
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Number statistics
        features['number_count'] = len(re.findall(r'\d+', text))
        
        # URL and mention indicators
        features['has_url'] = 1 if 'http' in text.lower() else 0
        features['has_mention'] = 1 if '@' in text else 0
        
        # Hindi-specific features
        features['devanagari_ratio'] = sum(1 for c in text if '\u0900' <= c <= '\u097F') / len(text) if text else 0
        features['english_ratio'] = sum(1 for c in text if c.isalpha() and ord(c) < 128) / len(text) if text else 0
        
        return features
    
    def extract_features_batch(self, texts: List[str]) -> pd.DataFrame:
        """Extract features for a batch of texts"""
        features_list = []
        for text in texts:
            features = self.extract_linguistic_features(text)
            features_list.append(features)
        
        return pd.DataFrame(features_list)


if __name__ == "__main__":
    # Example usage with your zip file
    zip_file_path = "path/to/Hindi_fake_news.zip"  # Replace with your actual zip file path
    
    # Initialize the loader
    loader = HindiFakeNewsDatasetLoader(zip_file_path=zip_file_path)
    
    try:
        # Extract zip file
        extracted_path = loader.extract_zip_file("extracted_hindi_news")
        
        # Load text files
        df = loader.load_text_files()
        
        if not df.empty:
            print(f"Loaded {len(df)} fake news articles")
            
            # Create balanced dataset (synthetic for demonstration)
            # In real scenario, you would provide genuine news texts
            balanced_df = loader.create_balanced_dataset(df)
            
            # Preprocess dataset
            df_processed = loader.preprocess_dataset(balanced_df)
            
            # Create train/val/test splits
            train_df, val_df, test_df = loader.create_train_val_test_split(df_processed)
            
            # Save processed datasets
            os.makedirs('processed_data', exist_ok=True)
            train_df.to_csv('processed_data/train_data.csv', index=False)
            val_df.to_csv('processed_data/val_data.csv', index=False)
            test_df.to_csv('processed_data/test_data.csv', index=False)
            
            # Extract additional features
            feature_extractor = FeatureExtractor()
            train_features = feature_extractor.extract_features_batch(train_df['cleaned_text'].tolist())
            train_features.to_csv('processed_data/train_features.csv', index=False)
            
            print("Data preprocessing completed successfully!")
            print(f"Sample processed text: {train_df['cleaned_text'].iloc[0][:200]}...")
            
            # Display some statistics
            print(f"\nDataset Statistics:")
            print(f"Total samples: {len(df_processed)}")
            print(f"Average text length: {df_processed['text'].str.len().mean():.2f} characters")
            print(f"Average cleaned text length: {df_processed['cleaned_text'].str.len().mean():.2f} characters")
            
        else:
            print("No data loaded. Please check your file paths.")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")
        
    # Alternative: If you already have extracted folder
    # loader_folder = HindiFakeNewsDatasetLoader(extracted_folder_path="path/to/extracted/folder")
    # df = loader_folder.load_text_files()