# scripts/prepare_dataset.py
import pandas as pd
from pathlib import Path
print("datasets loaded")
def load_language_dataset(lang_dir):
    """Load fake and real news for one language"""
    fake_dir = lang_dir / f'{lang_dir.name.split("_")[0]}_fake_news'
    real_dir = lang_dir / f'{lang_dir.name.split("_")[0]}_real_news'
    
    data = []
    
    # Load fake news
    for txt_file in fake_dir.glob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                data.append({'text': text, 'label': 0, 'language': lang_dir.name.split('_')[0].lower()})
    
    # Load real news
    for txt_file in real_dir.glob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                data.append({'text': text, 'label': 1, 'language': lang_dir.name.split('_')[0].lower()})
    
    return pd.DataFrame(data)

def create_unified_dataset():
    """Combine all languages into single dataset"""
    raw_dir = Path("/Users/Megha/finalyearproj/Hindi-Fake-News_detection/data/raw")
    all_data = []
    print("loaded dataset")
    
    for lang_folder in raw_dir.glob('*_F&R_News'):
        print(f"Loading {lang_folder.name}...")
        df = load_language_dataset(lang_folder)
        all_data.append(df)
        print(f"  - Loaded {len(df)} articles")
    
    # Combine all
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_dir = Path('/Users/Megha/finalyearproj/Hindi-Fake-News_detection/data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_dir / 'multilingual_dataset.csv', index=False)
    
    # Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total articles: {len(combined_df)}")
    print(f"\nBy language:")
    print(combined_df['language'].value_counts())
    print(f"\nBy label:")
    print(combined_df['label'].value_counts())
    print(f"\nLabel distribution by language:")
    print(combined_df.groupby(['language', 'label']).size())
    
    return combined_df

if __name__ == "__main__":
    create_unified_dataset()