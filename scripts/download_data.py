# scripts/download_data.py
import os
import zipfile
import requests
from pathlib import Path

DATASET_URLS = {
    'hindi': 'https://zenodo.org/record/11408513/files/Hindi_F&R_News.zip',
    'gujarati': 'https://zenodo.org/record/11408513/files/Gujarati_F&R_News.zip',
    'marathi': 'https://zenodo.org/record/11408513/files/Marathi_F&R_News.zip',
    'telugu': 'https://zenodo.org/record/11408513/files/Telugu_F&R_News.zip'
}

def download_and_extract():
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for lang, url in DATASET_URLS.items():
        zip_path = data_dir / f'{lang}.zip'
        extract_dir = data_dir / f'{lang.capitalize()}_F&R_News'
        
        # Download
        if not zip_path.exists():
            print(f"Downloading {lang}...")
            response = requests.get(url, stream=True)
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract
        if not extract_dir.exists():
            print(f"Extracting {lang}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
    print("Dataset preparation complete!")