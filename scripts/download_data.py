# --- scripts/download_data.py ---
#!/usr/bin/env python3
"""
Script to download required datasets for AMLPDS
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, dest_path):
    """Download file from URL to destination path"""
    logger.info(f"Downloading {url} to {dest_path}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded {dest_path}")

def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive"""
    logger.info(f"Extracting {archive_path} to {extract_to}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    
    logger.info(f"Extracted to {extract_to}")

def main():
    """Main function to download datasets"""
    # Create data directories
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (replace with actual URLs)
    datasets = {
        'phishtank': {
            'url': 'https://data.phishtank.com/data/online-valid.csv',
            'dest': raw_dir / 'phishtank' / 'phishtank.csv'
        },
        'enron': {
            'url': 'https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz',
            'dest': raw_dir / 'enron' / 'enron.tar.gz'
        }
    }
    
    # Download datasets
    for name, info in datasets.items():
        try:
            # Create directory
            info['dest'].parent.mkdir(parents=True, exist_ok=True)
            
            # Download if not exists
            if not info['dest'].exists():
                download_file(info['url'], info['dest'])
                
                # Extract if archive
                if info['dest'].suffix in ['.zip', '.gz']:
                    extract_archive(
                        str(info['dest']), 
                        str(info['dest'].parent)
                    )
            else:
                logger.info(f"{name} dataset already exists")
                
        except Exception as e:
            logger.error(f"Error downloading {name}: {e}")
            sys.exit(1)
    
    logger.info("All datasets downloaded successfully!")

if __name__ == '__main__':
    main()
