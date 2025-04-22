#!/usr/bin/env python
"""
Script to download real COVID-19 data for the Disease Prediction Project.

This script uses the data_fetcher module to download real data from various sources
and sets up the data directory structure for the project.
"""

import os
import sys
import logging
from datetime import datetime
import argparse

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check for required packages
packages_installed = True

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required for downloading data.")
    print("Install with: pip install requests")
    packages_installed = False

try:
    import pandas as pd
except ImportError:
    print("Error: 'pandas' package is required for processing data.")
    print("Install with: pip install pandas")
    packages_installed = False

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: 'beautifulsoup4' package is required for processing HTML.")
    print("Install with: pip install beautifulsoup4")
    packages_installed = False

if not packages_installed:
    print("\nPlease install the required packages and try again.")
    print("You can install all requirements with: pip install -r requirements.txt")
    sys.exit(1)

# Import project modules
try:
    from src.data_fetcher import (fetch_all_public_datasets, print_dataset_info,
                                fetch_cord19_metadata, fetch_covid19_cases,
                                fetch_covid_clinical_trials, fetch_covid_twitter_dataset)
    import src.data_collection as dc
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"data_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_data_directories():
    """Create necessary data directories if they don't exist."""
    dirs = [
        'data',
        'data/raw',
        'data/external',
        'data/processed',
        'data/interim'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Directory {d} is ready")

def main():
    """Main function to download and prepare real data."""
    parser = argparse.ArgumentParser(description='Download real data for COVID-19 detection project')
    parser.add_argument('--datasets', nargs='+', choices=['cord19', 'cdc', 'trials', 'twitter', 'all'],
                      default=['all'], help='Datasets to download (default: all)')
    parser.add_argument('--sample', action='store_true', help='Download only sample data (smaller/faster)')
    parser.add_argument('--limit', type=int, default=1000, help='Limit for number of records to fetch')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_data_directories()
    
    # Show available datasets
    print_dataset_info()
    
    # Determine which datasets to download
    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['cord19', 'cdc', 'trials', 'twitter']
    
    logger.info(f"Will download the following datasets: {datasets_to_download}")
    
    # Download selected datasets
    download_results = {}
    
    if 'cord19' in datasets_to_download:
        logger.info("Downloading CORD-19 dataset...")
        download_results['cord19'] = fetch_cord19_metadata()
        
        # Also use the data_collection module with real data flag
        dc.USE_SYNTHETIC_DATA = False
        cord19_sample = dc.fetch_cord19_dataset('data/raw', limit=args.limit)
        logger.info(f"CORD-19 sample processed: {cord19_sample}")
    
    if 'cdc' in datasets_to_download:
        logger.info("Downloading CDC case surveillance data...")
        download_results['cdc'] = fetch_covid19_cases()
    
    if 'trials' in datasets_to_download:
        logger.info("Downloading COVID-19 clinical trials data...")
        download_results['trials'] = fetch_covid_clinical_trials()
        
        # Also use the data_collection module with real data flag
        dc.USE_SYNTHETIC_DATA = False
        cdc_guidelines = dc.fetch_cdc_guidelines('data/raw')
        logger.info(f"CDC guidelines processed: {cdc_guidelines}")
    
    if 'twitter' in datasets_to_download:
        logger.info("Downloading COVID-19 Twitter data...")
        download_results['twitter'] = fetch_covid_twitter_dataset()
    
    # Summarize results
    logger.info("\nDownload Summary:")
    successful = 0
    failed = 0
    
    for dataset, path in download_results.items():
        if path:
            logger.info(f"✅ {dataset} dataset - {path}")
            successful += 1
        else:
            logger.info(f"❌ {dataset} dataset - Failed to download")
            failed += 1
    
    logger.info(f"\nResults: {successful} successful, {failed} failed")
    
    # Instructions for next steps
    logger.info("\nNext Steps:")
    logger.info("1. Open the notebooks in the notebooks/ directory to explore the data")
    logger.info("2. Run the NER notebooks to extract entities from real data")
    logger.info("3. Use the extracted entities for classification")
    
    # Set the global flag in data_collection.py to use real data by default
    logger.info("\nUpdating data_collection.py to use real data by default...")
    
    # We'll use Bash or file edit to modify the file, but here we just note the change
    logger.info("To permanently switch to real data, set USE_SYNTHETIC_DATA = False in src/data_collection.py")

if __name__ == "__main__":
    try:
        main()
        logger.info("Data download complete!")
    except Exception as e:
        logger.error(f"Error during data download: {e}", exc_info=True)
        sys.exit(1)