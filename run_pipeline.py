#!/usr/bin/env python
"""
Main pipeline script for COVID-19 Detection Project.

This script runs the complete pipeline from data collection
to NER extraction to classification dataset preparation.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
log_file = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the necessary directories and environment."""
    logger.info("Setting up environment")
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/external', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Add src directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
    
    logger.info("Environment setup complete")
    return True

def download_data(args):
    """Download necessary data for the pipeline."""
    logger.info("Downloading data")
    
    try:
        # Import and run data collection
        from src.scripts.download_real_data import main as download_main
        download_main()
        logger.info("CDC and CORD-19 data download complete")
        
        # Set up MIMIC data if available
        if args.with_mimic:
            from src.scripts.setup_mimic_data import main as setup_mimic_main
            setup_mimic_main()
            logger.info("MIMIC data setup complete")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False

def run_ner_extraction(args):
    """Run the NER extraction pipeline."""
    logger.info("Running NER extraction")
    
    try:
        # Run NER on MIMIC data if available
        if args.with_mimic:
            sys.path.insert(0, os.path.dirname(__file__))
            from tests.test_mimic_ner import main as test_mimic_main
            test_mimic_main()
            logger.info("NER extraction on MIMIC data complete")
        else:
            # Run basic NER test
            sys.path.insert(0, os.path.dirname(__file__))
            from tests.test_ner import main as test_ner_main
            test_ner_main()
            logger.info("NER extraction complete")
        
        # Check integration
        from tests.test_ner_integration import main as test_integration_main
        test_integration_main()
        logger.info("NER integration test complete")
        
        return True
    except Exception as e:
        logger.error(f"Error running NER extraction: {e}")
        return False

def prepare_classification_data(args):
    """Prepare the classification dataset."""
    logger.info("Preparing classification dataset")
    
    try:
        from src.scripts.prepare_classification_dataset import main as prepare_main
        prepare_main()
        logger.info("Classification dataset preparation complete")
        
        # Preview the dataset
        sys.path.insert(0, os.path.dirname(__file__))
        from src.scripts.preview_classification_data import main as preview_main
        preview_main()
        
        # Analyze the dataset if requested
        if args.analyze:
            from src.scripts.analyze_classification_dataset import main as analyze_main
            analyze_main()
            logger.info("Classification dataset analysis complete")
        
        return True
    except Exception as e:
        logger.error(f"Error preparing classification data: {e}")
        return False

def main():
    """Main pipeline function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the COVID-19 detection pipeline')
    parser.add_argument('--with-mimic', action='store_true', help='Include MIMIC data in the pipeline')
    parser.add_argument('--analyze', action='store_true', help='Run analysis on the classification dataset')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download step')
    parser.add_argument('--skip-ner', action='store_true', help='Skip NER extraction step')
    args = parser.parse_args()
    
    # Start pipeline
    logger.info("Starting COVID-19 detection pipeline")
    
    # Setup
    if not setup_environment():
        logger.error("Environment setup failed")
        return 1
    
    # Download data
    if not args.skip_download:
        if not download_data(args):
            logger.error("Data download failed")
            return 1
    else:
        logger.info("Skipping data download step")
    
    # Run NER extraction
    if not args.skip_ner:
        if not run_ner_extraction(args):
            logger.error("NER extraction failed")
            return 1
    else:
        logger.info("Skipping NER extraction step")
    
    # Prepare classification data
    if not prepare_classification_data(args):
        logger.error("Classification data preparation failed")
        return 1
    
    # Pipeline complete
    logger.info("COVID-19 detection pipeline completed successfully")
    print(f"\nPipeline completed successfully! See {log_file} for details.")
    print("\nNext steps:")
    print("1. Explore the notebooks: jupyter notebook notebooks/")
    print("2. Train models using the prepared dataset")
    print("3. Evaluate model performance")
    return 0

if __name__ == "__main__":
    sys.exit(main())