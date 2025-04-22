#!/usr/bin/env python
"""
Simple script to setup MIMIC data directories without requiring pandas.
"""

import os
import sys
import shutil
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
MIMIC_SOURCE_DIR = '/Users/Apexr/physionet.org/files/mimiciv/3.1'
PROJECT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROJECT_MIMIC_DIR = os.path.join(PROJECT_DATA_DIR, 'external', 'mimic')

def setup_mimic_directories():
    """Set up the necessary directories for MIMIC data."""
    # Create directory for organized MIMIC data in our project
    os.makedirs(PROJECT_MIMIC_DIR, exist_ok=True)
    logger.info(f"Created directory: {PROJECT_MIMIC_DIR}")
    
    # Create other necessary directories
    os.makedirs(os.path.join(PROJECT_DATA_DIR, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DATA_DIR, 'processed'), exist_ok=True)
    logger.info("Created project data directories")
    
    return True

def check_mimic_source():
    """Check if MIMIC source data exists."""
    if not os.path.exists(MIMIC_SOURCE_DIR):
        logger.error(f"MIMIC source directory not found: {MIMIC_SOURCE_DIR}")
        return False
    
    logger.info(f"MIMIC source directory found: {MIMIC_SOURCE_DIR}")
    
    # Check subdirectories
    hosp_dir = os.path.join(MIMIC_SOURCE_DIR, 'hosp')
    icu_dir = os.path.join(MIMIC_SOURCE_DIR, 'icu')
    
    if not os.path.exists(hosp_dir):
        logger.warning(f"Hospital data directory not found: {hosp_dir}")
    else:
        logger.info(f"Hospital data directory found: {hosp_dir}")
        
    if not os.path.exists(icu_dir):
        logger.warning(f"ICU data directory not found: {icu_dir}")
    else:
        logger.info(f"ICU data directory found: {icu_dir}")
    
    return True

def copy_important_files():
    """Copy important dictionary files from MIMIC source to project directory."""
    # Key dictionary files that are small enough to copy directly
    files_to_copy = [
        os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'd_icd_diagnoses.csv.gz'),
        os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'd_labitems.csv.gz')
    ]
    
    for src_file in files_to_copy:
        if os.path.exists(src_file):
            # Get base filename without path
            base_name = os.path.basename(src_file)
            # Remove .gz extension for our project
            if base_name.endswith('.gz'):
                dest_name = base_name[:-3]
            else:
                dest_name = base_name
            
            # Define destination path
            dest_file = os.path.join(PROJECT_MIMIC_DIR, dest_name)
            
            # Copy file
            try:
                # We'll just create an empty file for now - pandas would be needed to read and save
                with open(dest_file, 'w') as f:
                    f.write(f"This is a placeholder file for {base_name}. Please run the data extraction notebook to populate it.\n")
                
                logger.info(f"Created placeholder for {base_name} at {dest_file}")
            except Exception as e:
                logger.error(f"Error creating placeholder for {base_name}: {e}")
    
    return True

def create_readme():
    """Create README file in MIMIC directory with instructions."""
    readme_path = os.path.join(PROJECT_MIMIC_DIR, 'README.md')
    
    readme_content = f"""# MIMIC-IV Data for COVID-19 Detection
    
This directory contains MIMIC-IV data extracts for the COVID-19 detection project.

## Source Data
Original MIMIC-IV data is located at: 
{MIMIC_SOURCE_DIR}

## Setup Instructions

1. Run the data extraction notebook:
   ```
   jupyter notebook notebooks/06_mimic_data_exploration.ipynb
   ```

2. Execute all cells in the notebook to extract and process MIMIC-IV data

3. This will populate the following files:
   - patients_sample.csv: Sample of patient demographic data
   - admissions_sample.csv: Sample of admission data
   - diagnoses_sample.csv: Sample of ICD diagnosis codes
   - d_icd_diagnoses.csv: Dictionary of ICD codes and descriptions
   - d_labitems.csv: Dictionary of lab items
   - omr_sample.csv: Sample of Outpatient Medication Reconciliation data (text content)
   - labevents_sample.csv: Sample of laboratory test results

4. The notebook will also create:
   - classification_data.csv: Processed data ready for classification model
   - clinical_notes.csv: Extracted text data for NER

## Integration with Pipeline

The file `src/mimic_integration.py` provides functions to integrate this data with the COVID-19 detection pipeline.

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created README at {readme_path}")
    return True

def main():
    """Main function to set up MIMIC data integration."""
    logger.info("Setting up MIMIC data integration...")
    
    # Set up directories
    setup_mimic_directories()
    
    # Check MIMIC source data
    check_mimic_source()
    
    # Copy important files
    copy_important_files()
    
    # Create README
    create_readme()
    
    logger.info("MIMIC data setup complete! Please run the notebook to extract and process the data.")
    print("\nTo continue working with MIMIC data:")
    print("1. Install required packages: pip install pandas numpy matplotlib seaborn")
    print("2. Open the exploration notebook: jupyter notebook notebooks/06_mimic_data_exploration.ipynb")
    print("3. Run the notebook to extract and process MIMIC data")
    
if __name__ == "__main__":
    main()