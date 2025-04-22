#!/usr/bin/env python
"""
Setup script for MIMIC-IV Note Module integration.

This script extracts the MIMIC-IV note module ZIP file and integrates
clinical notes into our project structure for COVID-19 detection.
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import gzip
import logging
from datetime import datetime
import re
import shutil
import random
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Define paths
MIMIC_ZIP_PATH = '/Users/Apexr/Downloads/mimic-iv-note-deidentified-free-text-clinical-notes-2.2.zip'
MIMIC_DIR = os.path.join(project_root, 'data', 'external', 'mimic')
NOTE_MODULE_DIR = os.path.join(MIMIC_DIR, 'note_module')
SAMPLE_NOTES_PATH = os.path.join(NOTE_MODULE_DIR, 'clinical_notes_sample.csv')
COVID_NOTES_PATH = os.path.join(NOTE_MODULE_DIR, 'covid_notes.csv')
CLASSIFICATION_DATASET = os.path.join(project_root, 'data', 'processed', 'covid_classification_dataset.csv')
LABELED_DATASET_PATH = os.path.join(project_root, 'data', 'processed', 'covid_notes_classification.csv')

# COVID-related keywords
COVID_KEYWORDS = [
    'covid', 'covid-19', 'covid19', 'coronavirus', 'corona virus', 'sars-cov-2', 
    'sars cov 2', 'pandemic', 'novel coronavirus', 'cov-2'
]

def extract_zip_file():
    """Extract the MIMIC-IV Note module ZIP file."""
    if not os.path.exists(MIMIC_ZIP_PATH):
        logger.error(f"MIMIC-IV Note module ZIP file not found at: {MIMIC_ZIP_PATH}")
        return False
    
    # Check if files already exist
    if os.path.exists(os.path.join(NOTE_MODULE_DIR, 'discharge_notes.csv.gz')) and \
       os.path.exists(os.path.join(NOTE_MODULE_DIR, 'radiology_detail.csv.gz')):
        logger.info("Note files already extracted")
        return True
    
    logger.info(f"Extracting MIMIC-IV Note module from: {MIMIC_ZIP_PATH}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(NOTE_MODULE_DIR, exist_ok=True)
        
        # Extract ZIP file
        with zipfile.ZipFile(MIMIC_ZIP_PATH, 'r') as zip_ref:
            # Get a list of all files in the ZIP
            file_list = zip_ref.namelist()
            
            # Find the notes files (look for discharge_notes.csv.gz and radiology_detail.csv.gz)
            note_files = [f for f in file_list if ('discharge' in f.lower() or 'radiology' in f.lower()) and f.endswith('.gz')]
            
            if not note_files:
                logger.error("No note files found in the ZIP file")
                return False
            
            logger.info(f"Found {len(note_files)} note files")
            
            # Extract each note file to the note_module directory
            for note_file in note_files:
                # Get the base filename
                base_name = os.path.basename(note_file)
                target_path = os.path.join(NOTE_MODULE_DIR, base_name)
                
                logger.info(f"Extracting {note_file} to {target_path}")
                
                with zip_ref.open(note_file) as source, open(target_path, 'wb') as target:
                    # Read and write in chunks to handle large files
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break
                        target.write(chunk)
            
            logger.info(f"Extracted {len(note_files)} note files to: {NOTE_MODULE_DIR}")
            return True
    except Exception as e:
        logger.error(f"Error extracting ZIP file: {e}")
        return False

def sample_notes(sample_size=1000):
    """Create a sample of clinical notes for analysis."""
    # Check if sample already exists
    if os.path.exists(SAMPLE_NOTES_PATH):
        logger.info(f"Sample notes file already exists at: {SAMPLE_NOTES_PATH}")
        return True
    
    # Check if discharge notes file exists
    discharge_notes_path = os.path.join(NOTE_MODULE_DIR, 'discharge_notes.csv.gz')
    if not os.path.exists(discharge_notes_path):
        logger.error(f"Discharge notes file not found at: {discharge_notes_path}")
        return False
    
    logger.info(f"Creating a sample of {sample_size} notes from discharge notes")
    
    try:
        # Read the header to get column names
        with gzip.open(discharge_notes_path, 'rt') as f:
            header = f.readline().strip()
        
        columns = header.split(',')
        logger.info(f"Note columns: {columns}")
        
        # Create a temporary file for sampling
        temp_sample_path = os.path.join(NOTE_MODULE_DIR, 'temp_sample.csv')
        
        # Write header to the sample file
        with open(temp_sample_path, 'w') as f:
            f.write(header + '\n')
        
        # Count lines in the file (up to a limit)
        logger.info("Counting lines in the file (this may take a while)...")
        line_count = 0
        line_limit = 100000  # Limit counting to avoid waiting too long
        
        with gzip.open(discharge_notes_path, 'rt') as f:
            next(f)  # Skip header
            for _ in range(line_limit):
                if next(f, None) is None:
                    break
                line_count += 1
        
        if line_count == line_limit:
            logger.info(f"File has more than {line_limit} lines")
            total_lines = line_limit
        else:
            logger.info(f"File has {line_count} lines")
            total_lines = line_count
        
        # Calculate sampling probability
        sample_prob = min(1.0, sample_size / float(total_lines))
        
        # Sample lines from the file
        logger.info(f"Sampling with probability {sample_prob:.4f}")
        sampled_lines = 0
        
        with gzip.open(discharge_notes_path, 'rt') as f:
            next(f)  # Skip header
            for line in tqdm(f, desc="Sampling notes", total=total_lines):
                if random.random() <= sample_prob and sampled_lines < sample_size:
                    with open(temp_sample_path, 'a') as out_f:
                        out_f.write(line)
                    sampled_lines += 1
                
                # Stop if we have enough samples
                if sampled_lines >= sample_size:
                    break
        
        logger.info(f"Sampled {sampled_lines} notes")
        
        # Convert the temp file to DataFrame for final processing
        sample_df = pd.read_csv(temp_sample_path)
        
        # Remove the temp file
        os.remove(temp_sample_path)
        
        # Save the final sample
        sample_df.to_csv(SAMPLE_NOTES_PATH, index=False)
        logger.info(f"Saved sample notes to: {SAMPLE_NOTES_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error sampling notes: {e}")
        if os.path.exists(temp_sample_path):
            os.remove(temp_sample_path)
        return False

def identify_covid_notes():
    """Identify notes likely related to COVID-19."""
    if os.path.exists(COVID_NOTES_PATH) and os.path.getsize(COVID_NOTES_PATH) > 100:
        logger.info(f"COVID notes file already exists at: {COVID_NOTES_PATH}")
        return True
    
    if not os.path.exists(SAMPLE_NOTES_PATH):
        logger.error(f"Sample notes file not found at: {SAMPLE_NOTES_PATH}")
        return False
    
    logger.info(f"Identifying COVID-19 related notes from {SAMPLE_NOTES_PATH}")
    
    try:
        # Load the sample notes
        notes_df = pd.read_csv(SAMPLE_NOTES_PATH)
        
        if 'text' not in notes_df.columns:
            # Check if there's a column containing text
            text_columns = [col for col in notes_df.columns if 'text' in col.lower() or 'note' in col.lower()]
            if text_columns:
                text_col = text_columns[0]
                logger.info(f"Using column '{text_col}' as text content")
                notes_df['text'] = notes_df[text_col]
            else:
                logger.error("No text column found in sample notes")
                return False
        
        # Function to check if a note contains COVID keywords
        def contains_covid_keywords(text):
            if not isinstance(text, str):
                return False
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in COVID_KEYWORDS)
        
        # Identify COVID-related notes
        notes_df['is_covid_related'] = notes_df['text'].apply(contains_covid_keywords)
        covid_notes = notes_df[notes_df['is_covid_related']]
        
        # Add a likelihood score for being a positive COVID case
        def estimate_covid_positive(text):
            if not isinstance(text, str):
                return 0.0
            
            text_lower = text.lower()
            
            # Strong positive indicators
            if 'covid-19 positive' in text_lower or 'positive for covid' in text_lower:
                return 1.0
            
            # Look for phrases indicating a positive test
            positive_indicators = [
                'test positive', 'tested positive', 'positive test', 
                'covid positive', 'positive for covid-19',
                'confirmed covid', 'covid-19 confirmed'
            ]
            
            if any(indicator in text_lower for indicator in positive_indicators):
                return 0.9
            
            # Check for symptoms + positive language
            symptoms = ['fever', 'cough', 'shortness of breath', 'loss of taste', 'loss of smell']
            has_symptoms = any(symptom in text_lower for symptom in symptoms)
            
            if has_symptoms and ('probable covid' in text_lower or 'suspected covid' in text_lower):
                return 0.7
            
            if has_symptoms:
                return 0.3
            
            return 0.1
        
        covid_notes['covid_positive_likelihood'] = covid_notes['text'].apply(estimate_covid_positive)
        
        # If we have very few COVID notes, we need to create synthetic ones
        if len(covid_notes) < 10:
            logger.warning(f"Only found {len(covid_notes)} COVID notes, adding synthetic examples")
            
            # Create synthetic positive examples
            synthetic_positive = """
            Patient is a 45-year-old male who presents with fever, dry cough, fatigue, and loss of taste and smell for the past 5 days.
            Patient reports contact with a confirmed COVID case last week.
            Vitals: Temp 38.5°C, HR 95, BP 128/82, RR 18, O2 Sat 94% on room air.
            Physical exam reveals mild respiratory distress. Lungs with scattered rhonchi bilaterally.
            Assessment: Clinical presentation consistent with COVID-19 infection.
            Plan: COVID-19 PCR test ordered. Patient advised to self-isolate pending results.
            """
            
            # Get the columns from the notes DataFrame
            synthetic_df = pd.DataFrame(columns=notes_df.columns)
            
            # Add synthetic example
            synthetic_row = {col: '' for col in notes_df.columns}
            synthetic_row['text'] = synthetic_positive
            synthetic_row['is_covid_related'] = True
            synthetic_row['covid_positive_likelihood'] = 0.9
            
            # Add ID columns if they exist
            if 'note_id' in notes_df.columns:
                synthetic_row['note_id'] = 'SYNTH_001'
            if 'subject_id' in notes_df.columns:
                synthetic_row['subject_id'] = 'SYNTH_001'
                
            synthetic_df = pd.concat([synthetic_df, pd.DataFrame([synthetic_row])], ignore_index=True)
            
            # Combine with any real COVID notes
            covid_notes = pd.concat([covid_notes, synthetic_df], ignore_index=True)
            logger.info(f"Added synthetic examples, now have {len(covid_notes)} COVID notes")
        
        # Save the COVID-related notes
        covid_notes.to_csv(COVID_NOTES_PATH, index=False)
        
        logger.info(f"Identified {len(covid_notes)} COVID-related notes")
        logger.info(f"Saved COVID notes to: {COVID_NOTES_PATH}")
        
        # Print some statistics
        positive_cases = len(covid_notes[covid_notes['covid_positive_likelihood'] >= 0.7])
        suspected_cases = len(covid_notes[covid_notes['covid_positive_likelihood'].between(0.3, 0.7)])
        mentioned_cases = len(covid_notes[covid_notes['covid_positive_likelihood'] < 0.3])
        
        logger.info(f"Likely positive cases: {positive_cases}")
        logger.info(f"Suspected cases: {suspected_cases}")
        logger.info(f"Mentioned but unlikely cases: {mentioned_cases}")
        
        return True
    except Exception as e:
        logger.error(f"Error identifying COVID notes: {e}")
        return False

def create_labeled_dataset(max_samples=5000):
    """Create a labeled dataset by combining notes with classification labels."""
    if os.path.exists(LABELED_DATASET_PATH):
        logger.info(f"Labeled dataset already exists at: {LABELED_DATASET_PATH}")
        return True
    
    if not os.path.exists(CLASSIFICATION_DATASET):
        logger.error(f"Classification dataset not found at: {CLASSIFICATION_DATASET}")
        return False
    
    if not os.path.exists(COVID_NOTES_PATH):
        logger.info("COVID notes not found, identifying them first")
        if not identify_covid_notes():
            logger.error("Failed to identify COVID notes")
            return False
    
    logger.info(f"Creating labeled dataset from {CLASSIFICATION_DATASET} and {COVID_NOTES_PATH}")
    
    try:
        # Load classification dataset
        class_df = pd.read_csv(CLASSIFICATION_DATASET, nrows=max_samples)
        logger.info(f"Loaded {len(class_df)} classification records")
        
        # Load COVID notes
        covid_notes_df = pd.read_csv(COVID_NOTES_PATH)
        logger.info(f"Loaded {len(covid_notes_df)} COVID notes")
        
        # Create a mapping from positive/negative status to COVID notes
        positive_notes = covid_notes_df[covid_notes_df['covid_positive_likelihood'] >= 0.5]
        negative_notes = covid_notes_df[covid_notes_df['covid_positive_likelihood'] < 0.5]
        
        logger.info(f"Found {len(positive_notes)} positive notes and {len(negative_notes)} negative notes")
        
        # If there are no negative notes, create some from sample notes
        if len(negative_notes) == 0 and os.path.exists(SAMPLE_NOTES_PATH):
            logger.info("No negative COVID notes found, creating from sample notes")
            
            sample_notes_df = pd.read_csv(SAMPLE_NOTES_PATH)
            sample_notes_df = sample_notes_df.sample(min(100, len(sample_notes_df)))
            
            sample_notes_df['is_covid_related'] = False
            sample_notes_df['covid_positive_likelihood'] = 0.0
            
            negative_notes = sample_notes_df
        
        # Function to assign a random note based on COVID status
        def assign_note(row):
            if row['covid_positive'] == 1:
                # Assign a positive note
                if len(positive_notes) > 0:
                    note = positive_notes.sample(1).iloc[0]
                    return note['text']
                else:
                    return "No positive COVID note available"
            else:
                # Assign a negative note
                if len(negative_notes) > 0:
                    note = negative_notes.sample(1).iloc[0]
                    return note['text']
                else:
                    return "No negative COVID note available"
        
        # Assign notes to each classification record
        class_df['note_text'] = class_df.apply(assign_note, axis=1)
        
        # Save the labeled dataset
        class_df.to_csv(LABELED_DATASET_PATH, index=False)
        logger.info(f"Saved labeled dataset with {len(class_df)} records to: {LABELED_DATASET_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating labeled dataset: {e}")
        return False

def create_mimic_note_info():
    """Create a README file with information about the notes."""
    readme_path = os.path.join(NOTE_MODULE_DIR, 'README.md')
    
    try:
        with open(readme_path, 'w') as f:
            f.write("# MIMIC-IV Clinical Notes\n\n")
            f.write("This directory contains clinical notes from the MIMIC-IV Note module.\n\n")
            
            f.write("## Files\n\n")
            f.write("- `clinical_notes_sample.csv`: A sample of clinical notes from the MIMIC-IV dataset.\n")
            f.write("- `covid_notes.csv`: Clinical notes identified as potentially related to COVID-19.\n")
            f.write("- `discharge_notes.csv.gz`: Compressed discharge summary notes.\n")
            f.write("- `discharge_detail.csv.gz`: Additional information about discharge notes.\n")
            f.write("- `radiology_detail.csv.gz`: Radiology report details.\n\n")
            
            f.write("## COVID-19 Note Classification\n\n")
            f.write("The `covid_notes.csv` file includes a `covid_positive_likelihood` column with the following values:\n\n")
            f.write("- **0.9-1.0**: Strong evidence of COVID-19 positive status (confirmed diagnosis)\n")
            f.write("- **0.7-0.9**: Probable COVID-19 case (strong clinical evidence)\n")
            f.write("- **0.3-0.7**: Suspected COVID-19 case (some symptoms or indicators)\n")
            f.write("- **0.1-0.3**: COVID-19 mentioned but not likely a positive case\n\n")
            
            f.write("## Integration with Classification Dataset\n\n")
            f.write("A labeled dataset has been created at `data/processed/covid_notes_classification.csv` that combines:\n\n")
            f.write("1. The classification features from `covid_classification_dataset.csv`\n")
            f.write("2. Clinical notes from MIMIC-IV with COVID-related content\n\n")
            f.write("This integrated dataset is used for NER extraction and BioBERT model training.\n\n")
            
            f.write("## Source\n\n")
            f.write("The original data comes from the MIMIC-IV Note module v2.2, which contains de-identified clinical notes.\n")
            f.write("For more information, see: https://physionet.org/content/mimic-iv-note/2.2/\n")
        
        logger.info(f"Created README file at: {readme_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating README file: {e}")
        return False

def main():
    """Main function to set up the MIMIC-IV Note module."""
    logger.info("Starting MIMIC-IV Note module setup")
    
    # Extract the ZIP file
    if not extract_zip_file():
        logger.error("Failed to extract ZIP file")
        return 1
    
    # Sample the notes
    if not sample_notes():
        logger.error("Failed to sample notes")
        return 1
    
    # Identify COVID-related notes
    if not identify_covid_notes():
        logger.error("Failed to identify COVID notes")
        return 1
    
    # Create labeled dataset
    if not create_labeled_dataset():
        logger.error("Failed to create labeled dataset")
        return 1
    
    # Create a README file
    if not create_mimic_note_info():
        logger.error("Failed to create README file")
        return 1
    
    logger.info("MIMIC-IV Note module setup complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())