"""MIMIC-IV Integration Module

This module provides functions to integrate MIMIC-IV data into our COVID-19 detection pipeline.
It handles loading and preprocessing the data for both NER and classification tasks.
"""

import os
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the MIMIC data in our project
MIMIC_DIR = os.path.join('data', 'external', 'mimic')
MIMIC_SOURCE_DIR = '/Users/Apexr/physionet.org/files/mimiciv/3.1'

def load_mimic_demographics():
    """Load patient demographic data from MIMIC-IV"""
    patients_path = os.path.join(MIMIC_DIR, 'patients_sample.csv')
    admissions_path = os.path.join(MIMIC_DIR, 'admissions_sample.csv')
    
    try:
        patients_df = pd.read_csv(patients_path)
        logger.info(f"Loaded {len(patients_df)} patient records")
        
        admissions_df = pd.read_csv(admissions_path)
        logger.info(f"Loaded {len(admissions_df)} admission records")
        
        # Merge to get complete demographic information
        demographics = pd.merge(patients_df, admissions_df, on='subject_id', how='inner')
        logger.info(f"Combined demographics dataset contains {len(demographics)} records")
        
        return demographics
    except Exception as e:
        logger.error(f"Error loading demographic data: {e}")
        return pd.DataFrame()

def load_mimic_diagnoses():
    """Load diagnoses data from MIMIC-IV"""
    diagnoses_path = os.path.join(MIMIC_DIR, 'relevant_diagnoses.csv')
    d_icd_path = os.path.join(MIMIC_DIR, 'd_icd_diagnoses.csv')
    
    try:
        # Check if we have the relevant diagnoses file
        if os.path.exists(diagnoses_path):
            diagnoses_df = pd.read_csv(diagnoses_path)
        else:
            logger.warning(f"Relevant diagnoses file not found: {diagnoses_path}")
            return pd.DataFrame()
        
        # Load the ICD codes dictionary
        d_icd_df = pd.read_csv(d_icd_path)
        
        # Merge to get diagnosis descriptions
        merged_df = pd.merge(diagnoses_df, d_icd_df, on='icd_code', how='left')
        logger.info(f"Loaded and merged {len(merged_df)} diagnoses with descriptions")
        
        return merged_df
    except Exception as e:
        logger.error(f"Error loading diagnoses data: {e}")
        return pd.DataFrame()

def load_mimic_lab_results():
    """Load lab results data from MIMIC-IV"""
    lab_results_path = os.path.join(MIMIC_DIR, 'relevant_labevents.csv')
    lab_sample_path = os.path.join(MIMIC_DIR, 'labevents_sample.csv')
    d_labitems_path = os.path.join(MIMIC_DIR, 'd_labitems.csv')
    
    try:
        # Check if we have the relevant lab events file
        if os.path.exists(lab_results_path):
            lab_df = pd.read_csv(lab_results_path)
            logger.info(f"Loaded {len(lab_df)} relevant lab events")
        elif os.path.exists(lab_sample_path):
            lab_df = pd.read_csv(lab_sample_path)
            logger.info(f"Loaded {len(lab_df)} sample lab events")
        else:
            logger.warning(f"No lab events file found")
            return pd.DataFrame()
        
        # Load the lab items dictionary
        if os.path.exists(d_labitems_path):
            d_labitems_df = pd.read_csv(d_labitems_path)
            
            # Merge to get lab test descriptions
            merged_df = pd.merge(lab_df, d_labitems_df, on='itemid', how='left')
            logger.info(f"Merged {len(merged_df)} lab events with descriptions")
            
            return merged_df
        else:
            logger.warning(f"Lab items dictionary not found: {d_labitems_path}")
            return lab_df
    except Exception as e:
        logger.error(f"Error loading lab results data: {e}")
        return pd.DataFrame()

def load_mimic_text_data():
    """Load text data from MIMIC-IV for NER"""
    omr_path = os.path.join(MIMIC_DIR, 'omr_sample.csv')
    
    try:
        if os.path.exists(omr_path):
            omr_df = pd.read_csv(omr_path)
            logger.info(f"Loaded {len(omr_df)} OMR records for text analysis")
            
            # Extract text columns for NER
            text_columns = [col for col in omr_df.columns if omr_df[col].dtype == 'object']
            text_df = omr_df[['subject_id', 'hadm_id'] + text_columns]
            
            return text_df
        else:
            logger.warning(f"OMR data not found: {omr_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading text data: {e}")
        return pd.DataFrame()

def get_sample_clinical_notes():
    """Get sample clinical notes for NER from MIMIC-IV"""
    # First try to load from the Note module if available
    note_module_path = os.path.join(MIMIC_DIR, 'note_module', 'clinical_notes_sample.csv')
    covid_notes_path = os.path.join(MIMIC_DIR, 'note_module', 'covid_notes.csv')
    
    if os.path.exists(covid_notes_path):
        try:
            # Prefer COVID notes for our COVID detection task
            notes_df = pd.read_csv(covid_notes_path)
            if not notes_df.empty:
                # Rename columns to match expected format
                if 'text' in notes_df.columns and 'note_text' not in notes_df.columns:
                    notes_df = notes_df.rename(columns={'text': 'note_text'})
                
                # Ensure subject_id exists
                if 'subject_id' not in notes_df.columns and 'note_id' in notes_df.columns:
                    notes_df['subject_id'] = notes_df['note_id']
                
                # If hadm_id is required but missing, add a placeholder
                if 'hadm_id' not in notes_df.columns:
                    notes_df['hadm_id'] = notes_df['subject_id']
                
                logger.info(f"Loaded {len(notes_df)} COVID-related clinical notes from MIMIC-IV Note module")
                return notes_df
        except Exception as e:
            logger.error(f"Error loading COVID notes: {e}")
    
    if os.path.exists(note_module_path):
        try:
            # If COVID notes aren't available, try the general sample
            notes_df = pd.read_csv(note_module_path)
            if not notes_df.empty:
                # Rename columns to match expected format
                if 'text' in notes_df.columns and 'note_text' not in notes_df.columns:
                    notes_df = notes_df.rename(columns={'text': 'note_text'})
                
                # Ensure subject_id exists
                if 'subject_id' not in notes_df.columns and 'note_id' in notes_df.columns:
                    notes_df['subject_id'] = notes_df['note_id']
                
                # If hadm_id is required but missing, add a placeholder
                if 'hadm_id' not in notes_df.columns:
                    notes_df['hadm_id'] = notes_df['subject_id']
                
                logger.info(f"Loaded {len(notes_df)} clinical notes from MIMIC-IV Note module")
                return notes_df
        except Exception as e:
            logger.error(f"Error loading sample notes: {e}")
    
    # Fall back to the original method if Note module files aren't available
    text_df = load_mimic_text_data()
    
    if not text_df.empty:
        # Extract text columns and combine them
        text_columns = [col for col in text_df.columns if col not in ['subject_id', 'hadm_id']]
        
        # Combine all text columns into a single note for each patient
        notes = []
        for _, row in text_df.iterrows():
            note_parts = []
            for col in text_columns:
                if isinstance(row[col], str) and len(row[col].strip()) > 0:
                    note_parts.append(f"{col}: {row[col]}")
            
            if note_parts:
                note = "\n".join(note_parts)
                notes.append({
                    'subject_id': row['subject_id'],
                    'hadm_id': row['hadm_id'],
                    'note_text': note
                })
        
        if notes:
            notes_df = pd.DataFrame(notes)
            logger.info(f"Created {len(notes_df)} clinical notes for NER from OMR data")
            return notes_df
    
    logger.warning("No suitable clinical notes found in MIMIC-IV data")
    return pd.DataFrame()

def prepare_mimic_for_classification():
    """Prepare MIMIC-IV data for the classification pipeline"""
    # Load all the necessary components
    demographics = load_mimic_demographics()
    diagnoses = load_mimic_diagnoses()
    lab_results = load_mimic_lab_results()
    
    if demographics.empty:
        logger.warning("Cannot prepare classification data without demographics")
        return pd.DataFrame()
    
    try:
        # Start with demographics as our base
        classification_data = demographics.copy()
        
        # Add a COVID flag if we have diagnoses
        if not diagnoses.empty:
            # Find COVID-specific ICD codes
            covid_diagnoses = diagnoses[diagnoses['long_title'].str.contains('COVID|coronavirus|SARS-CoV', case=False, na=False)]
            covid_patients = covid_diagnoses['subject_id'].unique().tolist()
            
            # Add COVID flag to classification data
            classification_data['covid_diagnosis'] = classification_data['subject_id'].isin(covid_patients)
            
            logger.info(f"Identified {len(covid_patients)} patients with COVID-19 diagnoses")
        
        # Add lab result features if available
        if not lab_results.empty:
            # Find COVID-specific lab tests
            covid_tests = lab_results[lab_results['label'].str.contains('COVID|coronavirus|SARS', case=False, na=False)]
            
            if not covid_tests.empty:
                # Create a pivot table of COVID test results
                covid_test_pivot = covid_tests.pivot_table(
                    index='subject_id',
                    columns='itemid',
                    values='valuenum',
                    aggfunc='mean'
                )
                
                # Rename columns to test names
                itemid_to_label = dict(zip(covid_tests['itemid'], covid_tests['label']))
                covid_test_pivot.columns = [f"test_{itemid_to_label.get(col, col)}" for col in covid_test_pivot.columns]
                
                # Reset index to make subject_id a column again
                covid_test_pivot.reset_index(inplace=True)
                
                # Merge with classification data
                classification_data = pd.merge(classification_data, covid_test_pivot, on='subject_id', how='left')
                
                logger.info(f"Added {len(covid_test_pivot.columns)-1} COVID test features to classification data")
        
        # Save the prepared data
        output_path = os.path.join(MIMIC_DIR, 'classification_data.csv')
        classification_data.to_csv(output_path, index=False)
        logger.info(f"Saved classification data with {len(classification_data)} rows and {len(classification_data.columns)} features to {output_path}")
        
        return classification_data
    except Exception as e:
        logger.error(f"Error preparing classification data: {e}")
        return pd.DataFrame()

def extract_mimic_sample_data():
    """Extract sample data from the source MIMIC-IV dataset"""
    # Make sure the target directory exists
    os.makedirs(MIMIC_DIR, exist_ok=True)
    
    # Sample sizes
    SAMPLE_FRAC = 0.05  # 5% sample
    SMALL_SAMPLE_SIZE = 1000
    
    try:
        # Extract basic patient data (required for everything else)
        logger.info("Extracting patient data...")
        patients_path = os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'patients.csv.gz')
        patients_output = os.path.join(MIMIC_DIR, 'patients_sample.csv')
        
        # Sample patients
        patients_df = pd.read_csv(patients_path, compression='gzip')
        patients_sample = patients_df.sample(frac=SAMPLE_FRAC, random_state=42)
        patients_sample.to_csv(patients_output, index=False)
        logger.info(f"Saved {len(patients_sample)} patient records to {patients_output}")
        
        # Get the subject IDs from our sample to filter other tables
        subject_ids = patients_sample['subject_id'].unique().tolist()
        
        # Extract admissions for these patients
        logger.info("Extracting admission data...")
        admissions_path = os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'admissions.csv.gz')
        admissions_output = os.path.join(MIMIC_DIR, 'admissions_sample.csv')
        
        # Filter admissions to our patient sample
        admissions_df = pd.read_csv(admissions_path, compression='gzip')
        admissions_sample = admissions_df[admissions_df['subject_id'].isin(subject_ids)]
        admissions_sample.to_csv(admissions_output, index=False)
        logger.info(f"Saved {len(admissions_sample)} admission records to {admissions_output}")
        
        # Get hadm_ids from our admissions
        hadm_ids = admissions_sample['hadm_id'].unique().tolist()
        
        # Extract diagnoses for these admissions
        logger.info("Extracting diagnosis data...")
        diagnoses_path = os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'diagnoses_icd.csv.gz')
        diagnoses_output = os.path.join(MIMIC_DIR, 'diagnoses_sample.csv')
        
        # Read diagnoses in chunks due to potential size
        chunk_size = 100000
        diagnoses_chunks = []
        
        for chunk in pd.read_csv(diagnoses_path, chunksize=chunk_size, compression='gzip'):
            # Filter to our sample
            chunk_sample = chunk[chunk['hadm_id'].isin(hadm_ids)]
            if not chunk_sample.empty:
                diagnoses_chunks.append(chunk_sample)
                
        if diagnoses_chunks:
            diagnoses_sample = pd.concat(diagnoses_chunks, ignore_index=True)
            diagnoses_sample.to_csv(diagnoses_output, index=False)
            logger.info(f"Saved {len(diagnoses_sample)} diagnosis records to {diagnoses_output}")
            
            # Also save the ICD code dictionary
            logger.info("Saving ICD code dictionary...")
            d_icd_path = os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'd_icd_diagnoses.csv.gz')
            d_icd_output = os.path.join(MIMIC_DIR, 'd_icd_diagnoses.csv')
            d_icd_df = pd.read_csv(d_icd_path, compression='gzip')
            d_icd_df.to_csv(d_icd_output, index=False)
            logger.info(f"Saved {len(d_icd_df)} ICD codes to {d_icd_output}")
        
        # Extract OMR data for text content
        logger.info("Extracting OMR text data...")
        omr_path = os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'omr.csv.gz')
        omr_output = os.path.join(MIMIC_DIR, 'omr_sample.csv')
        
        # Process OMR in chunks
        chunk_size = 10000
        omr_chunks = []
        total_rows = 0
        max_rows = SMALL_SAMPLE_SIZE
        
        for chunk in pd.read_csv(omr_path, chunksize=chunk_size, compression='gzip'):
            # Filter to our sample
            chunk_sample = chunk[chunk['subject_id'].isin(subject_ids)]
            if not chunk_sample.empty:
                omr_chunks.append(chunk_sample)
                total_rows += len(chunk_sample)
                if total_rows >= max_rows:
                    break
                    
        if omr_chunks:
            omr_sample = pd.concat(omr_chunks, ignore_index=True)
            # Limit to max rows
            if len(omr_sample) > max_rows:
                omr_sample = omr_sample.head(max_rows)
            omr_sample.to_csv(omr_output, index=False)
            logger.info(f"Saved {len(omr_sample)} OMR records to {omr_output}")
        
        # Extract lab data dictionary 
        logger.info("Saving lab item dictionary...")
        d_labitems_path = os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'd_labitems.csv.gz')
        d_labitems_output = os.path.join(MIMIC_DIR, 'd_labitems.csv')
        d_labitems_df = pd.read_csv(d_labitems_path, compression='gzip')
        d_labitems_df.to_csv(d_labitems_output, index=False)
        logger.info(f"Saved {len(d_labitems_df)} lab items to {d_labitems_output}")
        
        # Extract a sample of lab events
        logger.info("Extracting lab events data...")
        labevents_path = os.path.join(MIMIC_SOURCE_DIR, 'hosp', 'labevents.csv.gz')
        labevents_output = os.path.join(MIMIC_DIR, 'labevents_sample.csv')
        
        # Process in chunks
        chunk_size = 100000
        lab_chunks = []
        total_rows = 0
        max_rows = SMALL_SAMPLE_SIZE
        
        for chunk in pd.read_csv(labevents_path, chunksize=chunk_size, compression='gzip'):
            # Filter to our sample
            chunk_sample = chunk[chunk['subject_id'].isin(subject_ids)]
            if not chunk_sample.empty:
                lab_chunks.append(chunk_sample)
                total_rows += len(chunk_sample)
                if total_rows >= max_rows:
                    break
                    
        if lab_chunks:
            lab_sample = pd.concat(lab_chunks, ignore_index=True)
            # Limit to max rows
            if len(lab_sample) > max_rows:
                lab_sample = lab_sample.head(max_rows)
            lab_sample.to_csv(labevents_output, index=False)
            logger.info(f"Saved {len(lab_sample)} lab events to {labevents_output}")
            
        logger.info("MIMIC-IV sample data extraction complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting MIMIC-IV sample data: {e}")
        return False
        
if __name__ == "__main__":
    # Extract data from source if needed
    if not os.path.exists(os.path.join(MIMIC_DIR, 'patients_sample.csv')):
        logger.info("No existing MIMIC samples found. Extracting from source...")
        extract_mimic_sample_data()
    
    # Test loading functions
    demographics = load_mimic_demographics()
    diagnoses = load_mimic_diagnoses()
    lab_results = load_mimic_lab_results()
    text_data = load_mimic_text_data()
    
    # Prepare data for our pipeline
    classification_data = prepare_mimic_for_classification()
    notes = get_sample_clinical_notes()
    
    print("\nData Summary:")
    print(f"Demographics: {len(demographics)} records")
    print(f"Diagnoses: {len(diagnoses)} records")
    print(f"Lab Results: {len(lab_results)} records")
    print(f"Text Data: {len(text_data)} records")
    print(f"Classification Data: {len(classification_data)} records with {len(classification_data.columns) if not classification_data.empty else 0} features")
    print(f"Clinical Notes: {len(notes)} notes")