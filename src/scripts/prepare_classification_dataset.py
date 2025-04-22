#!/usr/bin/env python
"""
Script to prepare the COVID-19 classification dataset by combining
CDC Case Surveillance data and MIMIC-IV clinical data.

This script performs the same tasks as the notebook but is more
robust when dealing with memory constraints or kernel issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification_dataset_prep.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the src directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Now import the necessary modules
try:
    from src.ner_extraction import extract_entities_from_text, format_entities_for_bert
    from src.mimic_integration import get_sample_clinical_notes
    NER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NER dependencies not available, will not include text features: {e}")
    NER_AVAILABLE = False

# Define directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
MIMIC_DIR = os.path.join(EXTERNAL_DIR, 'mimic')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Create output directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a symbolic link to the classification data in the MIMIC directory if it exists
MIMIC_CLASS_FILE = os.path.join(MIMIC_DIR, 'classification_data.csv')
PROCESSED_CLASS_FILE = os.path.join(PROCESSED_DIR, 'mimic_features.csv')
if os.path.exists(MIMIC_CLASS_FILE) and not os.path.exists(PROCESSED_CLASS_FILE):
    try:
        # Copy the file instead of creating a symlink for better compatibility
        import shutil
        shutil.copy2(MIMIC_CLASS_FILE, PROCESSED_CLASS_FILE)
        logger.info(f"Copied MIMIC classification data to {PROCESSED_CLASS_FILE}")
    except Exception as e:
        logger.error(f"Error copying MIMIC classification data: {e}")
        logger.debug(traceback.format_exc())

def load_cdc_data(sample_size=None):
    """
    Load CDC Case Surveillance data.
    
    Parameters:
    -----------
    sample_size : int, optional
        Number of rows to load (for memory-constrained environments)
    
    Returns:
    --------
    pandas.DataFrame or None
    """
    cdc_file = os.path.join(EXTERNAL_DIR, 'covid19_case_surveillance.csv')
    
    if not os.path.exists(cdc_file):
        logger.warning(f"CDC data file not found: {cdc_file}")
        return None
    
    logger.info(f"Loading CDC data from {cdc_file}")
    try:
        # Handle whitespace in column names
        if sample_size:
            df = pd.read_csv(cdc_file, nrows=sample_size, skipinitialspace=True)
        else:
            df = pd.read_csv(cdc_file, skipinitialspace=True)
        
        # Strip whitespace from columns
        df.columns = df.columns.str.strip()
        
        logger.info(f"Loaded {len(df)} CDC records with {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading CDC data: {e}")
        logger.debug(traceback.format_exc())
        return None

def load_mimic_data():
    """
    Load MIMIC-IV data files.
    
    Returns:
    --------
    tuple: (patients, diagnoses, icd_codes, labs) as pandas DataFrames or None
    """
    # File paths
    patients_file = os.path.join(MIMIC_DIR, 'patients_sample.csv')
    diagnoses_file = os.path.join(MIMIC_DIR, 'relevant_diagnoses.csv')
    icd_file = os.path.join(MIMIC_DIR, 'd_icd_diagnoses.csv')
    labs_file = os.path.join(MIMIC_DIR, 'relevant_labevents.csv')
    
    # Initialize results
    patients = None
    diagnoses = None
    icd_codes = None
    labs = None
    
    # Load patient data
    if os.path.exists(patients_file):
        try:
            patients = pd.read_csv(patients_file)
            logger.info(f"Loaded {len(patients)} MIMIC patient records")
        except Exception as e:
            logger.error(f"Error loading MIMIC patients data: {e}")
    else:
        logger.warning(f"MIMIC patients file not found: {patients_file}")
    
    # Load diagnoses data
    if os.path.exists(diagnoses_file):
        try:
            diagnoses = pd.read_csv(diagnoses_file)
            logger.info(f"Loaded {len(diagnoses)} MIMIC diagnosis records")
        except Exception as e:
            logger.error(f"Error loading MIMIC diagnoses data: {e}")
    else:
        logger.warning(f"MIMIC diagnoses file not found: {diagnoses_file}")
    
    # Load ICD codes dictionary
    if os.path.exists(icd_file):
        try:
            icd_codes = pd.read_csv(icd_file)
            logger.info(f"Loaded {len(icd_codes)} ICD code descriptions")
        except Exception as e:
            logger.error(f"Error loading ICD codes data: {e}")
    else:
        logger.warning(f"ICD codes file not found: {icd_file}")
    
    # Load lab results
    if os.path.exists(labs_file):
        try:
            labs = pd.read_csv(labs_file)
            logger.info(f"Loaded {len(labs)} lab result records")
        except Exception as e:
            logger.error(f"Error loading lab results data: {e}")
    else:
        logger.warning(f"Lab results file not found: {labs_file}")
    
    return patients, diagnoses, icd_codes, labs

def prepare_cdc_features(df):
    """
    Process CDC data for classification.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        CDC data
    
    Returns:
    --------
    pandas.DataFrame or None
    """
    if df is None or df.empty:
        logger.warning("No CDC data available for feature preparation")
        return None
    
    logger.info("Preparing CDC data for classification")
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Create target variable (1 for confirmed case, 0 for probable or missing)
    if 'current_status' in result.columns:
        result['covid_positive'] = result['current_status'].apply(
            lambda x: 1 if x and 'confirmed' in str(x).lower() else 0
        )
        logger.info(f"Created target variable: {result['covid_positive'].sum()} positive cases")
    else:
        logger.warning("Unable to create target variable: 'current_status' column missing")
        return None
    
    # Handle missing values
    for col in ['sex', 'age_group', 'hosp_yn', 'icu_yn', 'death_yn', 'medcond_yn']:
        if col in result.columns:
            result[col] = result[col].replace('Missing', np.nan)
    
    # Convert Yes/No columns to 1/0
    for col in ['hosp_yn', 'icu_yn', 'death_yn', 'medcond_yn']:
        if col in result.columns:
            result[col] = result[col].map({'Yes': 1, 'No': 0, 'Unknown': np.nan})
    
    # Create dummy variables for categorical columns
    if 'sex' in result.columns:
        sex_dummies = pd.get_dummies(result['sex'], prefix='sex')
        result = pd.concat([result, sex_dummies], axis=1)
    
    if 'age_group' in result.columns:
        age_dummies = pd.get_dummies(result['age_group'], prefix='age')
        result = pd.concat([result, age_dummies], axis=1)
    
    # Keep relevant columns only
    cols_to_keep = ['covid_positive', 'hosp_yn', 'icu_yn', 'death_yn', 'medcond_yn']
    
    # Add dummy columns
    cols_to_keep.extend([col for col in result.columns if col.startswith('sex_')])
    cols_to_keep.extend([col for col in result.columns if col.startswith('age_')])
    
    # Filter to columns that exist
    cols_to_keep = [col for col in cols_to_keep if col in result.columns]
    
    # Add record ID
    result['record_id'] = ['CDC_' + str(i) for i in range(len(result))]
    cols_to_keep.insert(0, 'record_id')  # Add to beginning
    
    # Return dataset with selected columns
    final_df = result[cols_to_keep]
    logger.info(f"Prepared CDC dataset with {len(final_df)} records and {len(cols_to_keep)} features")
    return final_df

def process_clinical_notes_with_ner():
    """
    Process clinical notes from MIMIC with NER to extract features.
    
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with NER features by patient ID
    """
    if not NER_AVAILABLE:
        logger.warning("NER dependencies not available, skipping text feature extraction")
        return None
    
    try:
        # Get clinical notes
        notes_df = get_sample_clinical_notes()
        
        if notes_df is None or notes_df.empty:
            logger.warning("No clinical notes available for NER")
            return None
        
        logger.info(f"Processing {len(notes_df)} clinical notes with NER")
        
        # Process each note with NER
        all_features = []
        
        for _, row in notes_df.iterrows():
            subject_id = row['subject_id']
            hadm_id = row['hadm_id']
            note_text = row['note_text']
            
            # Extract entities using rule-based NER
            try:
                entities = extract_entities_from_text(note_text, method="rule")
                
                # Count entities by type
                symptom_count = len(entities.get('SYMPTOM', []))
                time_count = len(entities.get('TIME', []))
                severity_count = len(entities.get('SEVERITY', []))
                
                # Extract specific symptoms (create flags for common symptoms)
                symptoms = [entity['text'].lower() for entity in entities.get('SYMPTOM', [])]
                
                # Common COVID-19 symptoms as binary features
                common_symptoms = [
                    "fever", "cough", "shortness of breath", "difficulty breathing",
                    "fatigue", "loss of taste", "loss of smell", "sore throat"
                ]
                
                symptom_features = {}
                for symptom in common_symptoms:
                    has_symptom = any(symptom in s for s in symptoms)
                    symptom_features[f"has_{symptom.replace(' ', '_')}"] = int(has_symptom)
                
                # Create record
                feature_record = {
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'ner_symptom_count': symptom_count,
                    'ner_time_count': time_count,
                    'ner_severity_count': severity_count,
                    **symptom_features
                }
                
                all_features.append(feature_record)
                
            except Exception as e:
                logger.error(f"Error processing note for subject {subject_id}: {e}")
                logger.debug(traceback.format_exc())
        
        if all_features:
            # Convert to DataFrame
            features_df = pd.DataFrame(all_features)
            
            # Group by subject_id and aggregate
            # For binary features, use max (1 if any note has the symptom)
            # For counts, use sum across all notes
            agg_dict = {'ner_symptom_count': 'sum', 'ner_time_count': 'sum', 'ner_severity_count': 'sum'}
            
            # Add max aggregation for all binary symptom features
            for col in features_df.columns:
                if col.startswith('has_'):
                    agg_dict[col] = 'max'
            
            # Group by subject_id
            grouped_features = features_df.groupby('subject_id').agg(agg_dict).reset_index()
            
            # Create record_id to match with other data
            grouped_features['record_id'] = ['MIMIC_' + str(sid) for sid in grouped_features['subject_id']]
            
            logger.info(f"Created NER features for {len(grouped_features)} patients")
            
            # Save the results for reference
            ner_output_path = os.path.join(OUTPUT_DIR, 'mimic_ner_results.json')
            with open(ner_output_path, 'w') as f:
                json.dump(all_features, f, indent=2)
            logger.info(f"Saved detailed NER results to {ner_output_path}")
            
            return grouped_features
        
        return None
        
    except Exception as e:
        logger.error(f"Error in NER processing: {e}")
        logger.debug(traceback.format_exc())
        return None

def prepare_mimic_features(patients, diagnoses, icd_codes, labs):
    """
    Process MIMIC data for classification.
    
    Parameters:
    -----------
    patients : pandas.DataFrame
        Patient demographics
    diagnoses : pandas.DataFrame
        Diagnoses with ICD codes
    icd_codes : pandas.DataFrame
        ICD code descriptions
    labs : pandas.DataFrame
        Laboratory test results
    
    Returns:
    --------
    pandas.DataFrame or None
    """
    # Check if pre-generated classification data exists
    mimic_class_file = os.path.join(PROCESSED_DIR, 'mimic_features.csv')
    if os.path.exists(mimic_class_file):
        try:
            logger.info(f"Loading pre-generated MIMIC classification data from {mimic_class_file}")
            df = pd.read_csv(mimic_class_file)
            
            # Ensure we have the covid_positive column
            if 'covid_diagnosis' in df.columns and 'covid_positive' not in df.columns:
                df['covid_positive'] = df['covid_diagnosis'].map({True: 1, 'True': 1, False: 0, 'False': 0})
                logger.info(f"Converted covid_diagnosis to covid_positive column")
            
            # Create record ID if not present
            if 'record_id' not in df.columns and 'subject_id' in df.columns:
                df['record_id'] = ['MIMIC_' + str(id) for id in df['subject_id']]
            
            # Process demographic features if not already processed
            if 'gender' in df.columns and not any(col.startswith('gender_') for col in df.columns):
                gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
                df = pd.concat([df, gender_dummies], axis=1)
                logger.info("Added gender dummy variables")
            
            # Select columns to keep
            cols_to_keep = ['record_id', 'covid_positive']
            
            # Add demographic columns
            gender_cols = [col for col in df.columns if col.startswith('gender_')]
            cols_to_keep.extend(gender_cols)
            
            # Add any other relevant columns
            for col in df.columns:
                if col.startswith('lab_') or col in ['anchor_age']:
                    cols_to_keep.append(col)
            
            # Select columns that exist
            cols_to_keep = [col for col in cols_to_keep if col in df.columns]
            
            # Return results
            final_df = df[cols_to_keep]
            logger.info(f"Prepared MIMIC dataset with {len(final_df)} records and {len(cols_to_keep)} features")
            return final_df
            
        except Exception as e:
            logger.error(f"Error loading pre-generated MIMIC classification data: {e}")
            logger.debug(traceback.format_exc())
            # Continue with normal processing
    
    if patients is None or patients.empty:
        logger.warning("No MIMIC patient data available for feature preparation")
        return None
    
    logger.info("Preparing MIMIC data for classification (from raw data)")
    
    # Start with patient data
    result = patients.copy()
    
    # Add COVID-19 flag based on diagnoses if available
    if diagnoses is not None and not diagnoses.empty and icd_codes is not None and not icd_codes.empty:
        try:
            # Merge diagnoses with ICD codes
            merged = pd.merge(diagnoses, icd_codes, on='icd_code', how='left')
            
            # Find COVID-related diagnoses
            covid_codes = merged[merged['long_title'].str.contains(
                'COVID|coronavirus|SARS-CoV', case=False, na=False
            )]
            
            # Get patients with COVID diagnoses
            covid_patients = covid_codes['subject_id'].unique()
            
            # Add flag to result
            result['covid_positive'] = result['subject_id'].isin(covid_patients).astype(int)
            logger.info(f"Found {len(covid_patients)} patients with COVID-19 diagnoses")
        except Exception as e:
            logger.error(f"Error processing diagnoses: {e}")
            logger.debug(traceback.format_exc())
            result['covid_positive'] = 0
    else:
        # If no diagnoses data, assume all negative
        result['covid_positive'] = 0
        logger.warning("No diagnoses data available, assuming all patients are COVID-19 negative")
    
    # Add lab data if available
    if labs is not None and not labs.empty:
        try:
            # Create pivot table with lab results
            lab_pivot = labs.pivot_table(
                index='subject_id',
                columns='itemid',
                values='valuenum',
                aggfunc='mean'
            )
            
            # Rename columns
            lab_pivot.columns = [f'lab_{col}' for col in lab_pivot.columns]
            lab_pivot.reset_index(inplace=True)
            
            # Merge with patient data
            result = pd.merge(result, lab_pivot, on='subject_id', how='left')
            logger.info(f"Added {len(lab_pivot.columns)-1} lab features")
        except Exception as e:
            logger.error(f"Error adding lab data: {e}")
            logger.debug(traceback.format_exc())
    
    # Process demographic features
    if 'gender' in result.columns:
        # Create dummy variables
        gender_dummies = pd.get_dummies(result['gender'], prefix='gender')
        result = pd.concat([result, gender_dummies], axis=1)
    
    # Create record ID
    result['record_id'] = ['MIMIC_' + str(id) for id in result['subject_id']]
    
    # Select columns to keep
    cols_to_keep = ['record_id', 'covid_positive']
    
    # Add demographic columns
    gender_cols = [col for col in result.columns if col.startswith('gender_')]
    cols_to_keep.extend(gender_cols)
    
    # Add age if available
    if 'anchor_age' in result.columns:
        cols_to_keep.append('anchor_age')
    
    # Add lab columns
    lab_cols = [col for col in result.columns if col.startswith('lab_')]
    cols_to_keep.extend(lab_cols)
    
    # Select columns that exist
    cols_to_keep = [col for col in cols_to_keep if col in result.columns]
    
    # Return results
    final_df = result[cols_to_keep]
    logger.info(f"Prepared MIMIC dataset with {len(final_df)} records and {len(cols_to_keep)} features")
    return final_df

def create_master_dataset(cdc_features, mimic_features, ner_features=None):
    """
    Combine CDC and MIMIC features into a master dataset.
    
    Parameters:
    -----------
    cdc_features : pandas.DataFrame
        Processed CDC data
    mimic_features : pandas.DataFrame
        Processed MIMIC data
    ner_features : pandas.DataFrame
        Features extracted from NER on clinical notes
    
    Returns:
    --------
    pandas.DataFrame or None
    """
    # List of datasets to combine
    datasets = []
    
    if cdc_features is not None and not cdc_features.empty:
        datasets.append(cdc_features)
        logger.info(f"Adding {len(cdc_features)} CDC records to master dataset")
    
    if mimic_features is not None and not mimic_features.empty:
        # If we have NER features, merge them with MIMIC features first
        if ner_features is not None and not ner_features.empty:
            try:
                # Merge on record_id
                mimic_with_ner = pd.merge(mimic_features, ner_features, on='record_id', how='left')
                logger.info(f"Added NER features to {len(mimic_features)} MIMIC records")
                datasets.append(mimic_with_ner)
            except Exception as e:
                logger.error(f"Error merging NER features with MIMIC data: {e}")
                logger.debug(traceback.format_exc())
                datasets.append(mimic_features)
        else:
            datasets.append(mimic_features)
        
        logger.info(f"Adding {len(mimic_features)} MIMIC records to master dataset")
    
    # Combine datasets
    if datasets:
        try:
            master_df = pd.concat(datasets, axis=0, ignore_index=True)
            logger.info(f"Created master dataset with {len(master_df)} records and {len(master_df.columns)} features")
            
            # Check target distribution
            positive_count = master_df['covid_positive'].sum()
            positive_pct = (positive_count / len(master_df)) * 100
            logger.info(f"COVID-19 positive cases: {positive_count} ({positive_pct:.2f}%)")
            logger.info(f"COVID-19 negative cases: {len(master_df) - positive_count} ({100 - positive_pct:.2f}%)")
            
            return master_df
        except Exception as e:
            logger.error(f"Error creating master dataset: {e}")
            logger.debug(traceback.format_exc())
            return None
    else:
        logger.warning("No datasets available to create master dataset")
        return None

def save_dataset(df, filename):
    """
    Save dataset to CSV file.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to save
    filename : str
        Output filename
    
    Returns:
    --------
    bool
        Success status
    """
    if df is None or df.empty:
        logger.warning("Cannot save empty dataset")
        return False
    
    output_path = os.path.join(PROCESSED_DIR, filename)
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved dataset to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        logger.debug(traceback.format_exc())
        return False

def main():
    """Main function to prepare classification dataset."""
    logger.info("Starting COVID-19 classification dataset preparation")
    
    # Step 1: Load data sources
    logger.info("Step 1: Loading data sources")
    
    # Load CDC data (with option to limit rows for memory-constrained environments)
    sample_size = None  # Set to a number like 10000 if memory is limited
    cdc_df = load_cdc_data(sample_size)
    
    # Load MIMIC data
    mimic_data = load_mimic_data()
    mimic_patients, mimic_diagnoses, mimic_icd, mimic_labs = mimic_data
    
    # Step 2: Prepare features for classification
    logger.info("Step 2: Preparing features for classification")
    
    # Process CDC data
    cdc_features = prepare_cdc_features(cdc_df)
    
    # Process MIMIC data
    mimic_features = None
    if mimic_patients is not None:
        mimic_features = prepare_mimic_features(mimic_patients, mimic_diagnoses, mimic_icd, mimic_labs)
    
    # Step 3: Process clinical notes with NER to extract text features
    logger.info("Step 3: Processing clinical notes with NER")
    ner_features = process_clinical_notes_with_ner()
    
    # Step 4: Create master dataset
    logger.info("Step 4: Creating master dataset")
    master_df = create_master_dataset(cdc_features, mimic_features, ner_features)
    
    # Step 5: Save dataset
    logger.info("Step 5: Saving dataset")
    if master_df is not None:
        save_dataset(master_df, 'covid_classification_dataset.csv')
        
        # Also save individual features for reference
        if cdc_features is not None:
            save_dataset(cdc_features, 'cdc_features.csv')
        if mimic_features is not None:
            save_dataset(mimic_features, 'mimic_features.csv')
        if ner_features is not None:
            save_dataset(ner_features, 'ner_features.csv')
    
    logger.info("Classification dataset preparation complete")
    
    # Report statistics
    if master_df is not None:
        print("\nClassification Dataset Summary:")
        print(f"- Total records: {len(master_df)}")
        print(f"- Features: {len(master_df.columns)}")
        print(f"- COVID-19 positive: {master_df['covid_positive'].sum()} ({master_df['covid_positive'].sum() / len(master_df) * 100:.2f}%)")
        print(f"- COVID-19 negative: {len(master_df) - master_df['covid_positive'].sum()} ({(1 - master_df['covid_positive'].sum() / len(master_df)) * 100:.2f}%)")
        
        # Print NER-specific stats if available
        ner_cols = [col for col in master_df.columns if col.startswith('ner_') or col.startswith('has_')]
        if ner_cols:
            print(f"- NER features: {len(ner_cols)}")
            print("  NER features included:")
            for col in ner_cols[:10]:  # Show up to 10 features
                print(f"    - {col}")
            if len(ner_cols) > 10:
                print(f"    - ... and {len(ner_cols) - 10} more")
        
        print(f"\nDataset saved to: {os.path.join(PROCESSED_DIR, 'covid_classification_dataset.csv')}")
    else:
        print("\nFailed to create classification dataset. Check the logs for details.")

if __name__ == "__main__":
    main()