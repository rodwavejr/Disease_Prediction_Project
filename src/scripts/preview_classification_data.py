#!/usr/bin/env python
"""
Script to preview the classification dataset features and structure.
Shows if NER features from clinical notes are properly integrated.
"""

import os
import csv
import json
from collections import Counter

# Define directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

def count_lines(file_path):
    """Count lines in a file."""
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def preview_csv(file_path, max_rows=5):
    """Preview a CSV file without pandas."""
    print(f"\nPreview of {file_path}:")
    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            print(f"Columns ({len(header)}):")
            
            # Check for NER features
            ner_cols = [col for col in header if col.startswith('ner_') or col.startswith('has_')]
            if ner_cols:
                print("\nNER FEATURES FOUND! Text data is integrated properly.")
                print(f"Found {len(ner_cols)} NER-derived features:")
                for col in ner_cols:
                    print(f"  - {col}")
                print("\nOther columns:")
            else:
                print("\nWARNING: No NER features found! Text data is NOT integrated.")
                print("All columns:")
            
            # Group columns by category
            grouped_cols = {
                'ID': [col for col in header if col in ['record_id', 'subject_id']],
                'Target': [col for col in header if col in ['covid_positive', 'covid_diagnosis']],
                'Demographics': [col for col in header if col.startswith('gender_') or col.startswith('sex_') or col.startswith('age_')],
                'Lab Results': [col for col in header if col.startswith('lab_')],
                'Hospital Metrics': [col for col in header if col in ['hosp_yn', 'icu_yn', 'death_yn']],
                'NER Metrics': [col for col in header if col.startswith('ner_')],
                'Symptom Indicators': [col for col in header if col.startswith('has_')],
                'Other': []
            }
            
            # Add other columns to "Other" category
            for col in header:
                if not any(col in group for group in grouped_cols.values()):
                    grouped_cols['Other'].append(col)
            
            # Print groups
            for group, cols in grouped_cols.items():
                if cols:
                    print(f"  {group} ({len(cols)}):")
                    for col in cols[:5]:  # Show only first 5
                        print(f"    - {col}")
                    if len(cols) > 5:
                        print(f"    - ...and {len(cols)-5} more")
            
            print("\nSample data:")
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                if len(row) > 5:
                    print(f"  Row {i+1}: {row[:5]}... (truncated)")
                else:
                    print(f"  Row {i+1}: {row}")
                
            # Get approximate line count
            line_count = count_lines(file_path)
            print(f"\nApprox. {line_count} records in file")
            
    except Exception as e:
        print(f"Error previewing file: {e}")

def check_ner_results():
    """Check if NER results file exists and contains processed notes."""
    ner_results_path = os.path.join(OUTPUT_DIR, 'mimic_ner_results.json')
    if os.path.exists(ner_results_path):
        try:
            # Just load the file to get the count of records
            with open(ner_results_path, 'r') as f:
                ner_results = json.load(f)
            print(f"\nNER results file found with {len(ner_results)} processed clinical notes")
            
            # Check for expected features
            if len(ner_results) > 0:
                print("\nSample NER extraction:")
                sample = ner_results[0]
                for key, value in sample.items():
                    if key in ['subject_id', 'hadm_id']:
                        print(f"  {key}: {value}")
                    elif isinstance(value, int):
                        print(f"  {key}: {value}")
                    elif key.startswith('has_'):
                        print(f"  {key}: {value}")
                    
            return True
        except Exception as e:
            print(f"Error reading NER results file: {e}")
            return False
    else:
        print("\nNo NER results file found (mimic_ner_results.json)")
        print("This suggests the NER extraction process hasn't run.")
        return False

def check_classification_dataset():
    """Check if the classification dataset exists and has NER features."""
    dataset_path = os.path.join(PROCESSED_DIR, 'covid_classification_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"\nClassification dataset not found at {dataset_path}")
        print("Please run prepare_classification_dataset.py first.")
        return False
    
    # Check NER features separately
    ner_features_path = os.path.join(PROCESSED_DIR, 'ner_features.csv')
    if os.path.exists(ner_features_path):
        print(f"\nNER features file found at {ner_features_path}")
        preview_csv(ner_features_path, max_rows=3)
    else:
        print(f"\nNo separate NER features file found.")
    
    # Preview the master dataset
    print(f"\nChecking classification dataset at {dataset_path}")
    preview_csv(dataset_path, max_rows=3)
    
    return True

def main():
    """Main function."""
    print("\n" + "="*60)
    print("COVID-19 CLASSIFICATION DATASET STRUCTURE ANALYSIS")
    print("="*60)
    print("\nThis script checks if NER features from clinical notes are")
    print("properly integrated into the classification dataset.")
    
    # Ensure directories exist
    if not os.path.exists(PROCESSED_DIR):
        print(f"Processed data directory not found at {PROCESSED_DIR}")
        print("Please run prepare_classification_dataset.py first.")
        return
    
    # Check NER results
    print("\n" + "="*60)
    print("CHECKING NER RESULTS")
    print("="*60)
    ner_results_exist = check_ner_results()
    
    # Check classification dataset
    print("\n" + "="*60)
    print("CHECKING CLASSIFICATION DATASET")
    print("="*60)
    dataset_exists = check_classification_dataset()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not ner_results_exist and not dataset_exists:
        print("\nBoth NER results and classification dataset are missing.")
        print("Please run the following scripts in order:")
        print("1. python test_mimic_ner.py (to extract entities from clinical notes)")
        print("2. python prepare_classification_dataset.py (to create the integrated dataset)")
    elif not ner_results_exist and dataset_exists:
        print("\nClassification dataset exists but NER results are missing.")
        print("This suggests the dataset may not include text-derived features.")
        print("Run: python test_mimic_ner.py")
        print("Then: python prepare_classification_dataset.py")
    elif ner_results_exist and not dataset_exists:
        print("\nNER results exist but classification dataset is missing.")
        print("Run: python prepare_classification_dataset.py")
    else:
        print("\nBoth NER results and classification dataset exist.")
        print("To evaluate the dataset for modeling, run:")
        print("python analyze_classification_dataset.py")

if __name__ == "__main__":
    main()