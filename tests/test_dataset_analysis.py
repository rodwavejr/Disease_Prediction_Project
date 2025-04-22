#!/usr/bin/env python
"""
Simple script to test the COVID-19 classification dataset analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main function to test loading and basic analysis of the dataset."""
    print("Starting COVID-19 classification dataset analysis test")
    
    # Define directories
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
    MIMIC_DIR = os.path.join(EXTERNAL_DIR, 'mimic')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    
    # Check if the classification data exists
    mimic_class_file = os.path.join(MIMIC_DIR, 'classification_data.csv')
    
    if os.path.exists(mimic_class_file):
        print(f"Loading MIMIC classification data from {mimic_class_file}")
        try:
            df = pd.read_csv(mimic_class_file)
            print(f"Loaded dataset with {len(df)} records and {len(df.columns)} features")
            
            # Display basic statistics
            print("\nBasic Statistics:")
            print(f"- Records: {len(df)}")
            print(f"- Features: {len(df.columns)}")
            
            # Check if covid_diagnosis column exists
            if 'covid_diagnosis' in df.columns:
                covid_counts = df['covid_diagnosis'].value_counts()
                print("\nCOVID-19 Distribution:")
                for value, count in covid_counts.items():
                    print(f"- {value}: {count} ({count/len(df)*100:.2f}%)")
            
            # Check demographic distributions
            if 'gender' in df.columns:
                gender_counts = df['gender'].value_counts()
                print("\nGender Distribution:")
                for gender, count in gender_counts.items():
                    print(f"- {gender}: {count} ({count/len(df)*100:.2f}%)")
            
            if 'anchor_age' in df.columns:
                print("\nAge Statistics:")
                print(f"- Mean age: {df['anchor_age'].mean():.1f}")
                print(f"- Median age: {df['anchor_age'].median():.1f}")
                print(f"- Age range: {df['anchor_age'].min()} to {df['anchor_age'].max()}")
            
            print("\nTest completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            return False
    else:
        print(f"Classification data file not found: {mimic_class_file}")
        return False

if __name__ == "__main__":
    main()