#!/usr/bin/env python
"""
Quick analysis of the COVID-19 classification dataset.
This script performs key analysis tasks and creates visualizations 
without requiring the full analysis pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

# Configure matplotlib settings simply
plt.style.use('default')

# Define directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
MIMIC_DIR = os.path.join(EXTERNAL_DIR, 'mimic')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')

# Create output directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(os.path.join(ANALYSIS_DIR, 'plots'), exist_ok=True)

def load_dataset():
    """Load the classification dataset from any available location."""
    # Check multiple possible locations for the dataset
    dataset_paths = [
        os.path.join(PROCESSED_DIR, 'covid_classification_dataset.csv'),
        os.path.join(PROCESSED_DIR, 'mimic_features.csv'),
        os.path.join(EXTERNAL_DIR, 'mimic', 'classification_data.csv')
    ]
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"Loading classification dataset from {dataset_path}")
            try:
                df = pd.read_csv(dataset_path)
                
                # If we have covid_diagnosis but not covid_positive, create it
                if 'covid_diagnosis' in df.columns and 'covid_positive' not in df.columns:
                    df['covid_positive'] = df['covid_diagnosis'].map({True: 1, 'True': 1, False: 0, 'False': 0})
                    print(f"Converted covid_diagnosis to covid_positive column")
                
                # Create record_id if not present
                if 'record_id' not in df.columns and 'subject_id' in df.columns:
                    df['record_id'] = ['MIMIC_' + str(id) for id in df['subject_id']]
                    print(f"Created record_id column from subject_id")
                
                print(f"Loaded dataset with {len(df)} records and {len(df.columns)} features")
                return df
            except Exception as e:
                print(f"Error loading dataset from {dataset_path}: {e}")
                traceback.print_exc()
    
    print(f"Classification dataset not found in any expected locations")
    return None

def analyze_target_distribution(df):
    """Analyze the distribution of the target variable."""
    target_col = 'covid_positive'
    if target_col not in df.columns and 'covid_diagnosis' in df.columns:
        target_col = 'covid_diagnosis'
    
    if target_col not in df.columns:
        print("Target variable not found in dataset")
        return None
    
    print("Analyzing target distribution")
    
    # Calculate distribution
    target_counts = df[target_col].value_counts().sort_index()
    target_pct = (target_counts / len(df) * 100).round(2)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    if target_col == 'covid_diagnosis':
        ax = sns.countplot(x=target_col, data=df, palette=['#3498db', '#e74c3c'])
    else:
        ax = sns.countplot(x=target_col, data=df, palette=['#3498db', '#e74c3c'])
    
    plt.title('Distribution of COVID-19 Cases', fontsize=14)
    plt.xlabel('COVID-19 Positive Status', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'target_distribution.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Print statistics
    print("\nCOVID-19 Distribution:")
    for i, (value, count) in enumerate(target_counts.items()):
        print(f"- {value}: {count} ({target_pct.iloc[i]}%)")
    
    ratio = target_counts.iloc[0] / max(1, target_counts.iloc[-1])
    print(f"Class ratio: {ratio:.1f}:1")
    if ratio > 3:
        print("Warning: Significant class imbalance detected")

def analyze_demographics(df):
    """Analyze demographic features."""
    print("\nAnalyzing demographic features")
    
    if 'gender' in df.columns:
        plt.figure(figsize=(8, 6))
        gender_counts = df['gender'].value_counts()
        sns.barplot(x=gender_counts.index, y=gender_counts.values)
        plt.title('Gender Distribution', fontsize=14)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'plots', 'gender_distribution.png'))
        plt.close()
        
        print("\nGender Distribution:")
        for gender, count in gender_counts.items():
            print(f"- {gender}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'anchor_age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['anchor_age'], kde=True, bins=20)
        plt.title('Age Distribution', fontsize=14)
        plt.xlabel('Age', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'plots', 'age_distribution.png'))
        plt.close()
        
        print("\nAge Statistics:")
        print(f"- Mean: {df['anchor_age'].mean():.1f}")
        print(f"- Median: {df['anchor_age'].median()}")
        print(f"- Range: {df['anchor_age'].min()} - {df['anchor_age'].max()}")
    
    # If we have both gender and target, analyze the relationship
    target_col = 'covid_positive' if 'covid_positive' in df.columns else 'covid_diagnosis'
    if 'gender' in df.columns and target_col in df.columns:
        plt.figure(figsize=(8, 6))
        gender_positive = df.groupby('gender')[target_col].mean() * 100
        sns.barplot(x=gender_positive.index, y=gender_positive.values)
        plt.title('COVID-19 Positive Rate by Gender', fontsize=14)
        plt.ylabel('Positive Rate (%)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'plots', 'gender_positive_rate.png'))
        plt.close()
        
        print("\nCOVID-19 Positive Rate by Gender:")
        for gender, rate in gender_positive.items():
            print(f"- {gender}: {rate:.1f}%")

def analyze_features(df):
    """Analyze key features in the dataset."""
    print("\nAnalyzing dataset features")
    
    # Exclude non-predictive columns
    exclude_cols = ['record_id', 'subject_id', 'covid_positive', 'covid_diagnosis']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nFeature summary ({len(feature_cols)} features):")
    for col in feature_cols[:10]:  # Limit to first 10 features
        if df[col].dtype in ['int64', 'float64']:
            print(f"- {col}: Mean={df[col].mean():.2f}, Median={df[col].median()}")
        else:
            print(f"- {col}: {df[col].nunique()} unique values")
    
    if len(feature_cols) > 10:
        print(f"... and {len(feature_cols) - 10} more features")
    
    # Create correlation matrix for numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(ANALYSIS_DIR, 'plots', 'correlation_matrix.png'))
        plt.close()
        
        print("\nTop Feature Correlations:")
        # Get top 5 correlations (excluding self-correlations)
        correlations = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Only use lower triangle
                    correlations.append((col1, col2, corr_matrix.loc[col1, col2]))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, corr in correlations[:5]:
            print(f"- {col1} and {col2}: {corr:.3f}")

def main():
    """Main function to run the analysis."""
    print("Starting COVID-19 Classification Dataset Quick Analysis")
    
    # Step 1: Load the dataset
    df = load_dataset()
    
    if df is None or df.empty:
        print("ERROR: Could not load classification dataset")
        sys.exit(1)
    
    # Step 2: Analyze target distribution
    analyze_target_distribution(df)
    
    # Step 3: Analyze demographics
    analyze_demographics(df)
    
    # Step 4: Analyze features
    analyze_features(df)
    
    print("\nAnalysis complete!")
    print(f"Visualizations saved to: {ANALYSIS_DIR}/plots/")

if __name__ == "__main__":
    main()