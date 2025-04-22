# COVID-19 Classification Dataset Analysis

This directory contains scripts for preparing and analyzing the COVID-19 classification dataset, combining data from CDC Case Surveillance and MIMIC-IV clinical data.

## Scripts

### 1. Preparation Script (`prepare_classification_dataset.py`)

This script prepares the COVID-19 classification dataset by:
- Loading CDC Case Surveillance data
- Loading and preprocessing MIMIC-IV clinical data
- Creating features for classification
- Combining data from both sources
- Saving the processed dataset

Features include:
- Patient demographics (gender, age)
- Health outcomes (hospitalization, ICU admission, death)
- Pre-existing conditions
- Laboratory results (for MIMIC data)

### 2. Analysis Script (`analyze_classification_dataset.py`)

This comprehensive script analyzes the prepared dataset to provide insights before model building:
- Class distribution analysis (COVID-19 positive/negative)
- Missing value analysis
- Feature distribution analysis
- Feature importance evaluation
- Correlation analysis
- Data source comparison

The results are saved as visualizations and a summary report in the `results/analysis` directory.

### 3. Quick Analysis Script (`quick_analysis.py`)

A simplified analysis script that focuses on the most important aspects of the dataset:
- COVID-19 distribution
- Demographic analysis
- Key feature analysis
- Correlation analysis

This script is faster to run and provides essential insights quickly.

### 4. Pipeline Script (`prepare_and_analyze.py`)

This script runs both the preparation and analysis scripts in sequence:
1. Prepares the classification dataset
2. Analyzes the prepared dataset
3. Provides detailed logging throughout the process

## Key Findings from the Analysis

- **Class Distribution**: There is a significant class imbalance, with approximately 80.5% negative cases and 19.5% positive cases (4.1:1 ratio)
- **Demographic Distribution**: 
  - Gender: 52.2% female, 47.8% male
  - Age: Mean age of 57.7 years, with a range from 18 to 91 years
- **COVID-19 by Gender**: Females have a slightly higher positive rate (20.5%) compared to males (18.4%)
- **Feature Correlations**: Most features have low correlation with COVID-19 status, suggesting complex relationships that may require advanced modeling techniques

## Running the Scripts

To run the full preparation and analysis pipeline:

```bash
cd /Users/Apexr/Documents/Disease_Prediction_Project
source covid_venv/bin/activate
python prepare_and_analyze.py
```

To run only the quick analysis:

```bash
cd /Users/Apexr/Documents/Disease_Prediction_Project
source covid_venv/bin/activate
python quick_analysis.py
```

## Visualizations

The analysis produces several visualizations in the `results/analysis/plots` directory:
- Target distribution (COVID-19 positive/negative cases)
- Age distribution
- Gender distribution
- COVID-19 positive rate by gender
- Feature correlation matrix

## Next Steps

1. Address the class imbalance using techniques like:
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Class weighting
   - Undersampling majority class

2. Feature engineering and selection:
   - Create new features from the existing ones
   - Select the most relevant features for classification
   - Handle missing values appropriately

3. Model building:
   - Try different classification algorithms
   - Tune hyperparameters
   - Evaluate model performance