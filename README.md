# COVID-19 Detection from Unstructured Medical Text

This project implements an NLP pipeline for detecting COVID-19 cases from unstructured medical text. It identifies potential COVID-19 cases based on symptoms, severity indicators, and other relevant entities extracted from clinical notes and patient-reported symptoms.

## Project Structure

```
Disease_Prediction_Project/
├── data/                 # Data directory (created as needed)
│   ├── raw/              # Raw data from various sources
│   └── processed/        # Processed data for model input
├── docs/                 # Documentation
│   └── project_overview.md  # Comprehensive project overview
├── notebooks/            # Jupyter notebooks for exploration and demonstrations
│   ├── 01_initial_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_building.ipynb
│   └── 04_ner_extraction_pipeline.ipynb
├── output/               # Model outputs and visualizations
├── src/                  # Source code
│   ├── data_collection.py   # Tools for collecting medical text data
│   ├── data_processing.py   # Data preprocessing functions
│   ├── modeling.py          # Model training and evaluation
│   ├── model_evaluation.py  # Detailed evaluation metrics
│   └── ner_extraction.py    # NER functionality for medical text
├── test_ner.py           # Test script for NER functionality
├── models/               # Trained models
└── results/              # Generated analysis results
```

## Pipeline Overview

Our COVID-19 detection pipeline consists of two major stages:

1. **Named Entity Recognition (NER)**: Extract medical entities from text, including:
   - Symptoms (fever, cough, loss of taste/smell, etc.)
   - Time expressions (onset, duration)
   - Severity indicators (mild, moderate, severe)
   - Pre-existing conditions
   - Medications
   - Social factors (exposure, travel)

2. **Classification with Transformer Models**: Use the extracted entities to determine whether the text indicates COVID-19 or another condition.

## Getting Started

### Prerequisites

- Python 3.8+
- Libraries: pandas, numpy, spacy, transformers, torch (see requirements.txt)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Disease_Prediction_Project
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download necessary spaCy models:
   ```
   python -m spacy download en_core_web_sm
   ```

### Running the Demo

1. Test the NER functionality on a synthetic clinical note:
   ```
   python test_ner.py
   ```

2. Run the Jupyter notebooks to see detailed examples:
   ```
   jupyter notebook notebooks/04_ner_extraction_pipeline.ipynb
   ```

## Usage

### Data Collection

```python
from src.data_collection import fetch_cord19_dataset, generate_synthetic_clinical_note

# Generate a synthetic clinical note
note = generate_synthetic_clinical_note(has_covid=True)

# Fetch sample CORD-19 dataset
data_path = fetch_cord19_dataset(output_dir="data/raw", limit=100)
```

### Named Entity Recognition

```python
from src.ner_extraction import extract_entities_from_text, format_entities_for_bert

# Extract medical entities from text
entities = extract_entities_from_text(
    text=note,
    method="rule"  # Options: "rule", "spacy", "transformer"
)

# Format for transformer model
transformer_input = format_entities_for_bert(note, entities)
```

## Current Status

We have successfully implemented:

1. **Data Collection Module**: Tools for gathering medical text data from various sources
2. **Named Entity Recognition (NER) Module**: Methods for extracting medical entities from text
3. **Entity Formatting**: Tools to convert extracted entities into a format for transformer models

Next up: Implementing the transformer-based classification model.

## Requirements

Key packages required for this project:
- pandas
- numpy
- spacy
- transformers
- torch
- matplotlib
- scikit-learn
- tqdm

A full requirements.txt will be updated as development continues.
