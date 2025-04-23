# COVID-19 Detection from Unstructured Medical Text

This project implements an end-to-end pipeline for detecting COVID-19 cases from unstructured medical text. It identifies potential COVID-19 cases based on symptoms, severity indicators, and other relevant entities extracted from clinical notes, combined with structured patient data.

## Demo Video

[![COVID-19 Detection Demo](https://img.shields.io/badge/Watch%20Demo-Video%20Preview-blue)](assets/videos/demo.mov)

> **Note:** The demo video is a large file (approximately 22.2 MB) and may not preview directly on GitHub. Click the badge above to download and view the video locally.

*Video Description: This demonstration shows the COVID-19 detection pipeline in action, including text processing, entity extraction, and classification of patient cases based on clinical notes.*

## Pipeline Overview

Our COVID-19 detection pipeline consists of two major stages:

1. **Named Entity Recognition (NER)**: Extract medical entities from text, including:
   - Symptoms (fever, cough, loss of taste/smell, etc.)
   - Time expressions (onset, duration)
   - Severity indicators (mild, moderate, severe)
   - Pre-existing conditions
   - Medications
   - Social factors (exposure, travel)

2. **Classification**: Use extracted entities + structured data to predict COVID-19 status
   - Combines NER features with demographics and clinical data
   - Uses transformer models for classification
   - Provides probability scores for risk assessment

## Project Structure

```
Disease_Prediction_Project/
├── assets/                 # Project assets
│   └── videos/             # Video assets for documentation
├── data/                   # Data directory (created as needed)
│   ├── raw/                # Raw data from various sources
│   ├── processed/          # Processed data for model input
│   ├── interim/            # Intermediate data
│   └── external/           # External datasets (MIMIC, CDC, etc.)
│       └── mimic/          # MIMIC clinical data
├── docs/                   # Documentation
│   ├── project_overview.md # Comprehensive project overview
│   ├── project_explanation.md # Project details
│   ├── README_analysis.md  # Analysis documentation
│   └── data_strategy.md    # Data integration strategy
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_initial_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_building.ipynb
│   ├── 04_ner_extraction_pipeline.ipynb
│   ├── 05_data_exploration.ipynb
│   ├── 06_mimic_data_exploration.ipynb
│   └── 07_classification_dataset_preparation.ipynb
├── output/                 # Model outputs and visualizations
│   ├── mimic_ner_results.json
│   ├── ner_test_results.json
│   └── test_integration/   # Integration test results
├── results/                # Analysis results and metrics
│   └── analysis/           # Analysis outputs
│       └── plots/          # Visualization plots
├── src/                    # Source code
│   ├── data_collection.py  # Tools for collecting data
│   ├── data_fetcher.py     # Tools for fetching data
│   ├── data_processing.py  # Data preprocessing functions
│   ├── ner_extraction.py   # NER for medical text
│   ├── data_integration.py # Integrates multiple data sources
│   ├── mimic_integration.py # MIMIC data integration utilities
│   ├── modeling.py         # Model training and evaluation
│   ├── model_evaluation.py # Detailed evaluation metrics
│   └── scripts/            # Utility scripts
│       ├── analyze_classification_dataset.py # Analysis script
│       ├── download_cdc_data.py # Download CDC data
│       ├── download_real_data.py # Download datasets
│       ├── generate_data.py # Generate test data
│       ├── prepare_and_analyze.py # Data preparation
│       ├── prepare_classification_dataset.py # Dataset preparation
│       ├── preview_classification_data.py # Data preview
│       ├── quick_analysis.py # Quick analysis tools
│       ├── run_venv.sh # Virtual environment setup
│       └── setup_mimic_data.py # MIMIC data setup
├── tests/                  # Test scripts
│   ├── test_dataset_analysis.py # Test dataset analysis
│   ├── test_ner.py         # Tests for NER functionality
│   ├── test_mimic_ner.py   # Tests for MIMIC-specific NER
│   └── test_ner_integration.py # Tests for NER integration
├── models/                 # Trained models
├── presentations/          # Project presentations
│   ├── COVID19_Detection_Project.pdf
│   ├── COVID19_Detection_Project.pptx
│   └── COVID19_Detection_Project.docx
├── run_pipeline.py         # Main pipeline script
└── requirements.txt        # Project dependencies
```

## Data Flow

This project follows a clear data flow:

1. **Data Collection**:
   - CDC COVID-19 Case Surveillance data
   - MIMIC-IV clinical data
   - Clinical trials and medical literature
   
2. **Text Processing**:
   - Extract medical entities from clinical notes using NER
   - Convert unstructured text to structured features
   
3. **Data Integration**:
   - Combine structured EHR data with NER features
   - Create comprehensive feature vectors
   
4. **Classification**:
   - Train models to predict COVID-19 status
   - Evaluate model performance
   - Generate risk assessments

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

### Data Setup

1. Download the necessary datasets:
   ```
   python src/scripts/download_real_data.py --datasets all
   ```

2. Set up MIMIC-IV data (if available):
   ```
   python src/scripts/setup_mimic_data.py
   ```

3. Run the Jupyter notebooks to explore the data:
   ```
   jupyter notebook notebooks/
   ```

4. Or use the complete pipeline script:
   ```
   python run_pipeline.py --with-mimic --analyze
   ```

## Usage

### Complete Pipeline (Quick Start)

To run the full end-to-end pipeline from NER to BioBERT classification:

```bash
# Activate the virtual environment
source covid_venv/bin/activate

# Run the master notebook
jupyter notebook notebooks/master/Text_Mining_NER_to_BioBERT_Pipeline.ipynb
```

### NER Pipeline

```python
from src.ner_extraction import extract_entities_from_text, format_entities_for_bert

# Extract medical entities from text
entities = extract_entities_from_text(
    text=clinical_note,
    method="rule"  # Options: "rule", "spacy", "transformer"
)

# Format for transformer model
transformer_input = format_entities_for_bert(clinical_note, entities)
```

### Classification

```python
# Prepare the classification dataset that combines NER features with structured data
python src/scripts/prepare_classification_dataset.py

# Use notebooks for model training and evaluation
jupyter notebook notebooks/03_model_building.ipynb
```

## Status and Capabilities

We have successfully implemented:

1. **NER Module**: Extracts medical entities from clinical text
   - Rule-based, spaCy, and transformer approaches
   - Symptom, time, and severity extraction

2. **Data Integration**: Combines data from multiple sources
   - CDC surveillance data
   - MIMIC-IV clinical data
   - NER-extracted features

3. **Classification Pipeline**: Ready for model training
   - Feature engineering
   - Model selection and evaluation
   - Performance metrics

## Next Steps

1. Implement the transformer-based classification model
2. Train custom NER models on real medical text
3. Evaluate and optimize the full pipeline

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

See requirements.txt for complete dependencies.