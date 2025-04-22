# Source Code Directory

This directory contains the core source code for the COVID-19 Detection Project.

## Directory Structure

### Core Modules
- `data_collection.py` - Functions for retrieving data from various sources
- `data_fetcher.py` - Tools for fetching and downloading datasets
- `data_integration.py` - Code for integrating data from multiple sources
- `data_processing.py` - Data cleaning and preprocessing functions
- `mimic_integration.py` - Integration with MIMIC-IV clinical database
- `model_evaluation.py` - Evaluation metrics and model assessment
- `modeling.py` - Machine learning model definition and training
- `ner_extraction.py` - Named Entity Recognition extraction pipeline

### Scripts
The `scripts/` directory contains utility scripts for data preparation, analysis, and project setup:

- **Data Download**
  - `download_cdc_data.py` - Download CDC COVID-19 case surveillance data
  - `download_real_data.py` - Download real-world datasets for analysis
  - `setup_mimic_data.py` - Set up MIMIC-IV clinical data for processing

- **Data Generation**
  - `generate_data.py` - Generate test data when real data not available

- **Data Processing**
  - `prepare_classification_dataset.py` - Prepare dataset for classification models
  - `prepare_and_analyze.py` - Combined data preparation and analysis script

- **Analysis**
  - `analyze_classification_dataset.py` - Analyze classification dataset statistics
  - `preview_classification_data.py` - Quick preview of classification dataset
  - `quick_analysis.py` - Simple exploratory analysis tools

- **Environment**
  - `run_venv.sh` - Shell script to activate and configure virtual environment

## Pipeline Flow

The source code implements a two-stage pipeline:

1. **NER Extraction Stage**:
   - Process clinical notes with NER
   - Extract medical entities (symptoms, time expressions, severity)
   - Convert to structured features

2. **Classification Stage**:
   - Combine NER features with structured data
   - Create classification dataset
   - Train and evaluate prediction models

The pipeline is orchestrated by the root-level `run_pipeline.py` script.