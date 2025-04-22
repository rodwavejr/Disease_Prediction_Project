# Data Directory for COVID-19 Detection Project

This directory contains all data used by the COVID-19 detection pipeline, organized by processing stage and source.

## Directory Structure

```
data/
├── raw/                # Original unprocessed data
├── processed/          # Processed data ready for modeling
├── interim/            # Intermediate data that has been transformed
└── external/           # Data from external sources (e.g., MIMIC, CDC)
    └── mimic/          # MIMIC-IV clinical data
```

## Data Sources

### CDC Case Surveillance Data
- Contains public COVID-19 case data
- Provides demographic and clinical information
- Located in `external/covid19_case_surveillance.csv`

### MIMIC-IV Clinical Data
- Contains hospital encounter data
- Located in `external/mimic/`
- Includes:
  - Patient demographics
  - Diagnoses with ICD codes
  - Lab results
  - Clinical notes (available in MIMIC-IV-Note module)

### CORD-19 Research Dataset
- Contains COVID-19 research papers and abstracts
- Used for entity recognition and knowledge extraction
- Located in `raw/cord19_*.csv`

## Processed Data

The key datasets produced by our pipeline include:

### NER Feature Datasets
- Extracted entities from clinical notes
- Located in `processed/ner_features.csv`

### Classification Datasets
- Combined features for COVID-19 prediction
- Located in `processed/covid_classification_dataset.csv`
- Includes both structured data and NER-derived features

## Data Access

The data in this directory is accessed by:
1. Scripts in the project root (e.g., prepare_classification_dataset.py)
2. Notebooks in the notebooks/ directory
3. Source modules in the src/ directory 

## Data Privacy

Some data in this project contains sensitive medical information:
- All datasets should be properly de-identified
- Do not commit raw data containing PHI to the repository
- Follow all applicable data use agreements

## Adding New Data

When adding new data sources:
1. Place raw data in the appropriate subdirectory
2. Document the source and format in this README
3. Update data loading scripts as needed