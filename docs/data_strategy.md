# COVID-19 Detection Pipeline: Data Strategy

This document outlines our data strategy for the COVID-19 detection pipeline, focusing on both unstructured text analysis with NER and structured data classification.

## Overview

Our pipeline uses a two-stage approach:

1. **Named Entity Recognition (NER)** to extract medical entities from unstructured text
2. **Classification** to predict COVID-19 likelihood based on structured data and extracted entities

## Data Sources

### Unstructured Text Data (for NER)

| Dataset | Description | Size | Content | Access |
|---------|-------------|------|---------|--------|
| **CORD-19** | COVID-19 Open Research Dataset | ~10GB (~400k papers) | Scientific papers on COVID-19 | Public |
| **Clinical Trials** | COVID-19 clinical trials | ~50MB (~10k trials) | Trial descriptions, eligibility criteria | Public |
| **Twitter Data** | COVID-19 tweets | Variable (~100MB/day) | Social media posts about symptoms | Public |

### Structured Data (for Classification)

| Dataset | Description | Size | Features | Access |
|---------|-------------|------|----------|--------|
| **CDC COVID-19 Case Surveillance** | De-identified patient data | ~1GB (~30M patients) | Demographics, outcomes, symptoms | Public |
| **MIMIC-III** | Clinical database | ~40GB | EHR data, clinical notes | Requires credentials |
| **i2b2 NLP Challenge** | Annotated clinical notes | Variable | Medical concepts, relations | Requires application |

## Data Integration Strategy

### Stage 1: NER Processing

1. **Data Collection**
   - Download CORD-19 dataset
   - Collect clinical trials descriptions
   - Process medical forum posts

2. **Text Preprocessing**
   - Clean and normalize text
   - Split into sentences
   - Filter for COVID-19 relevance

3. **Entity Extraction**
   - Extract symptoms and their severity
   - Identify temporal expressions
   - Recognize medications and conditions
   - Detect social factors (travel, exposure)

4. **Feature Generation**
   - Convert extracted entities to structured features
   - Generate entity counts and presence indicators
   - Create contextual features (symptom duration, etc.)

### Stage 2: Classification

1. **Structured Data Processing**
   - Download CDC COVID-19 Case Surveillance data
   - Clean and preprocess demographic features
   - Encode categorical variables

2. **Feature Integration**
   - Combine structured patient data with extracted features
   - Align timestamps and record IDs where possible
   - Create comprehensive feature vectors

3. **Model Training**
   - Split data into train/validation/test sets
   - Train transformer-based classification models
   - Optimize for sensitivity and specificity

## Data Schema Example

### NER Output Schema

```json
{
  "document_id": "doc_123",
  "text": "Patient presents with fever and cough for 3 days...",
  "entities": {
    "SYMPTOM": [
      {"text": "fever", "start": 24, "end": 29},
      {"text": "cough", "start": 34, "end": 39}
    ],
    "TIME": [
      {"text": "for 3 days", "start": 40, "end": 51}
    ],
    "SEVERITY": []
  }
}
```

### Classification Feature Schema

```json
{
  "patient_id": "PT12345",
  "demographics": {
    "age": 45,
    "gender": "Male"
  },
  "ner_features": {
    "symptom_count": 2,
    "has_fever": 1,
    "has_cough": 1,
    "symptom_duration_days": 3
  },
  "target": {
    "covid_test_result": "Positive",
    "has_covid": 1
  }
}
```

## Implementation Plan

1. **Data Acquisition (Current Phase)**
   - Identify and document available data sources
   - Develop data fetching and processing scripts
   - Create sample datasets for development

2. **NER Development**
   - Implement NER models (rule-based, spaCy, transformers)
   - Extract and structure medical entities
   - Generate features for classification

3. **Classification Development**
   - Integrate NER features with structured data
   - Train classification models
   - Evaluate performance on held-out test data

4. **Validation and Deployment**
   - Validate against clinical ground truth
   - Develop visualization and explanation tools
   - Create API for processing new text data

## Challenges and Considerations

1. **Data Privacy**: Ensure all patient data is properly de-identified and complies with regulations

2. **Data Quality**: Handle inconsistencies in how symptoms are described across sources

3. **Class Imbalance**: Address potential underrepresentation of COVID-positive cases

4. **Integration**: Successfully link unstructured text with structured patient records

5. **Evolving Knowledge**: Account for changing symptom profiles as new variants emerge

## Next Steps

1. Download and process the real datasets
2. Train NER models on unstructured text
3. Extract features from unstructured text
4. Integrate with structured EHR data
5. Train classification models on the combined dataset