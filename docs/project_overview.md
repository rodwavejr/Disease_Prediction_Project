# COVID-19 Detection from Unstructured Medical Text

## Project Focus

This project focuses on **detecting COVID-19 cases from unstructured medical text**. Many people experience symptoms that could be COVID-19, seasonal flu, common cold, or allergies. Our goal is to build an NLP pipeline that can identify likely COVID-19 cases from clinical notes, patient-reported symptoms, and other unstructured text sources.

## Why This Problem Matters

During the COVID-19 pandemic and beyond, distinguishing COVID-19 from similar respiratory conditions has been challenging:

1. **Similar symptom profiles** between COVID-19, flu, and common cold
2. **Limited testing availability** in many locations
3. **Delay in test results** after symptom onset
4. **Evolving symptom profiles** with new variants
5. **Vast amounts of unstructured medical text** that contain valuable diagnostic information

A robust NLP model could help:
- Identify potential COVID-19 cases earlier
- Prioritize patients for testing
- Track disease patterns in clinical notes
- Support doctors in triage decision-making 

## Technical Approach

Our technical pipeline consists of two major stages:

### Stage 1: Named Entity Recognition (NER)
We'll build and train a Named Entity Recognition model to identify key entities in unstructured text:
- **Symptoms** (fever, cough, fatigue, etc.)
- **Time expressions** (onset, duration)
- **Severity indicators** (mild, moderate, severe)
- **Pre-existing conditions**
- **Medications**
- **Social factors** (exposure, travel)

### Stage 2: Classification with Transformer Models
Using the extracted entities and their context, we'll build a transformer-based classification model to:
- Determine the likelihood of COVID-19 vs. other respiratory conditions
- Provide probability scores for different diagnoses
- Explain which factors most influenced the prediction

## Data Sources

We'll gather data from multiple sources:

1. **De-identified Electronic Health Records (EHRs)** containing:
   - Clinical notes
   - Patient intake forms
   - Discharge summaries

2. **Public Health Datasets**:
   - CORD-19 research paper corpus
   - CDC case definitions and symptom guidelines

3. **Online Medical Resources**:
   - Medical forums where patients describe symptoms
   - Telehealth consultation records (de-identified)

## Implementation Plan

### 1. Data Collection and Preparation ✅
- Collect unstructured text from various sources
- De-identify and preprocess the text
- Create annotation guidelines for NER
- Annotate a training dataset for symptoms and relevant entities

### 2. NER Model Development ✅
- Fine-tune existing biomedical NER models (BioBERT, ClinicalBERT)
- Train custom NER models to identify COVID-specific entities
- Evaluate NER performance on held-out test data
- Create structured representations from extracted entities

### 3. Classification Model Development ⏳
- Engineer features from extracted entities
- Implement transformer architecture with attention mechanisms
- Train models to distinguish between COVID-19 and other conditions
- Evaluate classification performance and calibrate probability outputs

### 4. Deployment and Validation 🔄
- Create an API for processing new clinical text
- Develop visualizations to explain model predictions
- Validate the system against confirmed diagnostic test results
- Measure impact on triage decision-making

## Current Progress

We have successfully implemented:

1. **Data Collection Module**: Tools for gathering medical text data from various sources like CDC data, clinical trials and medical literature.

2. **Named Entity Recognition (NER) Module**: Methods for extracting medical entities from unstructured text, with three implementation options:
   - Rule-based NER using regex patterns
   - spaCy-based NER with custom training capabilities
   - Transformer-based NER using pre-trained biomedical models

3. **Entity Formatting**: Tools to convert extracted entities into a structured format suitable for transformer model input.

Next, we will be implementing the transformer-based classification model that will take the extracted entities and determine the likelihood of COVID-19.

## Precedents and Related Work

Several research efforts have applied NLP to COVID-19 detection:

1. Wang et al. (2020) developed a BERT-based model to identify COVID-19 cases from clinical notes with 96% sensitivity.

2. The COVID-19 Natural Language Processing Consortium created resources for extracting symptoms from clinical text.

3. Healthtech companies like Nference and Jvion built NLP solutions to screen for COVID-19 risk factors in EHRs.

4. Research by Zuccon et al. demonstrated how transformer models can differentiate between COVID-19 and influenza based on symptom descriptions.

## Challenges and Considerations

We anticipate several challenges:

1. **Data quality and availability**: De-identified medical texts may have inconsistencies and variations in how symptoms are described.

2. **Class imbalance**: COVID-positive texts may be underrepresented compared to other conditions.

3. **Evolving knowledge**: Symptom profiles changed throughout the pandemic as new variants emerged.

4. **Privacy concerns**: Working with medical data requires strict privacy controls and compliance with regulations.

5. **Validation requirements**: Medical applications need rigorous validation before deployment.

## Expected Outcomes

By the end of this project, we aim to deliver:

1. A trained NER model that can extract relevant medical entities from unstructured text
2. A classification model that can distinguish COVID-19 from similar conditions
3. A comprehensive evaluation of model performance against gold-standard diagnoses
4. An interpretable system that explains why a particular classification was made
5. A prototype API that can be integrated with existing healthcare systems

Our ultimate goal is to create a tool that can help healthcare providers quickly identify potential COVID-19 cases from the vast amount of unstructured medical text they process daily, allowing for earlier intervention and more effective resource allocation.