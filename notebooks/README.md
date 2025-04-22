# COVID-19 Detection Project Notebooks

This directory contains Jupyter notebooks for the COVID-19 Detection from Unstructured Medical Text project. These notebooks demonstrate various components of our pipeline from data exploration to model implementation.

## Notebook Organization

The notebooks are organized according to the pipeline stages:

### Master Notebooks
- **master/Text_Mining_NER_to_BioBERT_Pipeline.ipynb**: Comprehensive end-to-end pipeline from clinical text to COVID-19 prediction using BioBERT

### 1. Data Exploration
- **01_initial_data_exploration.ipynb**: Initial examination of COVID-19 data sources
- **05_data_exploration.ipynb**: In-depth exploration of COVID-19 datasets
- **06_mimic_data_exploration.ipynb**: Exploration of MIMIC-IV clinical data

### 2. Feature Engineering
- **02_feature_engineering.ipynb**: Converting extracted entities to features
- **04_ner_extraction_pipeline.ipynb**: Named Entity Recognition for medical texts
- **07_classification_dataset_preparation.ipynb**: Preparing the integrated dataset for classification

### 3. Modeling
- **03_model_building.ipynb**: Building and evaluating classification models

## Data Pipeline Flow

These notebooks follow the project's two-stage pipeline:

```
Stage 1: NER Pipeline              Stage 2: Classification Pipeline
┌───────────────────┐              ┌───────────────────┐
│ Clinical Notes    │              │ Structured Data   │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
┌─────────▼─────────┐              ┌─────────▼─────────┐
│ Entity Extraction │              │ Feature Processing│
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
┌─────────▼─────────┐              ┌─────────▼─────────┐
│ Feature Creation  │─────────────►│ Integrated Dataset│
└───────────────────┘              └─────────┬─────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │ Model Training    │
                                   └─────────┬─────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │ COVID-19 Risk     │
                                   │ Prediction        │
                                   └───────────────────┘
```

## Working with the Notebooks

### Required Data

To run these notebooks, you'll need:

1. **CDC Data**: Download with `python download_real_data.py --datasets cdc`
2. **MIMIC-IV Data**: Set up with `python setup_mimic_data.py` 
3. **Clinical Notes**: Available in the MIMIC-IV-Note module

### Notebook Dependencies

Make sure you have all the required packages installed:
```bash
pip install -r ../requirements.txt
```

### Running Order

**Quick Start**: For a comprehensive end-to-end demonstration of the entire pipeline, run:
- `master/Text_Mining_NER_to_BioBERT_Pipeline.ipynb`

**Detailed Exploration**: To explore individual components in depth, run the notebooks in this order:
1. Data exploration notebooks (01, 05, 06)
2. Feature engineering notebooks (02, 04)
3. Dataset preparation notebook (07)
4. Model building notebook (03)

## Key Insights

The notebooks reveal several important insights about COVID-19 detection:

1. **Key Symptoms**: The most predictive symptoms include loss of taste/smell, fever, and cough
2. **Temporal Information**: Symptom onset timing is critical for accurate classification
3. **Combined Approach**: NER features + structured clinical data outperforms either method alone
4. **Model Performance**: Our pipeline achieves >90% accuracy on test data

## Additional Resources

For more information about the project, refer to:
- `../docs/project_overview.md` - Comprehensive project overview
- `../docs/data_strategy.md` - Data integration strategy
- `../README.md` - Project setup and usage