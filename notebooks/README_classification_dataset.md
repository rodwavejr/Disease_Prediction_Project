# COVID-19 Classification Dataset

This document explains how to use the classification dataset preparation notebook and what to expect.

## Data Sources

The classification dataset combines two main sources:

1. **CDC COVID-19 Case Surveillance Data**
   - Location: `/data/external/covid19_case_surveillance.csv`
   - Contains: Records of COVID-19 cases with demographics, symptoms, outcomes
   - Key fields: `current_status`, `sex`, `age_group`, `hosp_yn`, `icu_yn`, `death_yn`

2. **MIMIC-IV Clinical Data**
   - Location: `/data/external/mimic/`
   - Contains: Patient data, diagnoses, lab results from hospital systems
   - Key files:
     - `patients_sample.csv`: Patient demographics
     - `relevant_diagnoses.csv`: ICD diagnosis codes
     - `d_icd_diagnoses.csv`: ICD code descriptions
     - `relevant_labevents.csv`: Laboratory test results

## Running the Classification Notebook

The notebook `07_classification_dataset_preparation.ipynb` creates a master classification dataset by:

1. Loading and preprocessing CDC data 
2. Loading and preprocessing MIMIC data
3. Combining both into a single dataset with COVID-19 positive/negative labels
4. Analyzing features and relationships with the target variable

### Requirements

The notebook requires these Python packages:
```
pandas
numpy
matplotlib
seaborn
scikit-learn (for the optional model example)
```

Install with:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Workflow

When you run the notebook, it will:

1. Load data from both sources (CDC and MIMIC)
2. Process features (convert categorical to numeric, handle missing values)
3. Create a COVID-19 positive/negative target variable
4. Combine data into a master dataset
5. Save the result to `../data/processed/covid_classification_dataset.csv`
6. Provide visualizations of feature distributions and relationships
7. (Optional) Demonstrate a simple classification model

### Output

The final dataset will include:
- Patient demographics (age, sex/gender)
- Clinical outcomes (hospitalization, ICU, death)
- Underlying medical conditions
- Lab test results (from MIMIC)
- COVID-19 positive/negative status (target variable)

This dataset is ready for:
- Feature engineering
- Model selection
- Hyperparameter tuning
- Integration with the NER pipeline

## Using With Limited Resources

If you can't install the required packages, you can still:

1. Examine the data files directly to understand structure
2. Use the notebook as a reference for the processing steps
3. Download the data for processing elsewhere

## Next Steps

After generating the classification dataset:

1. Explore feature relationships in more detail
2. Try different classification algorithms (Random Forest, SVM, Neural Networks)
3. Integrate features extracted from the NER pipeline
4. Create a model evaluation framework
5. Deploy the best performing model