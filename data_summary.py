"""
Summarize COVID-19 datasets needed for our pipeline.
"""

def print_dataset_summary():
    """Print summary of datasets used in the COVID-19 detection pipeline."""
    
    print("=" * 50)
    print("COVID-19 DETECTION PIPELINE: DATA REQUIREMENTS")
    print("=" * 50)
    
    print("\nSTAGE 1: UNSTRUCTURED TEXT FOR NER")
    print("-" * 40)
    
    unstructured_datasets = [
        {
            "name": "CORD-19 Research Dataset",
            "description": "COVID-19 Open Research Dataset with scientific papers",
            "data_type": "Unstructured research papers and abstracts",
            "entity_types": ["symptoms", "medications", "medical conditions", "time expressions"],
            "source": "Allen Institute for AI",
            "url": "https://www.semanticscholar.org/cord19",
            "size": "~10GB (~400k papers)",
            "format": "JSON + PDF",
            "uses": ["Extract medical terminology", "Train NER on formal medical language"]
        },
        {
            "name": "Clinical Trials Data",
            "description": "Detailed descriptions of COVID-19 clinical trials",
            "data_type": "Semi-structured trial descriptions, eligibility criteria",
            "entity_types": ["symptoms", "severity indicators", "medications", "comorbidities"],
            "source": "ClinicalTrials.gov",
            "url": "https://clinicaltrials.gov/ct2/results?cond=COVID-19",
            "size": "~50MB (~10k trials)",
            "format": "JSON / XML",
            "uses": ["Extract symptom descriptions", "Identify inclusion/exclusion criteria"]
        },
        {
            "name": "COVID-19 Tweet Dataset",
            "description": "Tweets related to COVID-19 symptoms and experiences",
            "data_type": "Unstructured social media text",
            "entity_types": ["symptoms", "severity (layperson terms)", "timeline expressions"],
            "source": "Panacea Lab GitHub",
            "url": "https://github.com/thepanacealab/covid19_twitter",
            "size": "~30GB (many millions of tweets)",
            "format": "TSV",
            "uses": ["Understand colloquial symptom descriptions", "Extract real-world experiences"]
        }
    ]
    
    for i, dataset in enumerate(unstructured_datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Data Type: {dataset['data_type']}")
        print(f"   Entity Types: {', '.join(dataset['entity_types'])}")
        print(f"   Source: {dataset['source']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Format: {dataset['format']}")
        print(f"   Uses: {', '.join(dataset['uses'])}")
        print()
    
    print("\nSTAGE 2: STRUCTURED DATA FOR CLASSIFICATION")
    print("-" * 40)
    
    structured_datasets = [
        {
            "name": "CDC COVID-19 Case Surveillance",
            "description": "De-identified patient-level data on COVID-19 cases",
            "data_type": "Structured patient records",
            "features": ["demographics", "clinical outcomes", "symptoms", "comorbidities"],
            "target": "COVID-19 diagnosis (confirmed/probable)",
            "source": "CDC",
            "url": "https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/vbim-akqf",
            "size": "~1GB (~30M patients)",
            "format": "CSV",
            "uses": ["Train classification models", "Understand demographic risk factors"]
        },
        {
            "name": "MIMIC-III Clinical Database",
            "description": "Clinical database containing real patient records",
            "data_type": "Structured EHR data + unstructured clinical notes",
            "features": ["vitals", "lab results", "medications", "procedures", "diagnoses"],
            "target": "Various outcomes (can filter for COVID)",
            "source": "PhysioNet",
            "url": "https://physionet.org/content/mimiciii/1.4/",
            "size": "~40GB",
            "format": "CSV + text",
            "access": "Requires credential application",
            "uses": ["Extract clinical note patterns", "Identify clinical predictors"]
        },
        {
            "name": "i2b2 NLP Challenge Datasets",
            "description": "Annotated clinical notes for NLP tasks",
            "data_type": "Annotated unstructured clinical notes",
            "features": ["annotated medical concepts", "relations", "temporal information"],
            "source": "i2b2",
            "url": "https://www.i2b2.org/NLP/DataSets/Main.php",
            "size": "Variable",
            "format": "XML",
            "access": "Requires application",
            "uses": ["Train domain-specific NER models", "Validate NER performance"]
        }
    ]
    
    for i, dataset in enumerate(structured_datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Data Type: {dataset['data_type']}")
        print(f"   Features: {', '.join(dataset['features'])}")
        if "target" in dataset:
            print(f"   Target: {dataset['target']}")
        print(f"   Source: {dataset['source']}")
        if "access" in dataset:
            print(f"   Access: {dataset['access']}")
        print(f"   Size: {dataset.get('size', 'Variable')}")
        print(f"   Format: {dataset['format']}")
        print(f"   Uses: {', '.join(dataset['uses'])}")
        print()
    
    print("\nINTEGRATION STRATEGY")
    print("-" * 40)
    print("""
1. Stage 1 (NER) will use primarily CORD-19 and Clinical Trials data to extract:
   - Symptoms and their severity
   - Medications and treatments
   - Temporal expressions
   - Medical conditions

2. Stage 2 (Classification) will use:
   - CDC COVID-19 Case Surveillance data as the primary structured dataset
   - Features extracted from the NER stage
   - Demographic information from the CDC data
   
3. Pipeline connectivity:
   - NER extracts entities from unstructured text
   - These entities are structured into features
   - Features are combined with demographic data
   - The combined features feed into the classification model
   - Model predicts COVID-19 likelihood
    """)

if __name__ == "__main__":
    print_dataset_summary()