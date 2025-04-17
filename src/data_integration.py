"""
Data integration module for the COVID-19 detection pipeline.

This module handles:
1. Integration of extracted entities from unstructured text (Stage 1: NER)
2. Combination with structured data (Stage 2: Classification)
3. Creation of the final feature set for the classification model
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIntegrator:
    """
    Class to integrate data from multiple sources for the COVID-19 detection pipeline.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the DataIntegrator.
        
        Parameters:
        -----------
        data_dir : str
            Base directory for data files
        """
        self.data_dir = data_dir
        self.external_dir = os.path.join(data_dir, 'external')
        self.processed_dir = os.path.join(data_dir, 'processed')
        self.ner_results_dir = os.path.join(data_dir, 'ner_results')
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.external_dir, self.processed_dir, self.ner_results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info(f"DataIntegrator initialized with data directory: {data_dir}")
    
    def load_cdc_data(self, file_path=None, sample_size=None):
        """
        Load CDC COVID-19 case surveillance data.
        
        Parameters:
        -----------
        file_path : str
            Path to the CDC data CSV file. If None, looks in the external directory.
        sample_size : int
            Number of rows to load. If None, loads all data.
        
        Returns:
        --------
        pandas.DataFrame
            Loaded CDC data
        """
        if file_path is None:
            file_path = os.path.join(self.external_dir, 'covid19_case_surveillance.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"CDC data file not found: {file_path}")
            return None
        
        logger.info(f"Loading CDC data from {file_path}")
        
        try:
            if sample_size:
                df = pd.read_csv(file_path, nrows=sample_size)
                logger.info(f"Loaded {len(df)} rows from CDC data (sample)")
            else:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} rows from CDC data")
                
            return df
        except Exception as e:
            logger.error(f"Error loading CDC data: {e}")
            return None
    
    def load_cord19_data(self, file_path=None, sample_size=None):
        """
        Load CORD-19 metadata.
        
        Parameters:
        -----------
        file_path : str
            Path to the CORD-19 metadata CSV file. If None, looks in the external directory.
        sample_size : int
            Number of rows to load. If None, loads all data.
        
        Returns:
        --------
        pandas.DataFrame
            Loaded CORD-19 metadata
        """
        if file_path is None:
            file_path = os.path.join(self.external_dir, 'cord19_metadata.csv')
        
        if not os.path.exists(file_path):
            logger.warning(f"CORD-19 metadata file not found: {file_path}")
            return None
        
        logger.info(f"Loading CORD-19 metadata from {file_path}")
        
        try:
            if sample_size:
                df = pd.read_csv(file_path, nrows=sample_size)
                logger.info(f"Loaded {len(df)} rows from CORD-19 metadata (sample)")
            else:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} rows from CORD-19 metadata")
                
            return df
        except Exception as e:
            logger.error(f"Error loading CORD-19 metadata: {e}")
            return None
    
    def load_clinical_trials_data(self, file_path=None):
        """
        Load clinical trials data.
        
        Parameters:
        -----------
        file_path : str
            Path to the clinical trials JSON file. If None, looks in the external directory.
        
        Returns:
        --------
        pandas.DataFrame
            Loaded clinical trials data
        """
        if file_path is None:
            file_path = os.path.join(self.external_dir, 'covid19_clinical_trials.json')
        
        if not os.path.exists(file_path):
            logger.warning(f"Clinical trials file not found: {file_path}")
            return None
        
        logger.info(f"Loading clinical trials data from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                trials_data = json.load(f)
            
            # Extract the studies from the JSON structure
            if 'StudyFieldsResponse' in trials_data and 'StudyFields' in trials_data['StudyFieldsResponse']:
                studies = trials_data['StudyFieldsResponse']['StudyFields']
                df = pd.DataFrame(studies)
                logger.info(f"Loaded {len(df)} clinical trials")
                return df
            else:
                logger.warning(f"Unexpected structure in clinical trials data")
                return None
        except Exception as e:
            logger.error(f"Error loading clinical trials data: {e}")
            return None
    
    def load_ner_results(self, file_path=None):
        """
        Load results from the NER stage.
        
        Parameters:
        -----------
        file_path : str
            Path to the NER results JSON file. If None, looks in the ner_results directory.
        
        Returns:
        --------
        list
            List of NER results with extracted entities
        """
        if file_path is None:
            file_path = os.path.join(self.ner_results_dir, 'ner_results.json')
        
        if not os.path.exists(file_path):
            logger.warning(f"NER results file not found: {file_path}")
            return None
        
        logger.info(f"Loading NER results from {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                ner_results = json.load(f)
            
            logger.info(f"Loaded {len(ner_results)} documents with NER results")
            return ner_results
        except Exception as e:
            logger.error(f"Error loading NER results: {e}")
            return None
    
    def extract_features_from_ner(self, ner_results):
        """
        Extract structured features from NER results.
        
        Parameters:
        -----------
        ner_results : list
            List of documents with extracted entities
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with structured features
        """
        if not ner_results:
            logger.warning("No NER results to extract features from")
            return None
        
        logger.info("Extracting features from NER results")
        
        features = []
        
        for doc in ner_results:
            doc_features = {
                "document_id": doc.get("document_id", ""),
                "text_length": len(doc.get("text", "")),
            }
            
            # Count entities by type
            entity_counts = defaultdict(int)
            entities = doc.get("entities", {})
            
            for entity_type, entity_list in entities.items():
                entity_counts[f"{entity_type}_count"] = len(entity_list)
                
                # Extract text of first few entities
                if entity_list:
                    for i, entity in enumerate(entity_list[:3]):
                        doc_features[f"{entity_type}_{i+1}"] = entity.get("text", "")
            
            # Add counts to features
            doc_features.update(entity_counts)
            
            # Add to feature list
            features.append(doc_features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features)
        logger.info(f"Extracted {len(df.columns)} features from {len(df)} documents")
        
        return df
    
    def integrate_data(self, cdc_data=None, ner_features=None):
        """
        Integrate structured CDC data with NER features.
        
        In a real implementation, this would match structured patient records
        with the corresponding NER results.
        
        Parameters:
        -----------
        cdc_data : pandas.DataFrame
            Structured CDC data
        ner_features : pandas.DataFrame
            Features extracted from NER
        
        Returns:
        --------
        pandas.DataFrame
            Integrated dataset
        """
        if cdc_data is None and ner_features is None:
            logger.warning("No data to integrate")
            return None
        
        logger.info("Integrating CDC data with NER features")
        
        # If we only have one dataset, return it
        if cdc_data is None:
            logger.info("Only NER features available, returning those")
            return ner_features
        
        if ner_features is None:
            logger.info("Only CDC data available, returning that")
            return cdc_data
        
        # In a real implementation, we would link these datasets
        # For demonstration, we'll create a synthetic integrated dataset
        
        # Take a subset of CDC data
        cdc_subset = cdc_data.head(len(ner_features))
        
        # Reset indices to ensure alignment
        cdc_subset = cdc_subset.reset_index(drop=True)
        ner_features = ner_features.reset_index(drop=True)
        
        # Combine datasets
        integrated_data = pd.concat([cdc_subset, ner_features], axis=1)
        
        logger.info(f"Created integrated dataset with {len(integrated_data)} rows and {len(integrated_data.columns)} columns")
        
        return integrated_data
    
    def prepare_for_classification(self, integrated_data, target_column=None):
        """
        Prepare the integrated data for classification.
        
        Parameters:
        -----------
        integrated_data : pandas.DataFrame
            Integrated dataset
        target_column : str
            Name of the target column. If None, uses 'covid_status' or 'case_positive_specimen_interval'.
        
        Returns:
        --------
        tuple
            (X, y) feature matrix and target vector
        """
        if integrated_data is None:
            logger.warning("No integrated data to prepare")
            return None, None
        
        logger.info("Preparing integrated data for classification")
        
        # Copy the data to avoid modifying the original
        df = integrated_data.copy()
        
        # For demonstration, we'll use a placeholder target column
        if target_column is None:
            for col in ['covid_status', 'case_positive_specimen_interval', 'current_status']:
                if col in df.columns:
                    target_column = col
                    break
            
            if target_column is None:
                # Create a synthetic target if none exists
                logger.warning("No target column found, creating synthetic target")
                df['target'] = np.random.choice([0, 1], size=len(df))
                target_column = 'target'
        
        # Extract target
        y = df[target_column].copy()
        
        # Remove target and irrelevant columns
        drop_cols = [target_column, 'document_id']
        drop_cols = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=drop_cols)
        
        logger.info(f"Prepared classification data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def save_integrated_data(self, integrated_data, file_name='integrated_data.csv'):
        """
        Save the integrated data to a CSV file.
        
        Parameters:
        -----------
        integrated_data : pandas.DataFrame
            Integrated dataset
        file_name : str
            Name of the file to save
        
        Returns:
        --------
        str
            Path to saved file
        """
        if integrated_data is None:
            logger.warning("No integrated data to save")
            return None
        
        output_path = os.path.join(self.processed_dir, file_name)
        
        logger.info(f"Saving integrated data to {output_path}")
        
        try:
            integrated_data.to_csv(output_path, index=False)
            logger.info(f"Saved integrated data with {len(integrated_data)} rows and {len(integrated_data.columns)} columns")
            return output_path
        except Exception as e:
            logger.error(f"Error saving integrated data: {e}")
            return None

def demonstrate_data_flow():
    """Demonstrate the data flow for the COVID-19 detection pipeline."""
    
    print("=" * 50)
    print("COVID-19 DETECTION PIPELINE: DATA FLOW DEMONSTRATION")
    print("=" * 50)
    
    print("\nThis demonstration shows how data flows through our pipeline:")
    print("1. Unstructured text → NER → Extracted entities")
    print("2. Structured EHR data + Extracted entities → Classification model")
    
    # Demonstrate with sample data
    print("\nSTAGE 1: Unstructured Text Processing")
    print("-" * 40)
    print("Example clinical note:")
    print("-" * 40)
    
    clinical_note = """
Patient is a 45-year-old male who presents with fever, dry cough, and fatigue for the past 3 days. 
Patient also reports loss of taste and smell since yesterday.

Vitals: Temp 38.5°C, HR 95, BP 128/82, RR 18, O2 Sat 94% on room air. Physical exam reveals mild respiratory distress. 
Lungs with scattered rhonchi bilaterally. No rales or wheezes.

Assessment: Clinical presentation consistent with COVID-19 infection. 
Plan: COVID-19 PCR test ordered. Patient advised to self-isolate pending results. 
Symptomatic treatment with acetaminophen for fever. Follow up in 2-3 days.
    """
    
    print(clinical_note)
    
    print("\nExtracted entities (NER output):")
    print("-" * 40)
    
    ner_output = {
        "SYMPTOM": [
            {"text": "fever", "start": 53, "end": 58},
            {"text": "dry cough", "start": 60, "end": 69},
            {"text": "fatigue", "start": 75, "end": 82},
            {"text": "loss of taste", "start": 118, "end": 131},
            {"text": "loss of smell", "start": 136, "end": 149}
        ],
        "TIME": [
            {"text": "for the past 3 days", "start": 83, "end": 102},
            {"text": "since yesterday", "start": 150, "end": 165}
        ],
        "SEVERITY": [
            {"text": "mild respiratory", "start": 247, "end": 263}
        ]
    }
    
    for entity_type, entities in ner_output.items():
        print(f"{entity_type}:")
        for entity in entities:
            print(f"  - {entity['text']}")
    
    print("\nSTAGE 2: Structured Data Integration")
    print("-" * 40)
    
    # Create sample structured data
    structured_data = {
        "patient_id": "PT12345",
        "age": 45,
        "gender": "Male",
        "admission_date": "2023-01-15",
        "hospital": "General Hospital",
        "symptoms": "fever, dry cough, fatigue, loss of taste, loss of smell",
        "covid_test_result": "Positive",
        "has_covid": 1
    }
    
    print("Structured patient record:")
    for key, value in structured_data.items():
        print(f"  {key}: {value}")
    
    # Show extracted features
    print("\nFeatures extracted from NER for classification:")
    print("-" * 40)
    
    extracted_features = {
        "symptom_count": 5,
        "time_expression_count": 2,
        "severity_indicator_count": 1,
        "has_fever": 1,
        "has_cough": 1,
        "has_fatigue": 1,
        "has_taste_loss": 1,
        "has_smell_loss": 1,
        "symptom_duration_days": 3
    }
    
    for feature, value in extracted_features.items():
        print(f"  {feature}: {value}")
    
    # Show combined dataset
    print("\nCombined dataset for classification:")
    print("-" * 40)
    
    combined_features = {**structured_data, **extracted_features}
    
    print("Final feature set:")
    feature_count = 0
    for key, value in combined_features.items():
        if key != "has_covid" and key != "covid_test_result":
            print(f"  {key}: {value}")
            feature_count += 1
    
    print(f"\nTotal: {feature_count} features")
    print(f"Target: has_covid = {combined_features['has_covid']}")

if __name__ == "__main__":
    demonstrate_data_flow()