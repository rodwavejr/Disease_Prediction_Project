import os
import sys
import json
import pandas as pd
from tqdm import tqdm

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

from src.ner_extraction import extract_entities_from_text

def extract_features_from_notes(notes_df):
    """Extract NER features from clinical notes"""
    features = []
    
    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Extracting NER features"):
        record_id = row.get('record_id', idx)
        note_text = row['note_text']
        
        # Extract entities
        entities = extract_entities_from_text(note_text, method="rule")
        
        # Count entities by type
        symptom_count = len(entities.get('SYMPTOM', []))
        time_count = len(entities.get('TIME', []))
        severity_count = len(entities.get('SEVERITY', []))
        
        # Extract specific symptoms
        symptoms = [entity['text'].lower() for entity in entities.get('SYMPTOM', [])]
        
        # Create binary features for common symptoms
        common_symptoms = [
            "fever", "cough", "shortness of breath", "difficulty breathing",
            "fatigue", "loss of taste", "loss of smell", "sore throat"
        ]
        
        symptom_features = {}
        for symptom in common_symptoms:
            has_symptom = any(symptom in s for s in symptoms)
            symptom_features[f"has_{symptom.replace(' ', '_')}"] = int(has_symptom)
        
        # Create feature record
        feature_record = {
            'record_id': record_id,
            'ner_symptom_count': symptom_count,
            'ner_time_count': time_count,
            'ner_severity_count': severity_count,
            **symptom_features
        }
        
        features.append(feature_record)
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    return df

def main():
    # Load notes dataset
    notes_path = os.path.join(root_dir, 'data/processed/covid_notes_classification.csv')
    print(f"Loading notes from {notes_path}")
    
    if not os.path.exists(notes_path):
        print(f"Error: Notes file not found at {notes_path}")
        # Try to find another notes file
        processed_dir = os.path.join(root_dir, 'data/processed')
        possible_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        print(f"Available files in processed directory: {possible_files}")
        
        # Try to find any file with 'note' in the name
        note_files = [f for f in possible_files if 'note' in f.lower()]
        if note_files:
            notes_path = os.path.join(processed_dir, note_files[0])
            print(f"Using alternative notes file: {notes_path}")
        else:
            print("No suitable notes file found. Cannot proceed with NER feature extraction.")
            return
    
    # Load the notes
    notes_df = pd.read_csv(notes_path)
    print(f"Loaded {len(notes_df)} notes for NER processing")
    
    # Check if 'note_text' column exists
    if 'note_text' not in notes_df.columns:
        print("Error: 'note_text' column not found in the notes dataset.")
        print(f"Available columns: {notes_df.columns.tolist()}")
        
        # Try to find a column that might contain text
        text_columns = [col for col in notes_df.columns if any(x in col.lower() for x in ['text', 'note', 'content'])]
        if text_columns:
            print(f"Found potential text columns: {text_columns}")
            # Use the first one
            notes_df['note_text'] = notes_df[text_columns[0]]
            print(f"Using '{text_columns[0]}' as the notes text column")
        else:
            print("No suitable text column found. Cannot proceed with NER feature extraction.")
            return
    
    # Extract features
    ner_features = extract_features_from_notes(notes_df)
    
    # Save features
    output_path = os.path.join(root_dir, 'data/processed/ner_features.csv')
    ner_features.to_csv(output_path, index=False)
    
    print(f"Extracted {len(ner_features.columns) - 1} NER features for {len(ner_features)} notes")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()