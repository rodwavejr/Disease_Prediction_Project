"""
Test script for the NER extraction module.
Run this script to see the NER module in action with sample data.

This script supports both synthetic and real data modes:
- Default: Uses synthetic sample data for testing
- With --real flag: Attempts to use real data if available

Usage:
    python test_ner.py             # Use synthetic sample data
    python test_ner.py --real      # Use real data if available
"""

import os
import json
import re
import random
import argparse
from collections import defaultdict
import sys

# Check for optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: pandas not installed. Some functionality will be limited.")
    print("Install with: pip install pandas")
    PANDAS_AVAILABLE = False

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from src.data_collection import USE_SYNTHETIC_DATA
except ImportError as e:
    print(f"Warning: Unable to import from data_collection module: {e}")
    print("Falling back to using synthetic data only.")
    USE_SYNTHETIC_DATA = True

# Sample COVID-19 symptoms for demonstration
COVID_SYMPTOMS = [
    "fever", "cough", "shortness of breath", "difficulty breathing", 
    "fatigue", "muscle pain", "body ache", "headache", "loss of taste",
    "loss of smell", "sore throat", "congestion", "runny nose", "nausea",
    "vomiting", "diarrhea", "chills"
]

# Time expressions for rule-based detection
TIME_EXPRESSIONS = [
    "days ago", "weeks ago", "yesterday", "last week", "since", "for the past",
    "hours", "days", "weeks", "months", "began", "started", "onset"
]

# Severity indicators for rule-based detection
SEVERITY_INDICATORS = [
    "mild", "moderate", "severe", "slight", "significant", "extreme",
    "worsening", "improving", "persistent", "intermittent", "constant"
]

def get_sample_clinical_note(use_synthetic=True):
    """
    Returns a clinical note for testing purposes.
    
    Parameters:
    -----------
    use_synthetic : bool
        Whether to use synthetic data (True) or attempt to load real data (False)
        
    Returns:
    --------
    str
        Clinical note text
    """
    if use_synthetic:
        # Use synthetic data
        subjective = "Patient is a 45-year-old male who presents with fever, dry cough, and fatigue for the past 3 days. Patient also reports loss of taste and smell since yesterday."
        
        objective = "Vitals: Temp 38.5°C, HR 95, BP 128/82, RR 18, O2 Sat 94% on room air. Physical exam reveals mild respiratory distress. Lungs with scattered rhonchi bilaterally. No rales or wheezes."
        
        assessment = "Assessment: Clinical presentation consistent with COVID-19 infection."
        
        plan = "Plan: COVID-19 PCR test ordered. Patient advised to self-isolate pending results. Symptomatic treatment with acetaminophen for fever. Follow up in 2-3 days."
        
        # Combine note components
        note = f"{subjective}\n\n{objective}\n\n{assessment} {plan}"
        
        return note
    
    # If pandas is not available, we can't process real data files
    if not PANDAS_AVAILABLE:
        print("Pandas not available, falling back to synthetic note")
        return get_sample_clinical_note(use_synthetic=True)
        
    # Try to load real clinical note data
    # First check if we have MIMIC data
    mimic_path = 'data/external/mimic_notes.csv'
    
    if os.path.exists(mimic_path):
        try:
            # Read a sample of MIMIC notes (first 100 rows)
            mimic_df = pd.read_csv(mimic_path, nrows=100)
            
            # Check if we have the expected column
            if 'TEXT' in mimic_df.columns:
                # Get a note with COVID-relevant content
                covid_notes = mimic_df[mimic_df['TEXT'].str.contains('covid|coronavirus|sars', case=False, na=False)]
                
                if len(covid_notes) > 0:
                    note = covid_notes['TEXT'].iloc[0]
                    return note
                else:
                    # Return any note if no COVID-specific ones found
                    note = mimic_df['TEXT'].iloc[0]
                    return note
        except Exception as e:
            print(f"Error reading MIMIC data: {e}")
            print("Falling back to synthetic note")
    
    # Check if we have clinical trials data
    trials_path = 'data/external/covid19_clinical_trials.json'
    
    if os.path.exists(trials_path):
        try:
            with open(trials_path, 'r') as f:
                trials_data = json.load(f)
            
            trials_df = pd.DataFrame(trials_data['StudyFieldsResponse']['StudyFields'])
            
            if 'DetailedDescription' in trials_df.columns:
                descriptions = [desc[0] for desc in trials_df['DetailedDescription'] if desc and len(desc) > 0]
                if descriptions:
                    return descriptions[0]
        except Exception as e:
            print(f"Error reading clinical trials data: {e}")
            print("Falling back to synthetic note")
    
    # If we get here, we couldn't find real data, so use synthetic
    print("No suitable real data found, using synthetic clinical note")
    return get_sample_clinical_note(use_synthetic=True)

def extract_entities_with_rules(text):
    """Extract entities using simple rule-based patterns."""
    entities = {
        "SYMPTOM": [],
        "TIME": [],
        "SEVERITY": []
    }
    
    # Extract symptoms
    for symptom in COVID_SYMPTOMS:
        pattern = re.compile(r'\b({})\b'.format(re.escape(symptom)), re.IGNORECASE)
        for match in pattern.finditer(text):
            entities["SYMPTOM"].append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            })
    
    # Extract time expressions
    for time_expr in TIME_EXPRESSIONS:
        pattern = re.compile(r'.*({})'.format(re.escape(time_expr)), re.IGNORECASE)
        for match in pattern.finditer(text):
            entities["TIME"].append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            })
    
    # Extract severity indicators
    for severity in SEVERITY_INDICATORS:
        pattern = re.compile(r'\b({})\s+\w+'.format(re.escape(severity)), re.IGNORECASE)
        for match in pattern.finditer(text):
            entities["SEVERITY"].append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            })
    
    return entities

def format_for_transformer(text, entities):
    """Format extracted entities for a transformer model."""
    # Flatten all entities into a single list
    all_entities = []
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            all_entities.append({
                "text": entity["text"],
                "type": entity_type,
                "start": entity["start"],
                "end": entity["end"]
            })
    
    # Sort entities by start position
    all_entities.sort(key=lambda x: x["start"])
    
    # Create a list of entity mentions with their types
    entity_mentions = [f"{e['text']} [{e['type']}]" for e in all_entities]
    
    # Create the formatted input for transformer
    formatted_input = {
        "original_text": text,
        "entity_count": len(all_entities),
        "entities": all_entities,
        "formatted_text": " ".join(entity_mentions)
    }
    
    return formatted_input

def main():
    """Run a simple test of the NER module."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the NER extraction module')
    parser.add_argument('--real', action='store_true', help='Use real data if available (default: use synthetic)')
    args = parser.parse_args()
    
    # Set whether to use synthetic data
    global USE_SYNTHETIC_DATA
    USE_SYNTHETIC_DATA = not args.real
    
    print("COVID-19 Detection from Unstructured Medical Text")
    print("=" * 50)
    print("NER Module Test\n")
    
    # Display data source
    print(f"Using {'SYNTHETIC' if USE_SYNTHETIC_DATA else 'REAL'} data for this test")
    
    # Create output directory for results
    os.makedirs("output", exist_ok=True)
    
    # Get a clinical note
    print("Loading clinical note data...")
    clinical_note = get_sample_clinical_note(use_synthetic=USE_SYNTHETIC_DATA)
    print("\nCOVID-19 CLINICAL NOTE:")
    print("-" * 50)
    # Print a preview (first 500 chars) if note is long
    if len(clinical_note) > 500:
        print(clinical_note[:500] + "...\n[Note truncated for display, full text will be processed]")
    else:
        print(clinical_note)
    print("-" * 50)
    
    # Extract entities using rule-based NER
    print("\nExtracting entities using rule-based NER...")
    entities = extract_entities_with_rules(clinical_note)
    
    # Print extracted entities
    print("\nEXTRACTED ENTITIES:")
    print("-" * 50)
    for entity_type, entity_list in entities.items():
        print(f"\n{entity_type}:")
        if entity_list:
            for entity in entity_list[:10]:  # Show only the first 10 of each type
                print(f"  - {entity['text']}")
            if len(entity_list) > 10:
                print(f"  ... and {len(entity_list) - 10} more")
        else:
            print("  None found")
    
    # Format for transformer model
    transformer_input = format_for_transformer(clinical_note, entities)
    print("\nFORMATTED FOR TRANSFORMER MODEL:")
    print("-" * 50)
    formatted_text = transformer_input["formatted_text"]
    if len(formatted_text) > 500:
        print(formatted_text[:500] + "...\n[Text truncated for display]")
    else:
        print(formatted_text)
    
    # Save results to a file
    results = {
        "data_source": "real" if not USE_SYNTHETIC_DATA else "synthetic",
        "original_text": clinical_note,
        "entities": entities,
        "transformer_formatted": transformer_input
    }
    
    output_file = "output/ner_test_results_real.json" if not USE_SYNTHETIC_DATA else "output/ner_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\nEntity counts:")
    for entity_type, entity_list in entities.items():
        print(f"- {entity_type}: {len(entity_list)}")
    
    print("\nNext steps:")
    print("1. Use the extracted entities for the transformer classification model")
    print("2. Train a custom NER model on medical text")
    print("3. Integrate with the full COVID-19 detection pipeline")
    print("4. To try with real data, run: python test_ner.py --real")

if __name__ == "__main__":
    main()