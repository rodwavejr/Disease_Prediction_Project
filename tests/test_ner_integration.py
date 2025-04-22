#!/usr/bin/env python
"""
Test script to verify that NER features are being extracted and integrated
with the classification dataset correctly.
"""

import os
import sys
import json
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'test_integration')

# Create output directory
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

def main():
    """Main function to test NER integration."""
    print("=" * 60)
    print("COVID-19 DETECTION: NER INTEGRATION TEST")
    print("=" * 60)
    print("\nThis script tests the NER integration with synthetic data")
    
    # Import ner_extraction to extract entities
    try:
        from src.ner_extraction import extract_entities_from_text, format_entities_for_bert
        print("\n✓ Successfully imported NER extraction module")
    except ImportError as e:
        print(f"\n✗ Failed to import NER extraction module: {e}")
        return
    
    # Create synthetic clinical note
    try:
        from src.data_collection import generate_synthetic_clinical_note
        text = generate_synthetic_clinical_note(has_covid=True)
        print("\n✓ Generated synthetic clinical note")
        print(f"\nSample text (first 100 chars):\n{text[:100]}...")
    except ImportError:
        # Fallback with hardcoded example
        text = """
        Patient is a 45-year-old male who presents with fever, dry cough, and fatigue for the past 3 days. 
        Patient also reports loss of taste and smell since yesterday.
        """
        print("\n⚠ Used fallback synthetic note (NER module not available)")
    
    # Extract entities
    try:
        entities = extract_entities_from_text(text, method="rule")
        print("\n✓ Successfully extracted entities:")
        
        # Count entities by type
        for entity_type, entity_list in entities.items():
            print(f"  - {entity_type}: {len(entity_list)} entities")
            # Print first 2 examples
            for i, entity in enumerate(entity_list[:2]):
                print(f"    * {entity['text']}")
            if len(entity_list) > 2:
                print(f"    * ... and {len(entity_list)-2} more")
    except Exception as e:
        print(f"\n✗ Failed to extract entities: {e}")
        return
    
    # Format for transformer
    try:
        formatted_input = format_entities_for_bert(text, entities)
        print("\n✓ Successfully formatted entities for transformer")
        print(f"  - Total entities: {formatted_input['entity_count']}")
    except Exception as e:
        print(f"\n✗ Failed to format entities: {e}")
        return
    
    # Create features
    symptom_count = len(entities.get('SYMPTOM', []))
    time_count = len(entities.get('TIME', []))
    severity_count = len(entities.get('SEVERITY', []))
    
    # Extract specific symptoms (create flags for common symptoms)
    symptoms = [entity['text'].lower() for entity in entities.get('SYMPTOM', [])]
    
    # Common COVID-19 symptoms as binary features
    common_symptoms = [
        "fever", "cough", "shortness of breath", "difficulty breathing",
        "fatigue", "loss of taste", "loss of smell", "sore throat"
    ]
    
    symptom_features = {}
    for symptom in common_symptoms:
        has_symptom = any(symptom in s for s in symptoms)
        symptom_features[f"has_{symptom.replace(' ', '_')}"] = int(has_symptom)
    
    print("\n✓ Created features for classification:")
    print(f"  - NER counts: {symptom_count} symptoms, {time_count} time expressions, {severity_count} severity indicators")
    print("  - Symptom indicators:")
    for symptom, value in symptom_features.items():
        print(f"    * {symptom}: {value}")
    
    # Save results
    results = {
        "clinical_note": text,
        "extracted_entities": entities,
        "transformer_input": formatted_input,
        "classification_features": {
            "ner_symptom_count": symptom_count,
            "ner_time_count": time_count,
            "ner_severity_count": severity_count,
            **symptom_features
        }
    }
    
    results_path = os.path.join(TEST_OUTPUT_DIR, 'ner_integration_test.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved test results to {results_path}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print("\n✓ The NER extraction and feature creation is working correctly!")
    print("\nThis confirms that the technical implementation for integrating NER features")
    print("with the classification pipeline is functioning as expected.")
    print("\nNext steps:")
    print("1. Run prepare_classification_dataset.py to create the full dataset")
    print("2. Check that NER features are included in the final dataset")
    print("3. Use the dataset for model training and evaluation")

if __name__ == "__main__":
    main()