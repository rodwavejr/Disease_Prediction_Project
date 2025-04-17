"""
Test script for the NER extraction module.
Run this script to see the NER module in action with synthetic data.
"""

import os
import json
import re
import random
from collections import defaultdict

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

def generate_synthetic_clinical_note():
    """Generate a synthetic clinical note with COVID-19 symptoms."""
    subjective = "Patient is a 45-year-old male who presents with fever, dry cough, and fatigue for the past 3 days. Patient also reports loss of taste and smell since yesterday."
    
    objective = "Vitals: Temp 38.5°C, HR 95, BP 128/82, RR 18, O2 Sat 94% on room air. Physical exam reveals mild respiratory distress. Lungs with scattered rhonchi bilaterally. No rales or wheezes."
    
    assessment = "Assessment: Clinical presentation consistent with COVID-19 infection."
    
    plan = "Plan: COVID-19 PCR test ordered. Patient advised to self-isolate pending results. Symptomatic treatment with acetaminophen for fever. Follow up in 2-3 days."
    
    # Combine note components
    note = f"{subjective}\n\n{objective}\n\n{assessment} {plan}"
    
    return note

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
    print("COVID-19 Detection from Unstructured Medical Text")
    print("=" * 50)
    print("NER Module Test\n")
    
    # Create output directory for results
    os.makedirs("output", exist_ok=True)
    
    # Generate a synthetic clinical note
    print("Generating a synthetic COVID-19 clinical note...")
    clinical_note = generate_synthetic_clinical_note()
    print("\nCOVID-19 CLINICAL NOTE:")
    print("-" * 50)
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
        for entity in entity_list:
            print(f"  - {entity['text']}")
    
    # Format for transformer model
    transformer_input = format_for_transformer(clinical_note, entities)
    print("\nFORMATTED FOR TRANSFORMER MODEL:")
    print("-" * 50)
    print(transformer_input["formatted_text"])
    
    # Save results to a file
    results = {
        "original_text": clinical_note,
        "entities": entities,
        "transformer_formatted": transformer_input
    }
    
    with open("output/ner_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to output/ner_test_results.json")
    print("\nNext steps:")
    print("1. Use the extracted entities for the transformer classification model")
    print("2. Train a custom NER model on medical text")
    print("3. Integrate with the full COVID-19 detection pipeline")

if __name__ == "__main__":
    main()