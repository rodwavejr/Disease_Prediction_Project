#!/usr/bin/env python
"""
Test script to demonstrate NER extraction on MIMIC-IV data.
This script uses a combination of synthetic data when MIMIC data is not available.
"""

import os
import sys
import json
import re
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import pandas - required for MIMIC data
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("pandas not available, will use synthetic data only")
    PANDAS_AVAILABLE = False

# Import other modules we need
from test_ner import extract_entities_with_rules, format_for_transformer

# MIMIC data directory
MIMIC_DIR = os.path.join('data', 'external', 'mimic')

def get_mimic_text_sample():
    """Try to get a text sample from MIMIC data or fall back to synthetic."""
    if not PANDAS_AVAILABLE:
        logger.info("pandas not available, using synthetic sample")
        from test_ner import get_sample_clinical_note
        return get_sample_clinical_note(use_synthetic=True)
    
    # Try to load MIMIC text data
    try:
        # Try OMR data first
        omr_path = os.path.join(MIMIC_DIR, 'omr_sample.csv')
        if os.path.exists(omr_path):
            omr_df = pd.read_csv(omr_path)
            
            # Find rows with text content
            for _, row in omr_df.iterrows():
                # Get all string columns
                text_columns = [col for col in omr_df.columns if isinstance(row[col], str)]
                
                # Combine text from all columns
                if text_columns:
                    text_parts = []
                    for col in text_columns:
                        if isinstance(row[col], str) and len(row[col].strip()) > 0:
                            text_parts.append(f"{col}: {row[col]}")
                    
                    # If we found text, use it
                    if text_parts:
                        return "\\n".join(text_parts)
        
        logger.warning("No suitable text found in MIMIC data, using synthetic sample")
        from test_ner import get_sample_clinical_note
        return get_sample_clinical_note(use_synthetic=True)
        
    except Exception as e:
        logger.error(f"Error loading MIMIC text: {e}")
        from test_ner import get_sample_clinical_note
        return get_sample_clinical_note(use_synthetic=True)

def main():
    """Run a test of the NER module with MIMIC data."""
    parser = argparse.ArgumentParser(description='Test the NER extraction module with MIMIC data')
    parser.add_argument('--output', default='output/mimic_ner_results.json', 
                       help='Path to save results JSON file')
    args = parser.parse_args()
    
    print("COVID-19 Detection from MIMIC Data")
    print("=" * 50)
    print("NER Module Test with MIMIC\n")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get clinical note text
    print("Loading clinical note data from MIMIC...")
    clinical_note = get_mimic_text_sample()
    
    print("\nCLINICAL NOTE:")
    print("-" * 50)
    # Print a preview (first 500 chars) if note is long
    if len(clinical_note) > 500:
        print(clinical_note[:500] + "...\n[Note truncated for display, full text will be processed]")
    else:
        print(clinical_note)
    print("-" * 50)
    
    # Extract entities
    print("\nExtracting entities using rule-based NER...")
    entities = extract_entities_with_rules(clinical_note)
    
    # Print extracted entities
    print("\nEXTRACTED ENTITIES:")
    print("-" * 50)
    for entity_type, entity_list in entities.items():
        print(f"\n{entity_type}:")
        if entity_list:
            for entity in entity_list[:10]:  # Show only first 10
                print(f"  - {entity['text']}")
            if len(entity_list) > 10:
                print(f"  ... and {len(entity_list) - 10} more")
        else:
            print("  None found")
    
    # Format for transformer
    transformer_input = format_for_transformer(clinical_note, entities)
    
    # Save results
    results = {
        "data_source": "MIMIC" if PANDAS_AVAILABLE else "synthetic",
        "original_text": clinical_note,
        "entities": entities,
        "transformer_formatted": transformer_input
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print("\nEntity counts:")
    for entity_type, entity_list in entities.items():
        print(f"- {entity_type}: {len(entity_list)}")
    
    print("\nNext steps:")
    print("1. Modify the NER extraction rules to better handle clinical language")
    print("2. Train custom NER models on MIMIC data")
    print("3. Integrate extracted entities with the classification model")

if __name__ == "__main__":
    main()