"""
Named Entity Recognition (NER) tools for medical text analysis.

This module implements NER extraction for medical entities including:
- Symptoms
- Time expressions
- Severity indicators
- Medications
- Pre-existing conditions
"""

import os
import re
import json
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Entity types we're interested in extracting
ENTITY_TYPES = {
    "SYMPTOM": "Medical symptoms or clinical signs",
    "TIME": "Time expressions related to symptom onset or duration",
    "SEVERITY": "Indicators of symptom severity",
    "MEDICATION": "Medications mentioned in text",
    "CONDITION": "Pre-existing medical conditions",
    "SOCIAL": "Social factors like travel or exposure"
}

# Common COVID-19 symptoms for rule-based detection
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

class RuleBasedNER:
    """
    Simple rule-based NER for medical text.
    Uses regular expressions and keyword matching for entity extraction.
    """
    
    def __init__(self):
        """Initialize the rule-based NER model."""
        self.symptom_patterns = self._compile_patterns(COVID_SYMPTOMS, r'\b({})\b')
        self.time_patterns = self._compile_patterns(TIME_EXPRESSIONS, r'\b\w+\s+({})')
        self.severity_patterns = self._compile_patterns(SEVERITY_INDICATORS, r'\b({})\s+\w+')
        
        logger.info("Rule-based NER initialized")
    
    def _compile_patterns(self, terms, pattern_template):
        """Compile regex patterns for a list of terms."""
        patterns = []
        for term in terms:
            # Escape special regex characters in the term
            escaped_term = re.escape(term)
            # Format the pattern template with the escaped term
            pattern = pattern_template.format(escaped_term)
            # Compile the regex pattern
            patterns.append(re.compile(pattern, re.IGNORECASE))
        return patterns
    
    def extract_entities(self, text):
        """
        Extract entities from text using rule-based patterns.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary of extracted entities by type
        """
        entities = {
            "SYMPTOM": [],
            "TIME": [],
            "SEVERITY": []
        }
        
        # Extract symptoms
        for pattern in self.symptom_patterns:
            for match in pattern.finditer(text):
                entities["SYMPTOM"].append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract time expressions
        for pattern in self.time_patterns:
            for match in pattern.finditer(text):
                entities["TIME"].append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract severity indicators
        for pattern in self.severity_patterns:
            for match in pattern.finditer(text):
                entities["SEVERITY"].append({
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return entities

class SpacyNER:
    """
    spaCy-based NER for medical text.
    Supports both pre-trained models and custom training.
    """
    
    def __init__(self, model_name=None):
        """
        Initialize the spaCy NER model.
        
        Parameters:
        -----------
        model_name : str
            Name of the spaCy model to load or path to custom model
        """
        if model_name:
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"Model {model_name} not found. Loading en_core_web_sm.")
                self.nlp = spacy.load("en_core_web_sm")
        else:
            # Default to the small English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded default spaCy model: en_core_web_sm")
    
    def extract_entities(self, text):
        """
        Extract entities from text using spaCy.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary of extracted entities by type
        """
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entities[ent.label_].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        
        return dict(entities)
    
    def train(self, training_data, output_dir, n_iter=30):
        """
        Train a custom spaCy NER model.
        
        Parameters:
        -----------
        training_data : list
            List of (text, entities) tuples for training
        output_dir : str
            Directory to save the trained model
        n_iter : int
            Number of training iterations
            
        Returns:
        --------
        str
            Path to the saved model
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create a blank spaCy model
        nlp = spacy.blank("en")
        
        # Create the NER component
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner")
        else:
            ner = nlp.get_pipe("ner")
        
        # Add entity labels
        for _, annotations in training_data:
            for ent in annotations.get("entities", []):
                ner.add_label(ent[2])
        
        # Convert training data to spaCy format
        train_examples = []
        for text, annotations in training_data:
            entities = annotations.get("entities", [])
            example = {"text": text, "entities": entities}
            train_examples.append(example)
        
        # Get names of other pipes to disable during training
        pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        
        # Train the NER model
        with nlp.disable_pipes(*other_pipes), tqdm(total=n_iter) as pbar:
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(train_examples)
                losses = {}
                
                batches = minibatch(train_examples, size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts = [example["text"] for example in batch]
                    annotations = [{"entities": example["entities"]} for example in batch]
                    
                    nlp.update(
                        texts,
                        annotations,
                        drop=0.5,
                        losses=losses,
                        sgd=optimizer
                    )
                
                pbar.update(1)
                pbar.set_description(f"Losses: {losses}")
        
        # Save the model
        model_path = os.path.join(output_dir, "model")
        nlp.to_disk(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Update current model
        self.nlp = nlp
        
        return model_path

    def prepare_training_data(self, annotated_data, output_path=None):
        """
        Convert annotated data to spaCy training format.
        
        Parameters:
        -----------
        annotated_data : list
            List of annotated examples with text and entities
        output_path : str
            Path to save the DocBin file (optional)
            
        Returns:
        --------
        spacy.tokens.DocBin
            DocBin containing training examples
        """
        # Create DocBin
        doc_bin = DocBin()
        
        for text, annotations in annotated_data:
            doc = self.nlp.make_doc(text)
            ents = []
            
            for start, end, label in annotations.get("entities", []):
                span = doc.char_span(start, end, label=label)
                if span is None:
                    logger.warning(f"Entity [{start}:{end}] ({text[start:end]}) could not be aligned in '{text}'")
                else:
                    ents.append(span)
            
            doc.ents = ents
            doc_bin.add(doc)
        
        if output_path:
            doc_bin.to_disk(output_path)
            logger.info(f"Training data saved to {output_path}")
        
        return doc_bin

class TransformerNER:
    """
    Transformer-based NER for medical text using Hugging Face models.
    """
    
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1-squad"):
        """
        Initialize the transformer-based NER model.
        
        Parameters:
        -----------
        model_name : str
            Name of the transformer model to load
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
            logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            raise
    
    def extract_entities(self, text):
        """
        Extract entities from text using the transformer model.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary of extracted entities by type
        """
        results = self.nlp(text)
        
        # Group by entity type
        entities = defaultdict(list)
        current_entity = None
        current_type = None
        
        for i, entity in enumerate(results):
            # Check if this is a continuation of the previous entity
            if current_entity and entity["word"].startswith("##"):
                # Extend the current entity
                current_entity["text"] += entity["word"][2:]
                current_entity["end"] = entity["end"]
                current_entity["score"] = (current_entity["score"] + entity["score"]) / 2
            else:
                # If there was a previous entity, add it to the results
                if current_entity:
                    entities[current_type].append(current_entity)
                
                # Start a new entity
                current_type = entity["entity"]
                current_entity = {
                    "text": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "score": entity["score"]
                }
                
                # If this is the last entity, add it to the results
                if i == len(results) - 1:
                    entities[current_type].append(current_entity)
        
        return dict(entities)

def create_annotation_guide():
    """
    Create an annotation guide for medical NER.
    
    Returns:
    --------
    dict
        Annotation guide with entity types and examples
    """
    guide = {
        "entity_types": ENTITY_TYPES,
        "annotation_format": {
            "text": "Raw text",
            "entities": [
                [start_char, end_char, "ENTITY_TYPE"]
            ]
        },
        "examples": [
            {
                "text": "Patient reports severe cough for the past 3 days and mild fever since yesterday.",
                "entities": [
                    [15, 27, "SEVERITY SYMPTOM"],
                    [32, 49, "TIME"],
                    [54, 64, "SEVERITY SYMPTOM"],
                    [65, 83, "TIME"]
                ]
            },
            {
                "text": "Loss of taste and smell began 5 days after exposure to a confirmed case.",
                "entities": [
                    [0, 17, "SYMPTOM"],
                    [18, 38, "TIME"],
                    [39, 69, "SOCIAL"]
                ]
            }
        ]
    }
    
    return guide

def extract_entities_from_text(text, method="rule", model_name=None):
    """
    Extract medical entities from text using the specified method.
    
    Parameters:
    -----------
    text : str
        Text to analyze
    method : str
        Method to use for extraction ('rule', 'spacy', or 'transformer')
    model_name : str
        Name or path of the model to use (if applicable)
        
    Returns:
    --------
    dict
        Dictionary of extracted entities by type
    """
    if method == "rule":
        ner = RuleBasedNER()
    elif method == "spacy":
        ner = SpacyNER(model_name)
    elif method == "transformer":
        ner = TransformerNER(model_name)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return ner.extract_entities(text)

def format_entities_for_bert(text, entities):
    """
    Convert extracted entities to a format suitable for transformer input.
    
    Parameters:
    -----------
    text : str
        Original text
    entities : dict
        Dictionary of extracted entities by type
        
    Returns:
    --------
    dict
        Dictionary with formatted entity information for BERT
    """
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
    
    # Create the formatted input for BERT
    formatted_input = {
        "original_text": text,
        "entity_count": len(all_entities),
        "entities": all_entities,
        "formatted_text": " ".join(entity_mentions)
    }
    
    return formatted_input

def process_document_collection(documents, output_dir, method="rule", model_name=None):
    """
    Process a collection of documents and extract entities.
    
    Parameters:
    -----------
    documents : list
        List of document texts to process
    output_dir : str
        Directory to save the results
    method : str
        Method to use for extraction ('rule', 'spacy', or 'transformer')
    model_name : str
        Name or path of the model to use (if applicable)
        
    Returns:
    --------
    list
        List of processed documents with extracted entities
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, doc in enumerate(tqdm(documents, desc="Processing documents")):
        try:
            entities = extract_entities_from_text(doc, method, model_name)
            bert_input = format_entities_for_bert(doc, entities)
            
            result = {
                "document_id": f"doc_{i}",
                "text": doc,
                "entities": entities,
                "bert_formatted": bert_input
            }
            
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing document {i}: {e}")
    
    # Save results to disk
    output_path = os.path.join(output_dir, f"ner_results_{method}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved NER results to {output_path}")
    return results

def create_synthetic_training_data(num_examples=100):
    """
    Create synthetic training data for NER.
    
    Parameters:
    -----------
    num_examples : int
        Number of examples to generate
        
    Returns:
    --------
    list
        List of (text, annotations) tuples for training
    """
    from src.data_collection import generate_synthetic_clinical_note
    
    training_data = []
    for i in range(num_examples):
        # Generate a clinical note with random COVID status
        has_covid = np.random.random() > 0.5
        text = generate_synthetic_clinical_note(has_covid)
        
        # Create empty annotations
        annotations = {"entities": []}
        
        # Add to training data
        training_data.append((text, annotations))
    
    logger.info(f"Generated {num_examples} synthetic training examples")
    return training_data

if __name__ == "__main__":
    # Example usage
    from src.data_collection import generate_synthetic_clinical_note
    
    # Generate a sample clinical note
    sample_note = generate_synthetic_clinical_note(has_covid=True)
    print("Sample clinical note:")
    print(sample_note)
    print("-" * 80)
    
    # Extract entities using rule-based approach
    print("Rule-based NER results:")
    rule_entities = extract_entities_from_text(sample_note, method="rule")
    for entity_type, entities in rule_entities.items():
        print(f"\n{entity_type}:")
        for entity in entities:
            print(f"  - {entity['text']}")
    
    # Format for BERT
    bert_input = format_entities_for_bert(sample_note, rule_entities)
    print("\nFormatted for BERT:")
    print(bert_input["formatted_text"])