"""
Tools for collecting and preprocessing medical text data from various sources.

This module includes both synthetic data generation functions (for demonstration/development)
and real data collection functions. The synthetic generation will be replaced by real data
as it becomes available.
"""

import os
import re
import json
from datetime import datetime
import logging

# Handle import errors gracefully for better user experience
try:
    import requests
except ImportError:
    print("Warning: 'requests' package not found. Real data fetching will be disabled.")
    print("Install with: pip install requests")
    requests = None

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Warning: 'pandas' or 'numpy' package not found. Some functionality will be limited.")
    print("Install with: pip install pandas numpy")
    pd = None
    np = None
    
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Warning: 'beautifulsoup4' package not found. HTML parsing will be disabled.")
    print("Install with: pip install beautifulsoup4")
    BeautifulSoup = None
    
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    print("Warning: 'nltk' package not found. Sentence tokenization will use basic fallback.")
    print("Install with: pip install nltk")
    
    # Fallback tokenizer
    def sent_tokenize(text):
        """Simple fallback tokenizer that splits on periods."""
        return [s.strip() + '.' for s in text.split('.') if s.strip()]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of COVID-19 related symptoms for filtering
COVID_SYMPTOMS = [
    'fever', 'cough', 'shortness of breath', 'difficulty breathing', 
    'fatigue', 'muscle pain', 'body ache', 'headache', 'loss of taste',
    'loss of smell', 'sore throat', 'congestion', 'runny nose', 'nausea',
    'vomiting', 'diarrhea', 'chills'
]

# This flag determines whether to use synthetic or real data
# Set to False when real data is available
USE_SYNTHETIC_DATA = True

def generate_synthetic_medical_text(covid_focus=True, length='medium'):
    """
    Generate synthetic medical text for development and testing.
    
    Parameters:
    -----------
    covid_focus : bool
        Whether to focus on COVID-19 related content
    length : str
        Length of text to generate ('short', 'medium', 'long')
        
    Returns:
    --------
    str
        Generated medical text
    """
    # Base symptoms for text generation
    if covid_focus:
        symptoms = [
            "fever of 101.3F", "persistent dry cough", "fatigue", 
            "shortness of breath", "loss of taste", "loss of smell",
            "body aches", "headache", "sore throat", "congestion",
            "nausea", "diarrhea"
        ]
        
        conditions = [
            "COVID-19", "SARS-CoV-2 infection", "coronavirus",
            "respiratory infection", "viral pneumonia"
        ]
        
        severity = [
            "mild", "moderate", "severe", "critical",
            "requiring hospitalization", "requiring supplemental oxygen"
        ]
    else:
        symptoms = [
            "cough", "fever", "headache", "nausea", "vomiting", 
            "abdominal pain", "chest pain", "back pain", "dizziness",
            "fatigue", "weakness", "rash"
        ]
        
        conditions = [
            "common cold", "influenza", "pneumonia", "bronchitis",
            "strep throat", "sinusitis", "gastroenteritis"
        ]
        
        severity = [
            "mild", "moderate", "severe",
            "self-limiting", "improving", "worsening"
        ]
    
    # Medication mentions
    medications = [
        "acetaminophen", "ibuprofen", "azithromycin", "albuterol inhaler",
        "dexamethasone", "remdesivir", "hydroxychloroquine", "zinc supplements",
        "vitamin D", "cough suppressant"
    ]
    
    # Time expressions
    time_expressions = [
        "started 3 days ago", "began yesterday", "for the past week",
        "since last Tuesday", "for 5 days", "intermittent for 2 weeks",
        "worsening over 24 hours", "improving in the last 2 days"
    ]
    
    # Generate text based on length
    if length == 'short':
        num_sentences = np.random.randint(2, 5)
    elif length == 'medium':
        num_sentences = np.random.randint(5, 10)
    else:  # long
        num_sentences = np.random.randint(10, 15)
    
    # Template sentences
    templates = [
        "Patient presents with {symptom}, which {time}.",
        "Patient reports {symptom} and {symptom}, {severity} in nature.",
        "Examination reveals {symptom}.",
        "Patient was diagnosed with {condition} and prescribed {medication}.",
        "Symptoms include {symptom}, {symptom}, and {symptom}.",
        "Patient describes {symptom} {time}.",
        "Patient denies {symptom} but confirms {symptom}.",
        "Treatment with {medication} was initiated for {symptom}.",
        "{condition} is suspected based on {symptom} and {symptom}.",
        "Patient has a history of {condition} and now presents with {symptom}."
    ]
    
    # Generate sentences
    sentences = []
    for _ in range(num_sentences):
        template = np.random.choice(templates)
        
        # Fill in template with random choices
        sentence = template
        while "{symptom}" in sentence:
            sentence = sentence.replace("{symptom}", np.random.choice(symptoms), 1)
        while "{condition}" in sentence:
            sentence = sentence.replace("{condition}", np.random.choice(conditions), 1)
        while "{severity}" in sentence:
            sentence = sentence.replace("{severity}", np.random.choice(severity), 1)
        while "{medication}" in sentence:
            sentence = sentence.replace("{medication}", np.random.choice(medications), 1)
        while "{time}" in sentence:
            sentence = sentence.replace("{time}", np.random.choice(time_expressions), 1)
        
        sentences.append(sentence)
    
    # Join sentences into a paragraph
    text = " ".join(sentences)
    
    return text

def fetch_cord19_dataset(output_dir, limit=100):
    """
    Fetch a sample of the CORD-19 dataset and save it to disk.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the data
    limit : int
        Maximum number of papers to fetch
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path will be the same for both synthetic and real data
    output_path = os.path.join(output_dir, "cord19_sample.csv")
    
    # Check if we should use synthetic data
    if USE_SYNTHETIC_DATA:
        logger.info(f"Generating synthetic CORD-19 data (limit: {limit})")
        
        # Generate synthetic data for demonstration
        data = []
        for i in range(limit):
            data.append({
                "paper_id": f"paper_{i}",
                "title": f"COVID-19 Research Paper {i}",
                "abstract": generate_synthetic_medical_text(),
                "publish_time": "2020-05-01"
            })
        
        df = pd.DataFrame(data)
        
        # Save to disk
        df.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic CORD-19 sample to {output_path}")
        
        return output_path
    
    else:
        # URL for CORD-19 dataset
        cord19_url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv"
        
        try:
            logger.info(f"Fetching real CORD-19 dataset sample (limit: {limit})")
            
            # Create a requests session
            session = requests.Session()
            
            # Download the dataset
            response = session.get(cord19_url)
            response.raise_for_status()
            
            # Save raw CSV
            raw_path = os.path.join(output_dir, "cord19_full.csv")
            with open(raw_path, 'wb') as f:
                f.write(response.content)
            
            # Read and limit the dataset
            # Use error_bad_lines=False (renamed to on_bad_lines='skip' in newer pandas) to skip problematic lines
            try:
                # For newer pandas versions
                df = pd.read_csv(raw_path, on_bad_lines='skip')
                logger.info("Using pandas with on_bad_lines='skip' to handle malformed lines")
            except TypeError:
                # For older pandas versions
                df = pd.read_csv(raw_path, error_bad_lines=False)
                logger.info("Using pandas with error_bad_lines=False to handle malformed lines")
            
            logger.info(f"Successfully loaded CORD-19 dataset with {len(df)} entries")
            
            # Select a random sample to avoid bias from only taking the first rows
            if len(df) > limit:
                df_sample = df.sample(limit, random_state=42)
            else:
                df_sample = df
                
            logger.info(f"Selected {len(df_sample)} entries from CORD-19 dataset")
            
            # Save sample to disk
            df_sample.to_csv(output_path, index=False)
            logger.info(f"Saved real CORD-19 sample to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error fetching real CORD-19 dataset: {e}")
            
            # If real data fetch fails, fall back to synthetic
            logger.info("Falling back to synthetic CORD-19 data")
            return fetch_cord19_dataset(output_dir, limit)

def fetch_cdc_guidelines(output_dir):
    """
    Fetch CDC COVID-19 symptom guidelines and save to disk.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the data
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path for both synthetic and real data
    output_path = os.path.join(output_dir, "cdc_guidelines.json")
    
    # Use synthetic or real data based on flag
    if USE_SYNTHETIC_DATA:
        logger.info("Creating synthetic CDC COVID-19 guidelines")
        
        # Create synthetic CDC guidelines
        content = {
            "source": "CDC",
            "url": "https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html",
            "fetch_date": datetime.now().strftime("%Y-%m-%d"),
            "symptoms": [
                {
                    "name": "Fever or chills",
                    "common": True,
                    "timeframe": "2-14 days after exposure"
                },
                {
                    "name": "Cough",
                    "common": True,
                    "timeframe": "2-14 days after exposure"
                },
                {
                    "name": "Shortness of breath",
                    "common": True,
                    "timeframe": "2-14 days after exposure"
                },
                {
                    "name": "Fatigue",
                    "common": True,
                    "timeframe": "2-14 days after exposure"
                },
                {
                    "name": "Muscle or body aches",
                    "common": True,
                    "timeframe": "2-14 days after exposure"
                },
                {
                    "name": "Headache",
                    "common": True,
                    "timeframe": "2-14 days after exposure"
                },
                {
                    "name": "New loss of taste or smell",
                    "common": True,
                    "timeframe": "2-14 days after exposure"
                }
            ],
            "emergency_signs": [
                "Trouble breathing",
                "Persistent pain or pressure in the chest",
                "New confusion",
                "Inability to wake or stay awake",
                "Pale, gray, or blue-colored skin, lips, or nail beds"
            ]
        }
        
        # Save to disk
        with open(output_path, 'w') as f:
            json.dump(content, f, indent=2)
        
        logger.info(f"Saved synthetic CDC guidelines to {output_path}")
        return output_path
        
    else:
        # CDC symptoms page URL
        cdc_url = "https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html"
        
        try:
            logger.info("Fetching real CDC COVID-19 guidelines")
            
            # Create a requests session
            session = requests.Session()
            
            # Get the CDC page
            response = session.get(cdc_url)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant content (this is simplified and would need to be adapted to the actual page structure)
            symptoms_section = soup.find('div', {'id': 'symptoms-body-text'})
            
            # Create content object
            content = {
                "source": "CDC",
                "url": cdc_url,
                "fetch_date": datetime.now().strftime("%Y-%m-%d"),
                "html_content": str(symptoms_section),
                "text_content": symptoms_section.get_text() if symptoms_section else ""
            }
            
            # Save to disk
            with open(output_path, 'w') as f:
                json.dump(content, f, indent=2)
            
            logger.info(f"Saved real CDC guidelines to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error fetching real CDC guidelines: {e}")
            
            # Fall back to synthetic data if real fetch fails
            logger.info("Falling back to synthetic CDC guidelines")
            return fetch_cdc_guidelines(output_dir)

def preprocess_text(text):
    """
    Preprocess text for NLP analysis.
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
        
    Returns:
    --------
    str
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()

def filter_covid_relevant_text(text):
    """
    Filter text to keep only sentences relevant to COVID-19 symptoms.
    
    Parameters:
    -----------
    text : str
        Text to filter
        
    Returns:
    --------
    str
        Filtered text containing only COVID-relevant sentences
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Filter sentences containing COVID symptoms
    covid_sentences = []
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(symptom in lower_sentence for symptom in COVID_SYMPTOMS):
            covid_sentences.append(sentence)
    
    return ' '.join(covid_sentences)

def main():
    """Main function to handle command-line execution."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download data for COVID-19 detection project')
    parser.add_argument('--real', action='store_true', help='Use real data instead of synthetic')
    parser.add_argument('--output', default='../data/raw', help='Output directory path')
    parser.add_argument('--limit', type=int, default=10, help='Limit for number of records to fetch')
    
    args = parser.parse_args()
    
    # Set the data source flag without needing global (since we're in a function)
    global USE_SYNTHETIC_DATA  # Declare global before modifying
    USE_SYNTHETIC_DATA = not args.real
    
    return args

if __name__ == "__main__":
    args = main()
    
    output_dir = args.output
    
    # Print data source
    if USE_SYNTHETIC_DATA:
        print("Using SYNTHETIC data for demonstration purposes")
    else:
        print("Using REAL data from online sources")
    
    # Download data
    cord19_path = fetch_cord19_dataset(output_dir, limit=args.limit)
    cdc_path = fetch_cdc_guidelines(output_dir)
    
    print("\nData collection complete. Files saved to:")
    print(f"- CORD-19 sample: {cord19_path}")
    print(f"- CDC guidelines: {cdc_path}")
    
    print("\nNext steps:")
    print("1. Run the Jupyter notebooks to explore the data")
    print("2. Use the NER pipeline to extract entities from the text")
    print("3. Train the classification model on the extracted entities")