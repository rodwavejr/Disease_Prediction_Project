"""
Tools for collecting and preprocessing medical text data from various sources.
"""

import requests
import pandas as pd
import numpy as np
import os
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime
from nltk.tokenize import sent_tokenize
import logging

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
    # Example URL for CORD-19 dataset (this would need to be updated to a current source)
    # In a real implementation, you might download from Kaggle or similar
    cord19_url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv"
    
    try:
        logger.info(f"Fetching CORD-19 dataset sample (limit: {limit})")
        # For demonstration, we're just creating synthetic data
        # In a real implementation, you would:
        # df = pd.read_csv(cord19_url)
        
        # Create synthetic data for demonstration
        data = []
        for i in range(limit):
            data.append({
                "paper_id": f"paper_{i}",
                "title": f"COVID-19 Research Paper {i}",
                "abstract": generate_synthetic_medical_text(),
                "publish_time": "2020-05-01"
            })
        
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to disk
        output_path = os.path.join(output_dir, "cord19_sample.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved CORD-19 sample to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error fetching CORD-19 dataset: {e}")
        return None

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
    # CDC symptoms page URL
    cdc_url = "https://www.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html"
    
    try:
        logger.info("Fetching CDC COVID-19 guidelines")
        # In a real implementation, you would do:
        # response = requests.get(cdc_url)
        # soup = BeautifulSoup(response.text, 'html.parser')
        # content = extract_content_from_soup(soup)
        
        # For demonstration, create synthetic CDC guidelines
        content = {
            "source": "CDC",
            "url": cdc_url,
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
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to disk
        output_path = os.path.join(output_dir, "cdc_guidelines.json")
        with open(output_path, 'w') as f:
            json.dump(content, f, indent=2)
        
        logger.info(f"Saved CDC guidelines to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error fetching CDC guidelines: {e}")
        return None

def scrape_medical_forum(forum_url, num_pages=1, output_dir=None):
    """
    Scrape medical forum discussions about COVID-19 symptoms.
    
    Parameters:
    -----------
    forum_url : str
        URL of the medical forum
    num_pages : int
        Number of pages to scrape
    output_dir : str
        Directory to save the data
        
    Returns:
    --------
    list
        List of forum posts
    """
    # This is a placeholder for actual web scraping code
    # In a real implementation, you would use requests and BeautifulSoup
    
    logger.info(f"Scraping medical forum: {forum_url}")
    
    # Generate synthetic forum posts for demonstration
    posts = []
    for i in range(50):
        post = {
            "post_id": f"post_{i}",
            "title": f"Experiencing COVID symptoms - need advice (Thread {i})",
            "content": generate_synthetic_medical_text(is_forum=True),
            "date": "2021-08-15",
            "user_id": f"user_{np.random.randint(1, 100)}",
            "replies": []
        }
        
        # Add some replies
        num_replies = np.random.randint(0, 5)
        for j in range(num_replies):
            reply = {
                "reply_id": f"reply_{i}_{j}",
                "content": generate_synthetic_medical_text(is_reply=True),
                "date": "2021-08-16",
                "user_id": f"user_{np.random.randint(1, 100)}"
            }
            post["replies"].append(reply)
        
        posts.append(post)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "forum_posts.json")
        with open(output_path, 'w') as f:
            json.dump(posts, f, indent=2)
        logger.info(f"Saved forum posts to {output_path}")
    
    return posts

def download_synthetic_ehr_data(output_dir, num_records=100):
    """
    Download or generate synthetic EHR data for COVID-19 cases.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the data
    num_records : int
        Number of synthetic records to generate
        
    Returns:
    --------
    str
        Path to the saved file
    """
    logger.info(f"Generating {num_records} synthetic EHR records")
    
    records = []
    for i in range(num_records):
        has_covid = np.random.random() > 0.6  # 40% of records have COVID
        
        record = {
            "patient_id": f"PT{i:06d}",
            "age": np.random.randint(18, 85),
            "gender": np.random.choice(["M", "F"]),
            "admission_date": "2021-01-15",
            "chief_complaint": generate_chief_complaint(has_covid),
            "clinical_notes": generate_synthetic_clinical_note(has_covid),
            "diagnosis": "COVID-19" if has_covid else np.random.choice([
                "Influenza", "Common cold", "Allergic rhinitis", 
                "Bronchitis", "Pneumonia", "Upper respiratory infection"
            ]),
            "has_covid": has_covid
        }
        records.append(record)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(records)
    output_path = os.path.join(output_dir, "synthetic_ehr_data.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved synthetic EHR data to {output_path}")
    return output_path

def generate_synthetic_medical_text(is_forum=False, is_reply=False):
    """
    Generate synthetic medical text for demonstration purposes.
    
    Parameters:
    -----------
    is_forum : bool
        Whether this is a forum post
    is_reply : bool
        Whether this is a reply to a forum post
        
    Returns:
    --------
    str
        Synthetic medical text
    """
    # Symptoms
    symptoms = [
        "fever", "cough", "shortness of breath", "fatigue", 
        "muscle aches", "headache", "loss of taste", "loss of smell",
        "sore throat", "congestion", "nausea", "diarrhea"
    ]
    
    # Time expressions
    times = [
        "since yesterday", "for the past 3 days", "starting last week",
        "for about 24 hours", "intermittently for several days"
    ]
    
    # Severities
    severities = ["mild", "moderate", "severe", "intense", "slight"]
    
    # Generate text based on context
    if is_reply:
        templates = [
            "It sounds like you might have COVID-19. I had similar symptoms last month and tested positive.",
            "Those symptoms are consistent with COVID. You should get tested ASAP.",
            "Could be COVID or just a regular cold. The loss of taste/smell is a telling sign though.",
            "I experienced something similar. For me it was {symptom} and {symptom} {time}.",
            "When I had COVID, my main symptoms were {symptom} and {severity} {symptom}.",
        ]
    elif is_forum:
        templates = [
            "I've been experiencing {symptom} and {symptom} {time}. Could this be COVID?",
            "Started having {severity} {symptom} {time}, followed by {symptom}. Should I get tested?",
            "My symptoms: {symptom}, {symptom}, and {symptom}. No fever though. Worried it's COVID.",
            "Woke up with {severity} {symptom} and {symptom}. Has anyone had COVID start like this?",
            "Been having {symptom} {time}. Also {symptom} and {symptom}. What should I do?"
        ]
    else:
        # Clinical/academic text
        templates = [
            "Patient presented with {symptom} and {severity} {symptom} {time}.",
            "The most common symptoms observed were {symptom}, {symptom}, and {symptom}.",
            "Clinical manifestations include {symptom}, often accompanied by {symptom}.",
            "Initial symptoms typically include {severity} {symptom}, {symptom}, and {symptom}.",
            "COVID-19 patients frequently report {symptom} and {symptom} {time}."
        ]
    
    # Select random template and fill in placeholders
    template = np.random.choice(templates)
    text = template
    
    # Replace placeholders
    while "{symptom}" in text:
        text = text.replace("{symptom}", np.random.choice(symptoms), 1)
    
    while "{time}" in text:
        text = text.replace("{time}", np.random.choice(times), 1)
    
    while "{severity}" in text:
        text = text.replace("{severity}", np.random.choice(severities), 1)
    
    return text

def generate_chief_complaint(has_covid=True):
    """
    Generate a synthetic chief complaint.
    
    Parameters:
    -----------
    has_covid : bool
        Whether the patient has COVID-19
        
    Returns:
    --------
    str
        Synthetic chief complaint
    """
    if has_covid:
        templates = [
            "Fever and cough for 3 days",
            "Shortness of breath, fatigue",
            "Loss of taste and smell",
            "Fever, cough, body aches",
            "Persistent dry cough, fatigue"
        ]
    else:
        templates = [
            "Sore throat and congestion",
            "Seasonal allergies, runny nose",
            "Productive cough, no fever",
            "Ear pain and congestion",
            "Mild headache and fatigue"
        ]
    
    return np.random.choice(templates)

def generate_synthetic_clinical_note(has_covid=True):
    """
    Generate a synthetic clinical note.
    
    Parameters:
    -----------
    has_covid : bool
        Whether the patient has COVID-19
        
    Returns:
    --------
    str
        Synthetic clinical note
    """
    # Common note components
    subjective_templates = [
        "Patient is a {age}-year-old {gender} who presents with {complaint}.",
        "{age}-year-old {gender} with {complaint}.",
        "Patient reports {complaint}."
    ]
    
    covid_complaints = [
        "fever, dry cough, and fatigue for the past 3 days",
        "loss of taste and smell, fever, and body aches",
        "shortness of breath, dry cough, and fever",
        "fever, fatigue, and headache",
        "cough, loss of taste, and muscle aches"
    ]
    
    non_covid_complaints = [
        "nasal congestion, sneezing, and sore throat",
        "productive cough with green sputum",
        "ear pain and congestion",
        "sore throat and seasonal allergies",
        "headache and sinus pressure"
    ]
    
    age = np.random.randint(18, 85)
    gender = np.random.choice(["male", "female"])
    complaint = np.random.choice(covid_complaints if has_covid else non_covid_complaints)
    
    subjective = np.random.choice(subjective_templates).format(
        age=age, gender=gender, complaint=complaint
    )
    
    # Objective findings
    if has_covid:
        vitals = "Vitals: Temp 38.5°C, HR 95, BP 128/82, RR 18, O2 Sat 94% on room air."
        exam = "Physical exam reveals mild respiratory distress. Lungs with scattered rhonchi bilaterally. No rales or wheezes."
    else:
        vitals = "Vitals: Temp 37.2°C, HR 72, BP 120/78, RR 16, O2 Sat 98% on room air."
        exam = "Physical exam reveals clear lungs bilaterally. No respiratory distress. Mild pharyngeal erythema noted."
    
    objective = f"{vitals} {exam}"
    
    # Assessment and plan
    if has_covid:
        assessment = "Assessment: Clinical presentation consistent with COVID-19 infection."
        plan = "Plan: COVID-19 PCR test ordered. Patient advised to self-isolate pending results. Symptomatic treatment with acetaminophen for fever. Follow up in 2-3 days."
    else:
        conditions = ["Upper respiratory infection", "Seasonal allergies", "Acute bronchitis", "Common cold"]
        condition = np.random.choice(conditions)
        assessment = f"Assessment: {condition}, likely viral in origin."
        plan = f"Plan: Symptomatic treatment advised. Rest, fluids, and OTC medications as needed. Return if symptoms worsen or persist beyond 7 days."
    
    assessment_plan = f"{assessment} {plan}"
    
    # Combine note components
    note = f"{subjective}\n\n{objective}\n\n{assessment_plan}"
    
    return note

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

if __name__ == "__main__":
    # Example usage
    output_dir = "../data/raw"
    
    # Download data
    cord19_path = fetch_cord19_dataset(output_dir, limit=10)
    cdc_path = fetch_cdc_guidelines(output_dir)
    forum_posts = scrape_medical_forum("https://example-medical-forum.com", 
                                      output_dir=output_dir)
    ehr_path = download_synthetic_ehr_data(output_dir, num_records=20)
    
    print("Data collection complete. Files saved to:")
    print(f"- CORD-19 sample: {cord19_path}")
    print(f"- CDC guidelines: {cdc_path}")
    print(f"- Forum posts: {output_dir}/forum_posts.json")
    print(f"- Synthetic EHR data: {ehr_path}")