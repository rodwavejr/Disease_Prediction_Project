"""
Generate synthetic medical data for COVID-19 detection project.
"""

import os
import json
import csv
import random
from datetime import datetime, timedelta

def generate_patient_id():
    """Generate a random patient ID."""
    return f"PT{random.randint(10000, 99999)}"

def generate_age():
    """Generate a random age between 18 and 90."""
    return random.randint(18, 90)

def generate_gender():
    """Generate a random gender."""
    return random.choice(["Male", "Female"])

def generate_covid_symptoms():
    """Generate a list of COVID-19 symptoms."""
    all_symptoms = [
        "fever", "cough", "shortness of breath", "difficulty breathing", 
        "fatigue", "muscle pain", "body ache", "headache", "loss of taste",
        "loss of smell", "sore throat", "congestion", "runny nose", "nausea",
        "vomiting", "diarrhea", "chills"
    ]
    # Select 3-5 random symptoms
    num_symptoms = random.randint(3, 5)
    return random.sample(all_symptoms, num_symptoms)

def generate_non_covid_symptoms():
    """Generate a list of non-COVID-19 symptoms."""
    all_symptoms = [
        "sore throat", "nasal congestion", "sneezing", "ear pain",
        "headache", "runny nose", "productive cough", "itchy eyes",
        "fatigue", "body ache", "mild fever", "dizziness"
    ]
    # Select 2-4 random symptoms
    num_symptoms = random.randint(2, 4)
    return random.sample(all_symptoms, num_symptoms)

def generate_date(days_ago_min=1, days_ago_max=30):
    """Generate a random date within the specified range."""
    days_ago = random.randint(days_ago_min, days_ago_max)
    date = datetime.now() - timedelta(days=days_ago)
    return date.strftime("%Y-%m-%d")

def generate_hospital():
    """Generate a random hospital name."""
    hospitals = [
        "City General Hospital", "Memorial Medical Center", 
        "University Hospital", "Community Health Center",
        "Regional Medical Center"
    ]
    return random.choice(hospitals)

def generate_clinical_note(has_covid=True, symptoms=None, age=None, gender=None):
    """Generate a synthetic clinical note."""
    age = age or random.randint(18, 90)
    gender = gender or random.choice(["male", "female"])
    
    if has_covid:
        if not symptoms:
            symptoms = generate_covid_symptoms()
        
        symptom_text = ", ".join(symptoms[:-1]) + f" and {symptoms[-1]}" if len(symptoms) > 1 else symptoms[0]
        duration = random.choice(["2 days", "3 days", "5 days", "a week", "several days"])
        
        templates = [
            f"Patient is a {age}-year-old {gender} who presents with {symptom_text} for {duration}.",
            f"{age}-year-old {gender} with {symptom_text} starting {duration} ago.",
            f"Patient reports {symptom_text} that began {duration} ago."
        ]
        
        subjective = random.choice(templates)
        if "loss of taste" in symptoms or "loss of smell" in symptoms:
            subjective += " Patient also reports loss of taste and smell."
        
        vitals = f"Vitals: Temp {random.uniform(38.0, 39.2):.1f}°C, HR {random.randint(85, 100)}, BP {random.randint(120, 140)}/{random.randint(70, 90)}, RR {random.randint(18, 24)}, O2 Sat {random.randint(92, 95)}% on room air."
        exam = "Physical exam reveals mild respiratory distress. Lungs with scattered rhonchi bilaterally. No rales or wheezes."
        objective = f"{vitals} {exam}"
        
        assessment = "Assessment: Clinical presentation consistent with COVID-19 infection."
        
        plan = "Plan: COVID-19 PCR test ordered. Patient advised to self-isolate pending results. Symptomatic treatment with acetaminophen for fever. Follow up in 2-3 days."
        
    else:
        if not symptoms:
            symptoms = generate_non_covid_symptoms()
        
        symptom_text = ", ".join(symptoms[:-1]) + f" and {symptoms[-1]}" if len(symptoms) > 1 else symptoms[0]
        duration = random.choice(["2 days", "3 days", "5 days", "a week"])
        
        templates = [
            f"Patient is a {age}-year-old {gender} who presents with {symptom_text} for {duration}.",
            f"{age}-year-old {gender} with {symptom_text} starting {duration} ago.",
            f"Patient reports {symptom_text} that began {duration} ago."
        ]
        
        subjective = random.choice(templates)
        
        vitals = f"Vitals: Temp {random.uniform(36.8, 37.8):.1f}°C, HR {random.randint(60, 85)}, BP {random.randint(110, 130)}/{random.randint(70, 85)}, RR {random.randint(12, 18)}, O2 Sat {random.randint(96, 99)}% on room air."
        
        if "sore throat" in symptoms:
            exam = "Physical exam shows erythematous pharynx with tonsillar exudate. No respiratory distress."
        elif "congestion" in symptoms or "runny nose" in symptoms:
            exam = "Physical exam shows clear nasal discharge and mild nasal congestion. No respiratory distress."
        else:
            exam = "Physical exam is unremarkable with clear lungs bilaterally. No respiratory distress."
        
        objective = f"{vitals} {exam}"
        
        conditions = ["Upper respiratory infection", "Seasonal allergies", "Acute bronchitis", "Common cold", "Sinusitis"]
        condition = random.choice(conditions)
        assessment = f"Assessment: {condition}, likely viral in origin."
        
        plan = f"Plan: Symptomatic treatment advised. Rest, fluids, and OTC medications as needed. Return if symptoms worsen or persist beyond 7 days."
    
    # Combine note components
    note = f"{subjective}\n\n{objective}\n\n{assessment} {plan}"
    
    return note

def generate_covid_test_result(has_covid=True):
    """Generate a COVID-19 test result."""
    if has_covid:
        return "Positive" if random.random() < 0.95 else "Negative"  # 5% false negatives
    else:
        return "Negative" if random.random() < 0.98 else "Positive"  # 2% false positives

def generate_dataset(num_records=100, covid_ratio=0.5, output_file="data/raw/medical_records.csv"):
    """
    Generate a synthetic dataset of medical records.
    
    Parameters:
    -----------
    num_records : int
        Number of records to generate
    covid_ratio : float
        Ratio of COVID-19 positive cases
    output_file : str
        Path to save the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Define column names
    columns = [
        "record_id", "patient_id", "age", "gender", "admission_date", 
        "hospital", "symptoms", "clinical_note", "covid_test_result", 
        "has_covid"
    ]
    
    # Generate records
    records = []
    for i in range(num_records):
        has_covid = random.random() < covid_ratio
        
        # Generate basic info
        record_id = f"REC{i+1:06d}"
        patient_id = generate_patient_id()
        age = generate_age()
        gender = generate_gender()
        admission_date = generate_date()
        hospital = generate_hospital()
        
        # Generate symptoms and clinical note
        symptoms = generate_covid_symptoms() if has_covid else generate_non_covid_symptoms()
        clinical_note = generate_clinical_note(has_covid, symptoms, age, gender.lower())
        
        # Generate test result
        covid_test_result = generate_covid_test_result(has_covid)
        
        # Add to records
        records.append({
            "record_id": record_id,
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "admission_date": admission_date,
            "hospital": hospital,
            "symptoms": ", ".join(symptoms),
            "clinical_note": clinical_note,
            "covid_test_result": covid_test_result,
            "has_covid": 1 if has_covid else 0
        })
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    
    # Also save as JSON for convenience
    json_output = output_file.replace('.csv', '.json')
    with open(json_output, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Generated {num_records} records:")
    print(f"- CSV: {output_file}")
    print(f"- JSON: {json_output}")
    print(f"- COVID-19 positive cases: {int(num_records * covid_ratio)} ({covid_ratio*100:.0f}%)")
    print(f"- COVID-19 negative cases: {num_records - int(num_records * covid_ratio)} ({(1-covid_ratio)*100:.0f}%)")
    
    return output_file

if __name__ == "__main__":
    # Generate datasets with different sizes and ratios
    generate_dataset(
        num_records=500, 
        covid_ratio=0.4,  # 40% COVID-19 positive
        output_file="data/raw/medical_records.csv"
    )
    
    generate_dataset(
        num_records=100,
        covid_ratio=0.5,  # 50% COVID-19 positive (balanced)
        output_file="data/raw/balanced_dataset.csv"
    )
    
    # Generate a smaller test set
    generate_dataset(
        num_records=50,
        covid_ratio=0.4,
        output_file="data/raw/test_dataset.csv"
    )