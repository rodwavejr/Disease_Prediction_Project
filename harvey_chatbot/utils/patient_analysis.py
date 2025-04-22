import json
import os
import re
import random
import pandas as pd
from datetime import datetime

# Add parent directory to path to access project modules if needed
import sys
sys.path.append(os.path.abspath('../..'))

def calculate_covid_probability(entities, lab_results):
    """
    Calculate the probability of COVID-19 based on symptoms and lab results
    
    Parameters:
    - entities: Dictionary containing extracted symptoms, time expressions, and severity indicators
    - lab_results: Dictionary of lab test results
    
    Returns:
    - Probability score between 0 and 1
    """
    # Base probability starts at 0.1
    probability = 0.1
    
    # List of high COVID specificity symptoms
    covid_specific_symptoms = {
        "fever": 0.1,
        "cough": 0.1,
        "shortness of breath": 0.15,
        "loss of taste": 0.2,
        "loss of smell": 0.2,
        "fatigue": 0.05,
        "body aches": 0.05,
        "sore throat": 0.05,
        "headache": 0.05,
        "chills": 0.05,
        "difficulty breathing": 0.15
    }
    
    # Check for symptoms
    detected_symptoms = set()
    if 'SYMPTOM' in entities:
        for symptom_entity in entities['SYMPTOM']:
            symptom_text = symptom_entity['text'].lower()
            
            # Check against COVID-specific symptoms
            for covid_symptom, weight in covid_specific_symptoms.items():
                if covid_symptom in symptom_text and covid_symptom not in detected_symptoms:
                    probability += weight
                    detected_symptoms.add(covid_symptom)
    
    # Check severity
    has_severe_symptoms = False
    if 'SEVERITY' in entities:
        for severity in entities['SEVERITY']:
            severity_text = severity['text'].lower()
            if any(term in severity_text for term in ['severe', 'serious', 'acute']):
                has_severe_symptoms = True
                probability += 0.1
    
    # Check lab results if available
    if lab_results:
        # Check lymphocyte count (low in COVID)
        if lab_results.get('lymphocyte_count') and lab_results['lymphocyte_count'] < 1.0:
            probability += 0.1
        
        # Check CRP (elevated in COVID)
        if lab_results.get('c_reactive_protein') and lab_results['c_reactive_protein'] > 10:
            probability += 0.1
            
        # Check d-dimer (elevated in COVID)
        if lab_results.get('d_dimer') and lab_results['d_dimer'] > 0.5:
            probability += 0.1
            
        # Check ferritin (elevated in COVID)
        if lab_results.get('ferritin') and lab_results['ferritin'] > 300:
            probability += 0.05
    
    # Cap probability at 0.95 (never 100% certain)
    probability = min(probability, 0.95)
    
    return round(probability, 2)

def analyze_patient_text(user_message, patient_id=None):
    """
    Analyze the user's message and generate a relevant response about the patient
    
    Parameters:
    - user_message: The message from the user/doctor
    - patient_id: Optional ID of the patient being discussed
    
    Returns:
    - Response text
    """
    # Load patient data if patient_id is provided
    patient_data = None
    try:
        if patient_id:
            try:
                with open('../data/sample_patients.json', 'r') as f:
                    patients = json.load(f)
                    patient_data = next((p for p in patients if p["id"] == patient_id), None)
                    
                # Load clinical notes
                try:
                    with open(f'../data/clinical_notes/{patient_id}.txt', 'r') as f:
                        clinical_notes = f.read()
                        if patient_data:
                            patient_data['clinical_notes'] = clinical_notes
                except FileNotFoundError:
                    print(f"Clinical notes file not found for patient {patient_id}")
            except Exception as e:
                print(f"Error loading patient data: {e}")
    except Exception as e:
        print(f"Error in initial patient data loading: {e}")
    
    # Simple rule-based response system
    user_message = user_message.lower()
    
    try:
        # Check for specific questions about COVID
        if "covid" in user_message and any(word in user_message for word in ["probability", "likelihood", "risk", "chance"]):
            if patient_data:
                probability = patient_data.get('covid_probability', 0.5)
                return f"Based on the symptoms and lab results, the patient has a {probability*100:.1f}% likelihood of COVID-19. Key factors include: {', '.join(patient_data.get('symptoms', ['No symptoms recorded']))}."
            else:
                return "I need patient information to calculate COVID probability. Please select a patient first."
        
        # Questions about symptoms
        elif any(keyword in user_message for keyword in ["symptom", "symptoms", "presenting with", "complaining of", "chief complaint"]):
            if patient_data:
                symptoms = patient_data.get('symptoms', ['No symptoms recorded'])
                return f"The patient is presenting with: {', '.join(symptoms)}. The primary concerns are {', '.join(symptoms[:2])}."
            else:
                return "Please select a patient to view their symptoms."
        
        # Questions about tests
        elif any(keyword in user_message for keyword in ["test", "tests", "labs", "laboratory", "diagnostic", "workup", "results"]):
            if patient_data:
                lab_results = patient_data.get('lab_results', {})
                if lab_results:
                    # Format the most significant abnormal labs first
                    abnormal_labs = []
                    for key, value in lab_results.items():
                        if key == 'lymphocyte_count' and value < 1.0:
                            abnormal_labs.append(f"{key.replace('_', ' ')} is low at {value}")
                        elif key == 'c_reactive_protein' and value > 10:
                            abnormal_labs.append(f"{key.replace('_', ' ')} is elevated at {value}")
                        elif key == 'd_dimer' and value > 0.5:
                            abnormal_labs.append(f"{key.replace('_', ' ')} is elevated at {value}")
                    
                    if abnormal_labs:
                        return f"Lab results for the patient show several abnormalities: {'; '.join(abnormal_labs)}. These findings are consistent with inflammatory response seen in COVID-19."
                    else:
                        results_text = "; ".join([f"{key.replace('_', ' ')}: {value}" for key, value in list(lab_results.items())[:3]])
                        return f"Lab results for the patient: {results_text}. No significant abnormalities noted in key inflammatory markers."
                else:
                    return "No laboratory tests have been recorded for this patient yet."
            else:
                return "Please select a patient to view their laboratory tests."
        
        # Questions about specific lab tests
        elif any(lab in user_message for lab in ["lymphocyte", "lymphocytes", "wbc", "crp", "d-dimer", "ferritin"]):
            if patient_data and 'lab_results' in patient_data:
                lab_results = patient_data.get('lab_results', {})
                if "lymphocyte" in user_message:
                    value = lab_results.get('lymphocyte_count', 'unknown')
                    if value != 'unknown' and value < 1.0:
                        return f"Lymphocyte count is {value}, which is below normal range (1.0-4.8 K/uL). Lymphopenia is common in COVID-19 and correlates with disease severity."
                    else:
                        return f"Lymphocyte count is {value} (normal range: 1.0-4.8 K/uL)."
                elif "crp" in user_message or "c-reactive protein" in user_message:
                    value = lab_results.get('c_reactive_protein', 'unknown')
                    if value != 'unknown' and value > 10:
                        return f"C-reactive protein is {value} mg/L, which is elevated (normal: <10 mg/L). CRP is an inflammatory marker often elevated in COVID-19 and other infections."
                    else:
                        return f"C-reactive protein is {value} mg/L (normal: <10 mg/L)."
                elif "d-dimer" in user_message:
                    value = lab_results.get('d_dimer', 'unknown')
                    if value != 'unknown' and value > 0.5:
                        return f"D-dimer is {value} ug/mL, which is elevated (normal: <0.5 ug/mL). Elevated D-dimer can indicate increased clotting risk, which is associated with COVID-19."
                    else:
                        return f"D-dimer is {value} ug/mL (normal: <0.5 ug/mL)."
                else:
                    # Generic response about labs
                    lab_results_formatted = []
                    for k, v in list(lab_results.items())[:5]:
                        lab_results_formatted.append(f"{k.replace('_', ' ')}: {v}")
                    return f"The patient's lab results include: {', '.join(lab_results_formatted)}."
            else:
                return "Lab results are not available for this patient or no patient is selected."
        
        # General questions about prognosis or next steps
        elif any(keyword in user_message for keyword in ["prognosis", "recommend", "treatment", "next steps", "plan", "management"]):
            if patient_data:
                probability = patient_data.get('covid_probability', 0.5)
                if probability > 0.7:
                    return "Given the high likelihood of COVID-19, I recommend PCR testing, pulse oximetry monitoring, and considering antiviral treatment if confirmed. Isolation precautions should be implemented immediately."
                elif probability > 0.4:
                    return "The patient shows moderate risk for COVID-19. Recommend PCR testing and home isolation pending results. Monitor symptoms for progression, especially oxygen saturation and respiratory status."
                else:
                    return "COVID-19 is less likely but cannot be ruled out. Consider testing for other respiratory pathogens and COVID-19 PCR for completeness. Symptomatic management is recommended."
            else:
                return "I need patient information to provide treatment recommendations. Please select a patient."
        
        # Questions about epidemiology or patterns
        elif any(keyword in user_message for keyword in ["pattern", "outbreak", "epidemiology", "spread", "cluster"]):
            return "Based on current data, we're seeing clusters of cases with similar symptom profiles. The primary transmission pattern appears to be respiratory droplets with 3-5 day average incubation. Cases with loss of smell/taste have 72% COVID-19 positivity rate in our database."
        
        # Questions about imaging
        elif any(keyword in user_message for keyword in ["xray", "x-ray", "ct", "imaging", "radiographic", "chest"]):
            if patient_data:
                if patient_data.get('covid_probability', 0) > 0.7:
                    return "Chest imaging typically shows bilateral peripheral ground-glass opacities in COVID-19. For this patient with high probability, a chest X-ray or CT would likely show these characteristic findings. CT has higher sensitivity for early or mild disease."
                else:
                    return "For this patient, chest imaging would be valuable to assess for the bilateral peripheral ground-glass opacities characteristic of COVID-19. Normal imaging doesn't exclude COVID-19, especially in early disease."
            else:
                return "Chest imaging in COVID-19 typically shows bilateral peripheral ground-glass opacities. CT has higher sensitivity than X-ray, especially in early disease. Please select a patient for specific recommendations."
        
        # Questions about risk factors
        elif any(keyword in user_message for keyword in ["risk factor", "comorbid", "underlying", "predispos"]):
            if patient_data:
                age = patient_data.get('age', 'unknown')
                gender = patient_data.get('gender', 'unknown')
                risk_factors = []
                
                if age > 65:
                    risk_factors.append("advanced age")
                
                # Extract potential comorbidities from clinical notes if available
                if 'clinical_notes' in patient_data:
                    notes = patient_data['clinical_notes'].lower()
                    for condition in ["diabetes", "hypertension", "obesity", "asthma", "copd", "heart disease", "immunocompromised"]:
                        if condition in notes:
                            risk_factors.append(condition)
                
                if risk_factors:
                    return f"This {age}-year-old {gender} patient has {len(risk_factors)} identified risk factors for severe COVID-19: {', '.join(risk_factors)}. These increase the risk of hospitalization and complications."
                else:
                    return f"This {age}-year-old {gender} patient doesn't have obvious risk factors for severe COVID-19 in the available information. Continue to monitor closely regardless."
            else:
                return "Key risk factors for severe COVID-19 include advanced age, diabetes, hypertension, obesity, cardiovascular disease, and immunocompromised status. Please select a patient for personalized risk assessment."
        
        # General diagnostic reasoning
        elif any(keyword in user_message for keyword in ["why", "reason", "explain", "elaborate", "how do you know"]):
            if patient_data:
                probability = patient_data.get('covid_probability', 0.5)
                return f"My reasoning is based on symptom patterns, lab values, and epidemiological factors. This patient has a {probability*100:.1f}% COVID probability primarily due to the combination of specific symptoms like {', '.join(patient_data.get('symptoms', ['Unknown'])[:3])} and laboratory findings such as {list(patient_data.get('lab_results', {}).keys())[:2]}. These factors together create a clinical picture consistent with viral pneumonia."
            else:
                return "I can explain my diagnostic reasoning for a specific patient if you select one."
        
        # Default response
        else:
            if patient_data:
                return f"I'm analyzing data for {patient_data.get('name', 'the patient')}. What specific aspect would you like to explore? I can discuss symptoms, lab results, COVID-19 probability, treatment recommendations, or risk factors."
            else:
                return "I'm Harvey, a medical assistant for COVID-19 analysis. I can help analyze patient data, assess COVID-19 probability, and provide clinical decision support. Please select a patient to begin."
                
    except Exception as e:
        print(f"Error in response generation: {e}")
        
        # Fallback response if something goes wrong
        if patient_data:
            return f"I've reviewed the information for {patient_data.get('name', 'this patient')}. What would you like to know specifically? I can discuss their symptoms, test results, or treatment options."
        else:
            return "I'm here to help with medical analysis. Please select a patient, and I can provide information about their condition and treatment options."