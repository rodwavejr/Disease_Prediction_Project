from flask import Flask, render_template, request, jsonify
import json
import random
import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path to access project modules
sys.path.append(os.path.abspath('..'))

# Try to import NER or use a simplified version if not available
try:
    from src.ner_extraction import extract_entities_from_text
except ImportError:
    # Simplified entity extraction function if the main one isn't available
    def extract_entities_from_text(text, method="rule"):
        import re
        entities = {
            "SYMPTOM": [],
            "TIME": [],
            "SEVERITY": []
        }
        
        # Simple regex patterns for symptoms
        symptom_patterns = [
            r"fever", r"cough", r"shortness of breath", r"fatigue", 
            r"loss of taste", r"loss of smell", r"headache", r"sore throat"
        ]
        
        # Simple regex patterns for time expressions
        time_patterns = [
            r"\d+ days?", r"\d+ weeks?", r"since yesterday", 
            r"for the past \d+"
        ]
        
        # Simple regex patterns for severity
        severity_patterns = [
            r"mild", r"moderate", r"severe", r"significant", 
            r"pronounced", r"marked"
        ]
        
        # Find symptoms
        for pattern in symptom_patterns:
            for match in re.finditer(pattern, text.lower()):
                entities["SYMPTOM"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Find time expressions
        for pattern in time_patterns:
            for match in re.finditer(pattern, text.lower()):
                entities["TIME"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Find severity indicators
        for pattern in severity_patterns:
            for match in re.finditer(pattern, text.lower()):
                entities["SEVERITY"].append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        print(f"Extracted {len(entities['SYMPTOM'])} symptoms, {len(entities['TIME'])} time expressions, and {len(entities['SEVERITY'])} severity indicators")
        return entities

from utils.patient_analysis import analyze_patient_text, calculate_covid_probability

app = Flask(__name__)

# Load sample patient data
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_patients', methods=['GET'])
def get_patients():
    try:
        # Load patient data from the sample file
        with open('data/sample_patients.json', 'r') as f:
            patients = json.load(f)
        return jsonify({"success": True, "patients": patients})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/get_patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    try:
        # Load patient data from the sample file
        with open('data/sample_patients.json', 'r') as f:
            patients = json.load(f)
            
        # Find the specific patient
        patient = next((p for p in patients if p["id"] == patient_id), None)
        
        if patient:
            # Load the patient's clinical notes
            with open(f'data/clinical_notes/{patient_id}.txt', 'r') as f:
                clinical_notes = f.read()
            patient['clinical_notes'] = clinical_notes
            
            # Calculate COVID probability based on symptoms
            entities = extract_entities_from_text(clinical_notes, method="rule")
            covid_probability = calculate_covid_probability(entities, patient['lab_results'])
            patient['covid_probability'] = covid_probability
            
            return jsonify({"success": True, "patient": patient})
        else:
            return jsonify({"success": False, "error": "Patient not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        patient_id = data.get('patient_id', None)
        
        # Try to analyze the user's message and generate a response
        try:
            response = analyze_patient_text(user_message, patient_id)
        except Exception as analysis_error:
            print(f"Error in patient text analysis: {analysis_error}")
            # Fallback responses based on keywords in the user's message
            response = get_fallback_response(user_message, patient_id)
        
        return jsonify({
            "success": True, 
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        # Final fallback if everything else fails
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            "success": True,
            "response": "I'm having trouble processing that request right now. Could you try rephrasing or asking about a different aspect of the patient's case?",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# Fallback response function
def get_fallback_response(user_message, patient_id):
    """Provide a fallback response based on keywords in the user's message"""
    user_message = user_message.lower()
    
    # Load patient data if possible
    patient_data = None
    try:
        with open('data/sample_patients.json', 'r') as f:
            patients = json.load(f)
            if patient_id:
                patient_data = next((p for p in patients if p["id"] == patient_id), None)
    except Exception as e:
        print(f"Error loading patient data for fallback: {e}")
    
    # Fallback responses based on keywords
    if "symptom" in user_message or "presenting" in user_message:
        if patient_data and "symptoms" in patient_data:
            symptoms = ", ".join(patient_data["symptoms"])
            return f"The patient is presenting with {symptoms}."
        return "This patient has reported multiple symptoms including respiratory issues. What specific aspect would you like to know about?"
    
    elif "covid" in user_message and ("probability" in user_message or "likelihood" in user_message or "chance" in user_message):
        if patient_data and "covid_probability" in patient_data:
            prob = patient_data["covid_probability"] * 100
            return f"Based on clinical presentation, this patient has approximately {prob:.1f}% likelihood of COVID-19."
        return "I estimate this patient has a moderate to high probability of COVID-19 based on symptom patterns."
    
    elif any(term in user_message for term in ["lab", "test", "result"]):
        if patient_data and "lab_results" in patient_data:
            labs = [f"{k.replace('_', ' ')}: {v}" for k, v in list(patient_data["lab_results"].items())[:3]]
            return f"Key lab results include {', '.join(labs)}. Would you like more specific details about certain values?"
        return "The patient's laboratory workup includes several findings consistent with viral infection. Any specific lab values you're interested in?"
    
    elif any(term in user_message for term in ["treat", "recommend", "plan", "therapy"]):
        return "For this type of presentation, I would recommend supportive care, monitoring of vital signs, and testing to confirm diagnosis. Would you like more specific treatment recommendations?"
    
    elif any(term in user_message for term in ["prognosis", "outlook", "outcome"]):
        return "Prognosis depends on several factors including age, comorbidities, and symptom severity. With appropriate care and monitoring, most patients with this presentation have favorable outcomes."
    
    elif any(term in user_message for term in ["risk", "factor", "comorbid"]):
        return "Key risk factors to consider include age, immunocompromised status, cardiopulmonary disease, and metabolic disorders. Would you like to know about specific risk factors for this patient?"
    
    # Default fallback
    return "I understand you're asking about this patient. Could you please specify what aspect of the case you'd like to explore? I can discuss symptoms, lab results, treatment options, or diagnostic possibilities."

if __name__ == '__main__':
    app.run(debug=True)