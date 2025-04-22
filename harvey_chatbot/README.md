# Harvey - Medical AI Chatbot

Harvey is a minimalist medical chatbot interface for doctors and healthcare professionals to interact with patient data during disease outbreaks like COVID-19. Named after William Harvey, who discovered the circulatory system, this application helps medical professionals explore clinical data through natural language conversation.

## Features

- Clean, minimalist interface inspired by Claude
- Patient database with detailed clinical information
- Natural language interaction with patient records
- AI-powered analysis of clinical notes and lab results
- COVID-19 probability assessment based on symptoms and labs
- Real-time chat experience with medical context awareness

## Getting Started

### Prerequisites

- Python 3.8+
- Flask
- Additional requirements are listed in `requirements.txt`

### Installation

1. Clone the repository or navigate to the project directory
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000`

## Usage

1. Select a patient from the sidebar to load their clinical data
2. View patient details including symptoms, lab results, and COVID-19 probability
3. Ask questions about the patient in natural language
4. Get AI-assisted insights about diagnosis, lab interpretations, and treatment recommendations
5. Click "View Details" to see the complete patient record including clinical notes

## Example Queries

- "What are this patient's symptoms?"
- "What is the likelihood this patient has COVID-19?"
- "Why do you think this patient's probability is so high?"
- "What do the lab results indicate?"
- "What treatment would you recommend?"
- "What tests should be ordered next?"
- "Explain the significance of the d-dimer value."

## Technical Details

The application uses:
- Flask for the web server
- Custom NER (Named Entity Recognition) for analyzing clinical notes
- AI-powered medical reasoning for answering clinical queries
- Feature extraction from unstructured clinical text

## License

This project is for educational purposes only. Do not use for actual clinical decision-making.

## Acknowledgments

- Named after William Harvey (1578-1657), who discovered how blood circulates in the body
- Designed for medical professionals exploring COVID-19 patient data