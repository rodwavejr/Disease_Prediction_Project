#!/bin/bash

# Activate virtual environment
source covid_venv/bin/activate

# Echo which Python is being used
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Virtual environment: $VIRTUAL_ENV"

# List installed packages
echo -e "\nInstalled packages:"
pip list | grep -E 'spacy|pandas|numpy|transformers|torch'

# Create directories for data
mkdir -p data/raw data/processed data/external output

# Run NER test
echo -e "\nRunning NER test:"
python test_ner.py

# Run data flow demo
echo -e "\nRunning data flow demo:"
python data_flow_demo.py

# Print instructions for continuing
echo -e "\n\nTo continue working with this environment:"
echo "1. Activate: source covid_venv/bin/activate"
echo "2. Download data: python download_cdc_data.py"
echo "3. Explore data: jupyter notebook notebooks/05_data_exploration.ipynb"
echo "4. Deactivate when done: deactivate"