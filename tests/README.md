# Test Scripts for COVID-19 Detection Project

This directory contains test scripts for validating various components of the COVID-19 detection pipeline.

## Available Tests

### NER Tests

- **test_ner.py**: Tests the Named Entity Recognition functionality
  - Validates entity extraction from clinical notes
  - Tests rule-based, spaCy, and transformer approaches
  - Confirms proper formatting for downstream use

- **test_mimic_ner.py**: Tests NER specifically with MIMIC clinical notes
  - Works with MIMIC-IV clinical text data
  - Validates extraction quality on medical terminology
  - Generates structured features from MIMIC text

- **test_ner_integration.py**: Tests the integration of NER with classification
  - Verifies NER features are properly converted for classification
  - Validates joining logic between text-derived and structured features
  - Confirms data flow through the complete pipeline

### Data Tests

- **preview_classification_data.py**: Examines classification dataset structure
  - Displays feature categories and distributions
  - Checks for proper NER feature integration
  - Validates target variable distribution

- **analyze_classification_dataset.py**: Performs in-depth dataset analysis
  - Generates visualizations of data distributions
  - Evaluates feature importance and correlations
  - Produces summary reports for the dataset

## Running Tests

To run a test, use:
```bash
# Simple test
python tests/test_ner.py

# With the project's virtual environment
./run_venv.sh python tests/test_ner.py
```

## Test Output

Test results are saved to:
- `output/ner_test_results.json` - NER test output
- `output/mimic_ner_results.json` - MIMIC NER test output 
- `results/analysis/` - Analysis results and visualizations

## Adding New Tests

When adding new tests:
1. Follow the naming convention: `test_<component>.py`
2. Include proper error handling and logging
3. Save results to the appropriate output directory 
4. Update this README with the new test description