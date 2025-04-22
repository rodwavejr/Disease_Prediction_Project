#!/usr/bin/env python
"""
Script to prepare and analyze the COVID-19 classification dataset in sequence.

This script runs:
1. prepare_classification_dataset.py - To generate the dataset
2. analyze_classification_dataset.py - To analyze the prepared dataset
"""

import os
import sys
import subprocess
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path):
    """
    Run a Python script as a subprocess and log the result.
    
    Parameters:
    -----------
    script_path : str
        Path to the script to run
    
    Returns:
    --------
    bool
        True if the script ran successfully, False otherwise
    """
    script_name = os.path.basename(script_path)
    logger.info(f"Running {script_name}...")
    
    # Check for virtual environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(current_dir, "covid_venv", "bin", "python")
    
    if os.path.exists(venv_python):
        logger.info(f"Using virtual environment Python: {venv_python}")
        python_executable = venv_python
    else:
        logger.warning(f"Virtual environment not found, using system Python: {sys.executable}")
        python_executable = sys.executable
    
    try:
        # Run the script using the appropriate Python interpreter
        result = subprocess.run(
            [python_executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log stdout
        for line in result.stdout.strip().split('\n'):
            if line:
                logger.info(f"{script_name} output: {line}")
        
        logger.info(f"{script_name} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        
        # Log stdout
        for line in e.stdout.strip().split('\n'):
            if line:
                logger.info(f"{script_name} output: {line}")
        
        # Log stderr
        for line in e.stderr.strip().split('\n'):
            if line:
                logger.error(f"{script_name} error: {line}")
        
        return False

def main():
    """Main function to run the classification dataset preparation and analysis pipeline."""
    logger.info("Starting COVID-19 classification dataset preparation and analysis pipeline")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define script paths
    preparation_script = os.path.join(current_dir, "prepare_classification_dataset.py")
    analysis_script = os.path.join(current_dir, "analyze_classification_dataset.py")
    
    # Check if scripts exist
    if not os.path.exists(preparation_script):
        logger.error(f"Preparation script not found: {preparation_script}")
        sys.exit(1)
    
    if not os.path.exists(analysis_script):
        logger.error(f"Analysis script not found: {analysis_script}")
        sys.exit(1)
    
    # Step 1: Run the preparation script
    logger.info("Step 1: Preparing the classification dataset")
    if not run_script(preparation_script):
        logger.error("Dataset preparation failed. Stopping pipeline.")
        sys.exit(1)
    
    # Step 2: Run the analysis script
    logger.info("Step 2: Analyzing the classification dataset")
    if not run_script(analysis_script):
        logger.error("Dataset analysis failed.")
        sys.exit(1)
    
    logger.info("COVID-19 classification dataset preparation and analysis pipeline completed successfully")
    print("\nPipeline completed successfully!")
    print("Check the results directory for analysis results and plots.")

if __name__ == "__main__":
    main()