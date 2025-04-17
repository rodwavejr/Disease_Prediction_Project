"""
Download and process CDC COVID-19 Case Surveillance Public Use Data.
This script downloads real COVID-19 patient data from the CDC.
"""

import os
import sys
import urllib.request
import csv
import gzip
import shutil
import datetime

# Create data directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

for directory in [DATA_DIR, EXTERNAL_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

def download_cdc_data():
    """Download the CDC COVID-19 Case Surveillance Public Use Data."""
    # CDC case surveillance data URL
    url = "https://data.cdc.gov/api/views/vbim-akqf/rows.csv?accessType=DOWNLOAD"
    output_path = os.path.join(EXTERNAL_DIR, "covid19_case_surveillance.csv")
    
    print(f"Downloading CDC COVID-19 case surveillance data from {url}")
    print(f"This may take a few minutes as the file is large...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Downloaded CDC data to {output_path} ({file_size:.2f} MB)")
        return output_path
    except Exception as e:
        print(f"Error downloading CDC data: {e}")
        return None

def create_sample_dataset(full_path, sample_size=10000):
    """Create a smaller sample dataset for easier processing."""
    sample_path = os.path.join(PROCESSED_DIR, "covid19_sample.csv")
    
    print(f"Creating sample dataset with {sample_size} records...")
    
    try:
        with open(full_path, 'r') as infile, open(sample_path, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Write header
            header = next(reader)
            writer.writerow(header)
            
            # Write sample rows
            for i, row in enumerate(reader):
                if i < sample_size:
                    writer.writerow(row)
                else:
                    break
        
        print(f"Created sample dataset with {sample_size} records at {sample_path}")
        return sample_path
    except Exception as e:
        print(f"Error creating sample dataset: {e}")
        return None

def download_cord19_metadata():
    """Download the CORD-19 metadata file."""
    metadata_url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv"
    output_path = os.path.join(EXTERNAL_DIR, "cord19_metadata.csv")
    
    print(f"Downloading CORD-19 metadata from {metadata_url}")
    try:
        urllib.request.urlretrieve(metadata_url, output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Downloaded CORD-19 metadata to {output_path} ({file_size:.2f} MB)")
        return output_path
    except Exception as e:
        print(f"Error downloading CORD-19 metadata: {e}")
        return None

def download_covid_clinical_trials():
    """Download COVID-19 clinical trials data."""
    # Updated URL for COVID-19 clinical trials data
    url = "https://clinicaltrials.gov/api/query/full_studies?expr=COVID-19&fmt=json&max_rnk=100"
    output_path = os.path.join(EXTERNAL_DIR, "covid19_clinical_trials.json")
    
    print(f"Downloading COVID-19 clinical trials data from {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Downloaded clinical trials data to {output_path} ({file_size:.2f} MB)")
        return output_path
    except Exception as e:
        print(f"Error downloading clinical trials data: {e}")
        return None

if __name__ == "__main__":
    print("=== Downloading Real COVID-19 Datasets ===\n")
    
    # Download CDC data
    print("\n1. CDC COVID-19 Case Surveillance Data (Structured)")
    cdc_path = download_cdc_data()
    
    if cdc_path and os.path.exists(cdc_path):
        sample_path = create_sample_dataset(cdc_path)
    
    # Download CORD-19 metadata
    print("\n2. CORD-19 Research Papers Metadata (Unstructured Text)")
    cord19_path = download_cord19_metadata()
    
    # Download clinical trials data
    print("\n3. COVID-19 Clinical Trials Data (Unstructured Text)")
    trials_path = download_covid_clinical_trials()
    
    print("\n=== Download Summary ===")
    for dataset, path in [
        ("CDC COVID-19 Case Surveillance", cdc_path),
        ("CDC Sample Dataset", sample_path if 'sample_path' in locals() else None),
        ("CORD-19 Metadata", cord19_path),
        ("COVID-19 Clinical Trials", trials_path)
    ]:
        status = "✅ Downloaded" if path and os.path.exists(path) else "❌ Failed"
        print(f"{dataset}: {status}")
    
    print("\nNext step: Run notebooks/05_data_exploration.ipynb to explore the data")