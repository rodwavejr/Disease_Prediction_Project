"""
Data fetcher for COVID-19 datasets.
This module handles downloading and processing of real COVID-19 datasets.
"""

import os
import json
import datetime
import urllib.request
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DIR, EXTERNAL_DIR]:
    os.makedirs(directory, exist_ok=True)

def fetch_cord19_metadata():
    """
    Fetch CORD-19 metadata from the Allen AI repository.
    The full dataset is very large (10GB+), so we only fetch the metadata.csv file.
    
    Returns:
    --------
    str
        Path to downloaded metadata file
    """
    # CORD-19 metadata URL (public access)
    url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2021-05-24.tar.gz"
    metadata_url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/latest/metadata.csv"
    
    output_path = os.path.join(EXTERNAL_DIR, "cord19_metadata.csv")
    
    try:
        logger.info(f"Fetching CORD-19 metadata from {metadata_url}")
        urllib.request.urlretrieve(metadata_url, output_path)
        logger.info(f"Downloaded CORD-19 metadata to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error downloading CORD-19 metadata: {e}")
        return None

def fetch_covid19_cases():
    """
    Fetch COVID-19 case surveillance public use data from CDC.
    Returns path to downloaded file.
    
    Returns:
    --------
    str
        Path to downloaded surveillance data
    """
    # CDC case surveillance data URL
    url = "https://data.cdc.gov/api/views/vbim-akqf/rows.csv?accessType=DOWNLOAD"
    output_path = os.path.join(EXTERNAL_DIR, "covid19_case_surveillance.csv")
    
    try:
        logger.info(f"Fetching COVID-19 case surveillance data from CDC")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded COVID-19 case surveillance data to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error downloading COVID-19 case surveillance data: {e}")
        return None

def fetch_covid_clinical_trials():
    """
    Fetch COVID-19 clinical trials data from clinicaltrials.gov.
    This provides rich textual descriptions of COVID-19 symptoms and treatments.
    
    Returns:
    --------
    str
        Path to downloaded clinical trials data
    """
    # Clinical trials API for COVID-19 studies
    url = "https://clinicaltrials.gov/api/query/study_fields?expr=COVID-19&fields=BriefTitle,BriefSummary,DetailedDescription,Condition,ConditionBrowseLeafName&fmt=json&max_rnk=1000"
    output_path = os.path.join(EXTERNAL_DIR, "covid19_clinical_trials.json")
    
    try:
        logger.info(f"Fetching COVID-19 clinical trials data")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded COVID-19 clinical trials data to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error downloading COVID-19 clinical trials data: {e}")
        return None

def fetch_covid_twitter_dataset():
    """
    Fetch COVID-19 Twitter dataset from GitHub.
    This provides public discussions about COVID-19 symptoms and experiences.
    
    Returns:
    --------
    str
        Path to downloaded Twitter data
    """
    # Public dataset with COVID-19 tweets
    url = "https://github.com/thepanacealab/covid19_twitter/raw/master/dailies/2020-03-22/2020-03-22_clean-dataset.tsv.gz"
    output_path = os.path.join(EXTERNAL_DIR, "covid19_tweets.tsv.gz")
    extracted_path = os.path.join(EXTERNAL_DIR, "covid19_tweets.tsv")
    
    try:
        logger.info(f"Fetching COVID-19 Twitter dataset")
        urllib.request.urlretrieve(url, output_path)
        
        # Extract the gzipped file
        import gzip
        import shutil
        with gzip.open(output_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Downloaded and extracted COVID-19 Twitter data to {extracted_path}")
        return extracted_path
    except Exception as e:
        logger.error(f"Error downloading COVID-19 Twitter data: {e}")
        return None

def fetch_mimic_demo():
    """
    Fetch MIMIC-III demo dataset if available.
    The full MIMIC dataset requires credentialed access, but a demo version is available.
    
    Returns:
    --------
    str
        Path to downloaded MIMIC data
    """
    # MIMIC-III demo data (subset of real clinical data)
    url = "https://physionet.org/files/mimiciii-demo/1.4/NOTEEVENTS.csv.gz"
    output_path = os.path.join(EXTERNAL_DIR, "mimic_notes.csv.gz")
    extracted_path = os.path.join(EXTERNAL_DIR, "mimic_notes.csv")
    
    try:
        logger.info(f"Fetching MIMIC-III demo clinical notes")
        urllib.request.urlretrieve(url, output_path)
        
        # Extract the gzipped file
        import gzip
        import shutil
        with gzip.open(output_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Downloaded and extracted MIMIC-III notes to {extracted_path}")
        return extracted_path
    except Exception as e:
        logger.error(f"Error downloading MIMIC-III demo data: {e}")
        return None

def fetch_i2b2_sample():
    """
    Fetch i2b2 sample data if available.
    The full i2b2 dataset requires authorization, but sample data may be available.
    
    Returns:
    --------
    str
        Path to downloaded i2b2 data
    """
    # i2b2 sample data URL (if public access is available)
    url = "https://www.i2b2.org/NLP/DataSets/Main.php"
    output_path = os.path.join(EXTERNAL_DIR, "i2b2_sample.txt")
    
    try:
        logger.info(f"Fetching i2b2 sample data")
        # This would require authentication, so we'll just log it
        logger.info(f"Note: i2b2 datasets require credentialed access")
        logger.info(f"Please visit {url} to request access to the data")
        return None
    except Exception as e:
        logger.error(f"Error accessing i2b2 data: {e}")
        return None

def list_available_datasets():
    """
    List all available COVID-19 related datasets.
    
    Returns:
    --------
    list
        List of dataset descriptions
    """
    datasets = [
        {
            "name": "CORD-19 Research Papers",
            "description": "COVID-19 Open Research Dataset of scientific papers",
            "url": "https://www.semanticscholar.org/cord19",
            "data_type": "Unstructured text (research papers)",
            "size": "~10GB full dataset, ~200MB metadata only",
            "access": "Public"
        },
        {
            "name": "CDC COVID-19 Case Surveillance",
            "description": "De-identified patient-level data on COVID-19 cases",
            "url": "https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/vbim-akqf",
            "data_type": "Structured data (patient records)",
            "size": "~1GB",
            "access": "Public"
        },
        {
            "name": "COVID-19 Clinical Trials",
            "description": "Clinical trials related to COVID-19 with detailed descriptions",
            "url": "https://clinicaltrials.gov/ct2/results?cond=COVID-19",
            "data_type": "Semi-structured text (trial descriptions)",
            "size": "~50MB",
            "access": "Public"
        },
        {
            "name": "COVID-19 Twitter Dataset",
            "description": "Tweets related to COVID-19 symptoms and experiences",
            "url": "https://github.com/thepanacealab/covid19_twitter",
            "data_type": "Unstructured text (social media)",
            "size": "Variable (~100MB per day)",
            "access": "Public"
        },
        {
            "name": "MIMIC-III Clinical Database",
            "description": "Medical information for ICU patients (includes some COVID cases)",
            "url": "https://physionet.org/content/mimiciii/1.4/",
            "data_type": "Structured data + unstructured clinical notes",
            "size": "~40GB",
            "access": "Requires credential application"
        },
        {
            "name": "i2b2 NLP Research Datasets",
            "description": "Clinical NLP datasets with some COVID-19 related content",
            "url": "https://www.i2b2.org/NLP/DataSets/Main.php",
            "data_type": "Unstructured text (clinical notes)",
            "size": "Variable",
            "access": "Requires application"
        }
    ]
    
    return datasets

def print_dataset_info():
    """Print information about available datasets."""
    datasets = list_available_datasets()
    
    print("\n=== COVID-19 Datasets for NLP and Classification ===\n")
    
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset['name']}")
        print(f"   Description: {dataset['description']}")
        print(f"   Data Type: {dataset['data_type']}")
        print(f"   Size: {dataset['size']}")
        print(f"   Access: {dataset['access']}")
        print(f"   URL: {dataset['url']}")
        print()

def fetch_all_public_datasets():
    """
    Fetch all publicly available datasets.
    
    Returns:
    --------
    dict
        Dictionary with paths to all downloaded datasets
    """
    results = {}
    
    # Fetch CORD-19 metadata
    results['cord19'] = fetch_cord19_metadata()
    
    # Fetch CDC case surveillance data
    results['cdc'] = fetch_covid19_cases()
    
    # Fetch clinical trials data
    results['clinical_trials'] = fetch_covid_clinical_trials()
    
    # Fetch Twitter data
    results['twitter'] = fetch_covid_twitter_dataset()
    
    # Note about restricted datasets
    logger.info("Note: MIMIC-III and i2b2 datasets require credentialed access")
    logger.info("Please apply for access if needed for more clinical data")
    
    # Return successful downloads
    return {k: v for k, v in results.items() if v is not None}

if __name__ == "__main__":
    # Print available datasets
    print_dataset_info()
    
    # Fetch all publicly available datasets
    print("\nFetching public datasets...\n")
    downloaded = fetch_all_public_datasets()
    
    print("\nDownloaded datasets:")
    for name, path in downloaded.items():
        print(f"- {name}: {path}")