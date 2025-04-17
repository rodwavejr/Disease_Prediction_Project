"""
Functions for data loading, cleaning, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        The loaded data
    """
    return pd.read_csv(file_path)

def identify_column_types(df):
    """
    Identify numeric and categorical columns in a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to analyze
        
    Returns:
    --------
    tuple
        (numeric_columns, categorical_columns)
    """
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numeric_columns, categorical_columns

def create_preprocessing_pipeline(numeric_columns, categorical_columns):
    """
    Create a preprocessing pipeline with scikit-learn.
    
    Parameters:
    -----------
    numeric_columns : list
        List of numeric column names
    categorical_columns : list
        List of categorical column names
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        A preprocessing pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])
    
    return preprocessor

def split_features_target(df, target_column):
    """
    Split dataframe into features and target.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to split
    target_column : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (X, y) where X is the features dataframe and y is the target series
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for data processing.")