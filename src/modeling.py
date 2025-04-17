"""
Functions for building, training, and evaluating prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def train_test_validation_split(X, y, test_size=0.2, validation_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    test_size : float
        Proportion of the dataset to include in the test split
    validation_size : float
        Proportion of the training data to include in the validation split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict and predict_proba methods
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        True labels
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probability predictions if available
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        has_prob = True
    except (AttributeError, IndexError):
        has_prob = False
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Add probability-based metrics if available
    if has_prob:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        metrics['pr_auc'] = auc(recall, precision)
    
    return metrics

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list, optional
        List of class names
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob, label='Model'):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    label : str
        Label for the ROC curve
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for model training and evaluation.")