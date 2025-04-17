"""
Functions for evaluating model performance and generating visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve,
    confusion_matrix, 
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def evaluate_binary_classifier(model, X, y_true, threshold=0.5):
    """
    Evaluate a binary classification model with various metrics.
    
    Parameters:
    -----------
    model : object
        Trained model with predict and predict_proba methods
    X : array-like
        Features to use for prediction
    y_true : array-like
        True labels
    threshold : float
        Decision threshold for binary classification
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred)
    }
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    # Additional data for plotting
    metrics['y_pred_proba'] = y_pred_proba
    metrics['precision_curve'] = precision
    metrics['recall_curve'] = recall
    
    return metrics

def plot_roc_curve_comparison(models_data, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models on the same graph.
    
    Parameters:
    -----------
    models_data : dict
        Dictionary with model names as keys and tuples of (y_true, y_pred_proba) as values
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    for model_name, (y_true, y_pred_proba) in models_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Reference line for random classifier
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, model_name='Model', figsize=(10, 8)):
    """
    Plot precision-recall curve for a model.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model for the plot title
    figsize : tuple
        Figure size (width, height)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.grid(True)
    plt.show()

def plot_calibration_curve(y_true, y_pred_proba, model_name='Model', n_bins=10, figsize=(10, 8)):
    """
    Plot calibration curve to check probability calibration.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model for the plot title
    n_bins : int
        Number of bins for the calibration curve
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot perfectly calibrated
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Plot calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=model_name)
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix_with_percentages(y_true, y_pred, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix with both counts and percentages.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list
        List of class names
    figsize : tuple
        Figure size (width, height)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Counts')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Percentages (%)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

def threshold_optimization(y_true, y_pred_proba, metric='f1', thresholds=None):
    """
    Find the optimal decision threshold for a specified metric.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    metric : str
        Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
    thresholds : array-like
        List of thresholds to try. If None, creates a range from 0.01 to 0.99.
        
    Returns:
    --------
    dict
        Dictionary with optimal threshold and corresponding metric values
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Metric '{metric}' not recognized")
        
        results.append({'threshold': threshold, metric: score})
    
    results_df = pd.DataFrame(results)
    optimal_threshold = results_df.loc[results_df[metric].idxmax(), 'threshold']
    
    # Create complete metrics at optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    metrics = {
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy_score(y_true, y_pred_optimal),
        'precision': precision_score(y_true, y_pred_optimal),
        'recall': recall_score(y_true, y_pred_optimal),
        'f1': f1_score(y_true, y_pred_optimal),
        'results_df': results_df
    }
    
    return metrics

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for model evaluation and visualization.")