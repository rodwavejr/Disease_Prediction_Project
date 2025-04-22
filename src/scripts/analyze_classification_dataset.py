#!/usr/bin/env python
"""
Script to analyze the prepared COVID-19 classification dataset before model building.

This script performs exploratory data analysis on the master dataset to identify:
- Class distribution and imbalance
- Feature statistics and distributions
- Relationships between features and target variable
- Data quality issues (missing values, outliers)
- Potential feature importance

The analysis is saved as both visual plots and a summary report.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback
import logging
import json
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

# Configure matplotlib and seaborn - using simple settings to avoid style conflicts
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)  # Reset to matplotlib defaults
plt.style.use('default')  # Use default style 
sns.set_style("whitegrid")  # Set seaborn style explicitly

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classification_dataset_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')

# Create output directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(os.path.join(ANALYSIS_DIR, 'plots'), exist_ok=True)

def load_dataset():
    """
    Load the prepared classification dataset.
    
    Returns:
    --------
    pandas.DataFrame or None
    """
    # Check multiple possible locations for the dataset
    dataset_paths = [
        os.path.join(PROCESSED_DIR, 'covid_classification_dataset.csv'),
        os.path.join(PROCESSED_DIR, 'mimic_features.csv'),
        os.path.join(DATA_DIR, 'external', 'mimic', 'classification_data.csv')
    ]
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            logger.info(f"Loading classification dataset from {dataset_path}")
            try:
                df = pd.read_csv(dataset_path)
                
                # If we have covid_diagnosis but not covid_positive, create it
                if 'covid_diagnosis' in df.columns and 'covid_positive' not in df.columns:
                    df['covid_positive'] = df['covid_diagnosis'].map({True: 1, 'True': 1, False: 0, 'False': 0})
                    logger.info(f"Converted covid_diagnosis to covid_positive column")
                
                # Create record_id if not present
                if 'record_id' not in df.columns and 'subject_id' in df.columns:
                    df['record_id'] = ['MIMIC_' + str(id) for id in df['subject_id']]
                    logger.info(f"Created record_id column from subject_id")
                
                logger.info(f"Loaded dataset with {len(df)} records and {len(df.columns)} features")
                return df
            except Exception as e:
                logger.error(f"Error loading dataset from {dataset_path}: {e}")
                logger.debug(traceback.format_exc())
    
    logger.warning(f"Classification dataset not found in any expected locations")
    return None

def analyze_target_distribution(df):
    """
    Analyze the distribution of the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    dict
        Distribution statistics
    """
    if 'covid_positive' not in df.columns:
        logger.warning("Target variable 'covid_positive' not found in dataset")
        return None
    
    logger.info("Analyzing target distribution")
    
    # Calculate distribution
    target_counts = df['covid_positive'].value_counts().sort_index()
    target_pct = (target_counts / len(df) * 100).round(2)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='covid_positive', data=df, palette=['#3498db', '#e74c3c'])
    plt.title('Distribution of COVID-19 Cases', fontsize=14)
    plt.xlabel('COVID-19 Positive Status (0=No, 1=Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels
    for i, count in enumerate(target_counts):
        plt.text(i, count + 100, f"{count}\n({target_pct[i]}%)", 
                 ha='center', va='bottom', fontsize=12)
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'target_distribution.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Create statistics
    stats = {
        'negative_count': int(target_counts.get(0, 0)),
        'positive_count': int(target_counts.get(1, 0)),
        'negative_pct': float(target_pct.get(0, 0)),
        'positive_pct': float(target_pct.get(1, 0)),
        'class_ratio': float(target_counts.get(0, 0) / max(1, target_counts.get(1, 0)))
    }
    
    logger.info(f"Target distribution: {stats['positive_count']} positive ({stats['positive_pct']}%), " +
                f"{stats['negative_count']} negative ({stats['negative_pct']}%)")
    
    if stats['class_ratio'] > 3:
        logger.warning(f"Significant class imbalance detected: {stats['class_ratio']:.1f}:1 ratio")
    
    return stats

def analyze_missing_values(df):
    """
    Analyze missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    dict
        Missing value statistics
    """
    logger.info("Analyzing missing values")
    
    # Calculate missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    # Create sorted dataframe of missing values
    missing_df = pd.DataFrame({
        'column': missing.index,
        'count': missing.values,
        'percent': missing_pct.values
    })
    missing_df = missing_df.sort_values('count', ascending=False)
    
    # Only keep columns with missing values
    missing_df = missing_df[missing_df['count'] > 0]
    
    # Create plot for top 20 columns with missing values
    if not missing_df.empty:
        plt.figure(figsize=(12, 8))
        top_missing = missing_df.head(20)
        
        sns.barplot(x='percent', y='column', data=top_missing, palette='viridis')
        plt.title('Percentage of Missing Values by Feature', fontsize=14)
        plt.xlabel('Percentage Missing', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        # Save plot
        plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'missing_values.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
    
    # Create statistics
    stats = {
        'total_missing_cells': int(missing.sum()),
        'total_cells': int(df.size),
        'overall_missing_pct': float((missing.sum() / df.size * 100).round(2)),
        'columns_with_missing': int(sum(missing > 0)),
        'columns_complete': int(sum(missing == 0)),
        'top_missing_columns': missing_df.head(10).to_dict('records')
    }
    
    logger.info(f"Missing values: {stats['overall_missing_pct']}% of all cells, " +
                f"{stats['columns_with_missing']} columns have missing values")
    
    return stats

def analyze_feature_distributions(df):
    """
    Analyze distributions of numeric and categorical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    dict
        Feature distribution statistics
    """
    logger.info("Analyzing feature distributions")
    
    # Separate numeric and categorical features
    # Exclude the record_id and target column
    features = [col for col in df.columns if col not in ['record_id', 'covid_positive']]
    numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Add binary features (0/1) to categorical
    for col in numeric_features.copy():
        if set(df[col].dropna().unique()).issubset({0, 1}):
            categorical_features.append(col)
            numeric_features.remove(col)
    
    # Analyze numeric features
    numeric_stats = {}
    if numeric_features:
        numeric_df = df[numeric_features].describe().transpose()
        numeric_df['missing'] = df[numeric_features].isnull().sum()
        numeric_df['missing_pct'] = (df[numeric_features].isnull().sum() / len(df) * 100).round(2)
        
        # Plot histograms for top 16 numeric features (or all if less than 16)
        n_plots = min(16, len(numeric_features))
        if n_plots > 0:
            n_cols = min(4, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            plt.figure(figsize=(16, n_rows * 4))
            for i, col in enumerate(numeric_features[:n_plots]):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f'Distribution of {col}')
                plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'numeric_distributions.png')
            plt.savefig(plot_path)
            plt.close()
        
        numeric_stats = {
            'count': len(numeric_features),
            'features': numeric_df.to_dict('index')
        }
    
    # Analyze categorical features
    categorical_stats = {}
    if categorical_features:
        # Calculate statistics for each categorical feature
        cat_stats = {}
        for col in categorical_features:
            value_counts = df[col].value_counts().sort_values(ascending=False)
            cat_stats[col] = {
                'unique_values': int(df[col].nunique()),
                'missing': int(df[col].isnull().sum()),
                'missing_pct': float((df[col].isnull().sum() / len(df) * 100).round(2)),
                'top_values': value_counts.head(5).to_dict()
            }
        
        # Plot bar charts for top categorical features (by number of unique values)
        top_cat_features = sorted(
            [(col, df[col].nunique()) for col in categorical_features], 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        if top_cat_features:
            n_plots = min(10, len(top_cat_features))
            n_cols = min(2, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            plt.figure(figsize=(16, n_rows * 4))
            for i, (col, _) in enumerate(top_cat_features[:n_plots]):
                plt.subplot(n_rows, n_cols, i + 1)
                top_n = min(10, df[col].nunique())
                
                # Get top values sorted by frequency
                value_counts = df[col].value_counts().head(top_n)
                sns.barplot(x=value_counts.index, y=value_counts.values)
                
                plt.title(f'Top Values for {col}')
                plt.xticks(rotation=90)
                plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'categorical_distributions.png')
            plt.savefig(plot_path)
            plt.close()
        
        categorical_stats = {
            'count': len(categorical_features),
            'features': cat_stats
        }
    
    # Return all statistics
    stats = {
        'numeric_features': numeric_stats,
        'categorical_features': categorical_stats
    }
    
    logger.info(f"Analyzed {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
    
    return stats

def analyze_feature_importance(df):
    """
    Analyze feature importance using mutual information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    dict
        Feature importance statistics
    """
    if 'covid_positive' not in df.columns:
        logger.warning("Target variable 'covid_positive' not found in dataset")
        return None
    
    logger.info("Analyzing feature importance")
    
    # Exclude record_id from analysis
    features = [col for col in df.columns if col not in ['record_id', 'covid_positive']]
    
    if not features:
        logger.warning("No features available for importance analysis")
        return None
    
    # Prepare data - impute missing values to use mutual information
    X = df[features].copy()
    y = df['covid_positive'].copy()
    
    # Simple imputation for analysis purposes
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Calculate mutual information
    try:
        mi_scores = mutual_info_classif(X_imputed, y, random_state=42)
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title('Feature Importance (Mutual Information)', fontsize=14)
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        # Save plot
        plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'feature_importance.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        # Create statistics
        stats = {
            'top_features': feature_importance.head(20).to_dict('records'),
            'low_importance_features': feature_importance.tail(20).to_dict('records'),
            'mean_importance': float(feature_importance['importance'].mean()),
            'median_importance': float(feature_importance['importance'].median())
        }
        
        logger.info(f"Feature importance analysis complete. Top feature: {top_features.iloc[0]['feature']}")
        
        return stats
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}")
        logger.debug(traceback.format_exc())
        return None

def analyze_feature_correlations(df):
    """
    Analyze correlations between numeric features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    dict
        Correlation statistics
    """
    logger.info("Analyzing feature correlations")
    
    # Get numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Ensure we have enough numeric features
    if len(numeric_features) < 2:
        logger.warning("Not enough numeric features for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix', fontsize=14)
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'correlation_matrix.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Find highest correlations (excluding self-correlations)
    correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only use lower triangle
                correlations.append({
                    'feature1': col1,
                    'feature2': col2,
                    'correlation': float(corr_matrix.loc[col1, col2])
                })
    
    # Sort by absolute correlation value
    correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    # Get highest positive and negative correlations
    high_positive = [c for c in correlations if c['correlation'] > 0.5][:10]
    high_negative = [c for c in correlations if c['correlation'] < -0.5][:10]
    
    # Create statistics
    stats = {
        'highest_positive_correlations': high_positive,
        'highest_negative_correlations': high_negative,
        'correlation_with_target': []
    }
    
    # Check correlations with target
    if 'covid_positive' in numeric_features:
        target_corrs = []
        for col in numeric_features:
            if col != 'covid_positive':
                target_corrs.append({
                    'feature': col,
                    'correlation': float(corr_matrix.loc[col, 'covid_positive'])
                })
        
        # Sort by absolute correlation
        target_corrs = sorted(target_corrs, key=lambda x: abs(x['correlation']), reverse=True)
        stats['correlation_with_target'] = target_corrs[:20]
    
    logger.info("Feature correlation analysis complete")
    
    return stats

def analyze_data_source_comparison(df):
    """
    Compare records from different data sources.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    dict
        Comparison statistics
    """
    if 'record_id' not in df.columns:
        logger.warning("Record ID not found in dataset")
        return None
    
    logger.info("Analyzing data source comparison")
    
    # Determine data sources
    source_map = df['record_id'].apply(lambda x: x.split('_')[0] if '_' in str(x) else 'Unknown')
    source_counts = source_map.value_counts()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x=source_map, palette='Set3')
    plt.title('Records by Data Source', fontsize=14)
    plt.xlabel('Data Source', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels
    for i, count in enumerate(source_counts):
        plt.text(i, count + 10, count, ha='center', va='bottom', fontsize=12)
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'data_source_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Compare COVID-19 positive rates by source
    source_df = pd.DataFrame({
        'source': source_map,
        'covid_positive': df['covid_positive']
    })
    
    positive_rates = source_df.groupby('source')['covid_positive'].mean().round(4) * 100
    
    # Create positive rate plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=positive_rates.index, y=positive_rates.values, palette='Set1')
    plt.title('COVID-19 Positive Rate by Data Source', fontsize=14)
    plt.xlabel('Data Source', fontsize=12)
    plt.ylabel('Positive Rate (%)', fontsize=12)
    
    # Add percentage labels
    for i, rate in enumerate(positive_rates):
        plt.text(i, rate + 1, f"{rate:.1f}%", ha='center', va='bottom', fontsize=12)
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_DIR, 'plots', 'positive_rate_by_source.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # Create statistics
    stats = {
        'sources': source_counts.to_dict(),
        'positive_rates': positive_rates.to_dict()
    }
    
    logger.info(f"Data source comparison complete. Found data from {len(source_counts)} sources")
    
    return stats

def generate_summary_report(all_stats):
    """
    Generate a summary report of all analyses.
    
    Parameters:
    -----------
    all_stats : dict
        Dictionary containing all analysis results
    """
    logger.info("Generating summary report")
    
    # Create report file
    report_path = os.path.join(ANALYSIS_DIR, 'analysis_summary.md')
    
    with open(report_path, 'w') as f:
        # Title and timestamp
        f.write(f"# COVID-19 Classification Dataset Analysis\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Dataset overview
        f.write("## Dataset Overview\n\n")
        if 'dataset_info' in all_stats:
            info = all_stats['dataset_info']
            f.write(f"- **Records**: {info['total_records']}\n")
            f.write(f"- **Features**: {info['total_features']}\n")
            f.write(f"- **Data sources**: {', '.join(info.get('sources', ['Unknown']))}\n\n")
        
        # Target distribution
        f.write("## Target Distribution\n\n")
        if 'target_distribution' in all_stats:
            target = all_stats['target_distribution']
            f.write(f"- **COVID-19 Positive**: {target['positive_count']} ({target['positive_pct']}%)\n")
            f.write(f"- **COVID-19 Negative**: {target['negative_count']} ({target['negative_pct']}%)\n")
            f.write(f"- **Class Ratio**: {target['class_ratio']:.2f}:1 (negative:positive)\n\n")
            
            if target['class_ratio'] > 3:
                f.write("*Note: Significant class imbalance detected. Consider using balancing techniques.*\n\n")
            
            f.write("![Target Distribution](plots/target_distribution.png)\n\n")
        
        # Missing values
        f.write("## Missing Values\n\n")
        if 'missing_values' in all_stats:
            missing = all_stats['missing_values']
            f.write(f"- **Overall missing**: {missing['overall_missing_pct']}% of all cells\n")
            f.write(f"- **Columns with missing values**: {missing['columns_with_missing']} out of {missing['columns_with_missing'] + missing['columns_complete']}\n\n")
            
            if missing['columns_with_missing'] > 0:
                f.write("### Top columns with missing values\n\n")
                f.write("| Column | Missing Count | Missing Percent |\n")
                f.write("|--------|--------------|----------------|\n")
                
                for col in missing['top_missing_columns']:
                    f.write(f"| {col['column']} | {col['count']} | {col['percent']}% |\n")
                
                f.write("\n![Missing Values](plots/missing_values.png)\n\n")
        
        # Feature importance
        f.write("## Feature Importance\n\n")
        if 'feature_importance' in all_stats and all_stats['feature_importance']:
            importance = all_stats['feature_importance']
            f.write("### Top 10 most important features\n\n")
            f.write("| Feature | Importance Score |\n")
            f.write("|---------|------------------|\n")
            
            for feature in importance['top_features'][:10]:
                f.write(f"| {feature['feature']} | {feature['importance']:.6f} |\n")
            
            f.write("\n![Feature Importance](plots/feature_importance.png)\n\n")
        
        # Feature correlations
        f.write("## Feature Correlations\n\n")
        if 'feature_correlations' in all_stats and all_stats['feature_correlations']:
            correlations = all_stats['feature_correlations']
            
            f.write("### Highest positive correlations\n\n")
            f.write("| Feature 1 | Feature 2 | Correlation |\n")
            f.write("|-----------|-----------|-------------|\n")
            
            for corr in correlations['highest_positive_correlations'][:5]:
                f.write(f"| {corr['feature1']} | {corr['feature2']} | {corr['correlation']:.4f} |\n")
            
            f.write("\n### Highest negative correlations\n\n")
            f.write("| Feature 1 | Feature 2 | Correlation |\n")
            f.write("|-----------|-----------|-------------|\n")
            
            for corr in correlations['highest_negative_correlations'][:5]:
                f.write(f"| {corr['feature1']} | {corr['feature2']} | {corr['correlation']:.4f} |\n")
            
            f.write("\n![Correlation Matrix](plots/correlation_matrix.png)\n\n")
            
            if 'correlation_with_target' in correlations and correlations['correlation_with_target']:
                f.write("### Features most correlated with COVID-19 positive outcome\n\n")
                f.write("| Feature | Correlation |\n")
                f.write("|---------|-------------|\n")
                
                for corr in correlations['correlation_with_target'][:10]:
                    f.write(f"| {corr['feature']} | {corr['correlation']:.4f} |\n")
        
        # Data source comparison
        f.write("## Data Source Comparison\n\n")
        if 'data_source_comparison' in all_stats and all_stats['data_source_comparison']:
            source_comp = all_stats['data_source_comparison']
            
            f.write("### Records by data source\n\n")
            f.write("| Source | Count |\n")
            f.write("|--------|-------|\n")
            
            for source, count in source_comp['sources'].items():
                f.write(f"| {source} | {count} |\n")
            
            f.write("\n### COVID-19 positive rate by data source\n\n")
            f.write("| Source | Positive Rate |\n")
            f.write("|--------|---------------|\n")
            
            for source, rate in source_comp['positive_rates'].items():
                f.write(f"| {source} | {rate:.2f}% |\n")
            
            f.write("\n![Data Source Comparison](plots/data_source_comparison.png)\n")
            f.write("\n![Positive Rate by Source](plots/positive_rate_by_source.png)\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        # Add automatic recommendations based on findings
        recommendations = []
        
        # Check for class imbalance
        if ('target_distribution' in all_stats and 
            all_stats['target_distribution']['class_ratio'] > 3):
            recommendations.append("- **Class Imbalance**: Consider using techniques like SMOTE, class weighting, or undersampling to address the class imbalance.")
        
        # Check for missing values
        if ('missing_values' in all_stats and 
            all_stats['missing_values']['overall_missing_pct'] > 5):
            recommendations.append("- **Missing Values**: Consider imputation strategies for features with high missing values.")
        
        # Check for highly correlated features
        if ('feature_correlations' in all_stats and 
            all_stats['feature_correlations'] and 
            len(all_stats['feature_correlations']['highest_positive_correlations']) > 0 and 
            abs(all_stats['feature_correlations']['highest_positive_correlations'][0]['correlation']) > 0.9):
            recommendations.append("- **Collinearity**: Consider removing highly correlated features to reduce dimensionality.")
        
        # Add recommendations to report
        if recommendations:
            for rec in recommendations:
                f.write(f"{rec}\n")
        else:
            f.write("- No automatic recommendations generated.\n")
    
    # Also save the statistics as JSON for programmatic access
    json_path = os.path.join(ANALYSIS_DIR, 'analysis_results.json')
    
    with open(json_path, 'w') as f:
        # Convert any numpy or pandas types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json_stats = {}
        for key, value in all_stats.items():
            if isinstance(value, dict):
                json_stats[key] = {k: convert_to_serializable(v) for k, v in value.items()}
            else:
                json_stats[key] = convert_to_serializable(value)
        
        json.dump(json_stats, f, indent=2)
    
    logger.info(f"Summary report generated: {report_path}")
    logger.info(f"Analysis results saved as JSON: {json_path}")

def main():
    """Main function to analyze the classification dataset."""
    logger.info("Starting COVID-19 classification dataset analysis")
    
    # Step 1: Load the dataset
    logger.info("Step 1: Loading the classification dataset")
    df = load_dataset()
    
    if df is None or df.empty:
        logger.error("Failed to load classification dataset. Please run prepare_classification_dataset.py first.")
        print("\nERROR: Classification dataset not found or empty.")
        print("Please run prepare_classification_dataset.py first to create the dataset.")
        sys.exit(1)
    
    # Create dataset information
    dataset_info = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'sources': set(df['record_id'].apply(lambda x: str(x).split('_')[0] if '_' in str(x) else 'Unknown'))
    }
    
    # Step 2: Run various analyses
    logger.info("Step 2: Running analyses")
    
    all_stats = {
        'dataset_info': dataset_info,
        'target_distribution': analyze_target_distribution(df),
        'missing_values': analyze_missing_values(df),
        'feature_distributions': analyze_feature_distributions(df),
        'feature_importance': analyze_feature_importance(df),
        'feature_correlations': analyze_feature_correlations(df),
        'data_source_comparison': analyze_data_source_comparison(df)
    }
    
    # Step 3: Generate summary report
    logger.info("Step 3: Generating summary report")
    generate_summary_report(all_stats)
    
    # Print analysis results
    print("\nCOVID-19 Classification Dataset Analysis Complete!")
    print(f"Total records: {dataset_info['total_records']}")
    print(f"Total features: {dataset_info['total_features']}")
    
    if 'target_distribution' in all_stats and all_stats['target_distribution']:
        target = all_stats['target_distribution']
        print(f"COVID-19 positive: {target['positive_count']} ({target['positive_pct']}%)")
        print(f"COVID-19 negative: {target['negative_count']} ({target['negative_pct']}%)")
    
    print(f"\nAnalysis results saved to: {ANALYSIS_DIR}")
    print(f"Summary report: {os.path.join(ANALYSIS_DIR, 'analysis_summary.md')}")
    print(f"Plots directory: {os.path.join(ANALYSIS_DIR, 'plots')}")

if __name__ == "__main__":
    main()