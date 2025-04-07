# No-Code ML Platform

A user-friendly machine learning platform that allows you to perform data analysis and model training without writing code.

## Features

- Data Upload: Support for CSV and Excel files
- Exploratory Data Analysis (EDA)
- Categorical Data Analysis
- Data Preprocessing
- Model Training with multiple algorithms
- Model Comparison
- Performance Metrics and Visualizations

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```
   streamlit run app.py
   ```

2. Upload your dataset (CSV or Excel file)
3. Navigate through the different pages using the sidebar:
   - Main page: Data upload and EDA
   - Categorical Analysis: Analysis of categorical variables
   - Data Preprocessing: Handle missing values and encode categorical variables
   - Model Training: Train and compare different machine learning models

## Supported File Types

- CSV files (.csv)
- Excel files (.xlsx, .xls)

## Model Types

### Classification
- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- SVM
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest
- Gradient Boosting
- AdaBoost
- SVM
- Decision Tree
- K-Nearest Neighbors

## Validation Methods

- Train-Test Split (80:20)
- 5-Fold Cross Validation 