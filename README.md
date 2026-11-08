# Heart Disease Classification

## Student Information

Name: MAGESHWARAN P  
BITS ID: 2026AK05470

## Problem Statement

Predict the presence of heart disease using 13 clinical and
demographic features.

## Dataset

- Dataset: `heart.csv`
- Records: 1,025
- Input features: 13
- Target column: `target`
- Class 0: No Heart Disease
- Class 1: Heart Disease

## Models Implemented

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## Evaluation Metrics

- Accuracy
- ROC AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient

## Streamlit Application

The application evaluates all five models on the same uploaded
labelled CSV file, ranks the models using F1 Score, identifies
the winner, and supports detailed model comparison.

## Repository Files

- `app.py`
- `requirements.txt`
- `heart.csv`
- `test_data.csv`
- `ML_Assignment_2.ipynb`
- `models/`
- `artifacts/`