# Autism-Spectrum-Disorder-Detection-using-ML
A Machine Learning project for early and efficient detection of Autism Spectrum Disorder (ASD), using dimensionality reduction and multiple classification models.

## Overview
This project applies machine learning techniques to detect Autism Spectrum Disorder using a dataset compiled from multiple sources. The pipeline includes data preprocessing, exploratory data analysis, feature selection using Information Gain (Entropy & Gini Index), and model comparison between SVM, ANN, and MLP.
**Key Outcome: The Multilayer Perceptron (MLP) model using entropy-selected features achieved the highest accuracy of 89.6%.**

## Problem Statement
Diagnosing ASD is challenging due to the complexity and variability of symptoms, often leading to misdiagnosis or delays. This project aims to create an automated, accurate, and accessible screening method using ML.

## Data
- Sources: Three publicly available Kaggle datasets spanning toddlers, children, and adults.
  - Kaggle Links to Datasets:
     - [Link Text]https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers/data
     - [Link Text]https://www.kaggle.com/datasets/uppulurimadhuri/dataset/data
     - [Link Text]https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults/data

- Screening Basis: A1–A9 question-based assessments plus demographic and clinical attributes.
- Target: Binary classification of ASD traits (Yes/No).

## Data Preprocessing
Unified datasets by standardizing column names and selecting shared features.

**Addressed:**
- Missing values (imputed using mode)
- Irregular cardinalities (standardized labels)
- Label encoding for categorical variables

## Exploratory Data Analysis
- Correlation Heatmap: Revealed inverse relationships between age and several screening questions (A6–A9).
- Bar Plots: Visualized skewed distributions in ethnicity, gender, and test completion methods, revealing insights into dataset bias and balance.

## Methodology
### Feature Selection
Selected top 10 out of 16 features using:
- Information Gain (Entropy): Slightly better performance
- Information Gain (Gini Index): Similar results, slight feature variation

### Models Used
#### Support Vector Machine (SVM)
 - RBF kernel
 - C = 1.0, gamma = 'scale'

#### Artificial Neural Network (ANN)
 - Hidden Layer: 32 neurons (ReLU), Output: Sigmoid
 - Optimizer: Adam, Loss: Binary Crossentropy

#### Multilayer Perceptron (MLP)
- 1 hidden layer (32 neurons, ReLU)
- Output layer: Softmax
- Max Iterations: 500, Random State: 42

## Evaluation Results
| Model | Entropy Accuracy | Gini Accuracy |
|-------|------------------|---------------|
| MLP   | 89.6%            | 88.2%         |
| ANN   | 89.3%            | 83.8%         |
| SVM   | 86.0%            | 82.0%         |

**MLP outperformed both SVM and ANN in both feature settings.**

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook / Google Colab

## Future Enhancements
- Add ensemble models for better accuracy
- Increase demographic diversity in datasets
- Deploy a screening web app for real-time user interaction
