# Credit-Card-Fraud-Prediction-using-Machine-Learning

## Project Overview

This project aims to develop a machine learning model to predict fraudulent credit card transactions. By leveraging various data analysis and machine learning techniques, the goal is to identify potential fraud cases with high accuracy. This README provides an overview of the project, installation instructions, usage guidelines, and more.


## Project Description

The Credit Card Fraud Prediction project involves building a machine learning model to detect fraudulent transactions from a dataset of credit card transactions. The dataset contains various features of transactions, and the model's objective is to classify transactions as fraudulent or legitimate.

**Key Features:**
- Metadata Analysis
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Model training using different machine learning algorithms
- Evaluation and comparison of model performance

## Data

The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets?search=credit+card+fraud) from Kaggle. It includes:
- **Features:** Various attributes of the transactions
- **Target Variable:** `Class` (1 for fraudulent transactions, 0 for legitimate transactions)

**Note:** The dataset has been anonymized for privacy reasons.

## Technologies Used

- **Python** for data analysis and machine learning
- **Pandas** for data manipulation
- **NumPy** for numerical operations
- **Scikit-learn** for machine learning algorithms and evaluation
- **Matplotlib** and **Seaborn** for data visualization
- **Jupyter Notebook** for interactive analysis and documentation

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Credit-Card-Prediction-Using-Machine-Learning.git
   cd Credit-Card-Prediction-Using-Machine-Learning
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Open the Jupyter Notebook:**
   Launch Jupyter Notebook and open `Credit Card Fraud Prediction.ipynb` to start exploring and analyzing the data.

2. **Data Preprocessing:**
   Follow the steps in the notebook for data cleaning, feature engineering, and splitting the dataset.

3. **Model Training:**
   The notebook contains sections where different machine learning models are trained, including Logistic Regression, Random Forest, and Gradient Boosting.

4. **Results Analysis:**
   Review the results and performance metrics in the notebook, including Accuracy,precision, recall, and F1-score. Visualizations are provided to help interpret model performance.


## Results

The project includes detailed analysis of the model's performance. Key metrics include:
- Accuracy
- Precision
- Recall
- F1-Score

Refer to the `Credit Card Fraud Prediction.ipynb` notebook for detailed analysis and visualizations.


## Final Model

The model build using RandomForest machine learning algorithm gives more accuracy score than other machine learning model which is finilized and saved for future prediction.
