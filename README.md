# Production-Grade Customer Churn ML Pipeline

An end-to-end Machine Learning pipeline for predicting customer churn using Scikit-learn Pipeline API. The project demonstrates production-ready ML practices including automated preprocessing, hyperparameter tuning, model evaluation, and pipeline export for deployment.

---

## Project Objective

Customer churn prediction helps businesses identify customers likely to leave a service. This project builds a reusable machine learning pipeline capable of:

- Handling missing values
- Encoding categorical features
- Scaling numerical features
- Training multiple models
- Hyperparameter optimization using GridSearchCV
- Exporting the final pipeline for future predictions

---

## Dataset

**Telco Customer Churn Dataset**: The dataset contains customer demographics, account information, services subscribed, billing details, and churn status.

Target Variable:

- Churn = Yes (1)
- Churn = No (0)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Machine Learning Workflow

### 1. Data Preprocessing

Scikit-learn's Pipeline API to automate preprocessing.

#### Numerical Features

- Missing value imputation (Median)
- Standard Scaling

#### Categorical Features

- Missing value imputation (Most Frequent)
- One-Hot Encoding

---

### 2. Model Training

Two machine learning algorithms evaluated:

#### - Logistic Regression

Used as a baseline linear classifier.

#### - Random Forest Classifier

Used for capturing complex non-linear relationships.

---

### 3. Hyperparameter Tuning

GridSearchCV used with 5-fold Cross Validation to automatically find the best model configuration.

Parameters explored:

#### - Logistic Regression

- C = 0.01
- C = 0.1
- C = 1
- C = 10

#### - Random Forest

- n_estimators = 100, 200
- max_depth = 5, 10, None

---

### 4. Evaluation Metrics

The trained models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Classification Report

---

### 5. Model Export

The best-performing pipeline is saved using Joblib:

```python
joblib.dump(best_model, "models/churn_pipeline.pkl")
```

The exported file contains:

- Data preprocessing pipeline
- Feature encoding
- Feature scaling
- Trained machine learning model

---

## Project Structure

```text
customer-churn-pipeline/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   └── churn_pipeline.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---
## Commands Summary
python -m venv venv

venv\Scripts\activate

pip install pandas numpy scikit-learn joblib

python src/train.py

python src/predict.py

## Training the Model

Run:

```bash
python src/train.py
```

The script will:

- Load the dataset
- Preprocess features
- Train multiple models
- Tune hyperparameters
- Evaluate performance
- Save the best pipeline

---

## Making Predictions

Run:

```bash
python src/predict.py
```

Example Output:

```text
Churn Prediction: 1
```

Where:

- 1 = Customer likely to churn
- 0 = Customer likely to stay

---

## Skills Demonstrated

- Data Preprocessing
- Feature Engineering
- Scikit-learn Pipeline API
- ColumnTransformer
- Hyperparameter Optimization
- GridSearchCV
- Model Evaluation
- Model Serialization
- Production-Ready ML Workflow

---

## Future Improvements

- XGBoost Integration
- LightGBM Integration
- SHAP Explainability
- FastAPI Deployment
- Docker Containerization
- CI/CD Automation

---
