import pandas as pd
import joblib

model = joblib.load(
    "models/churn_pipeline.pkl"
)

sample = pd.DataFrame([
    {
        "gender":"Female",
        "SeniorCitizen":0,
        "Partner":"Yes",
        "Dependents":"No",
        "tenure":24,
        "PhoneService":"Yes",
        "MultipleLines":"No",
        "InternetService":"Fiber optic",
        "OnlineSecurity":"No",
        "OnlineBackup":"Yes",
        "DeviceProtection":"No",
        "TechSupport":"No",
        "StreamingTV":"Yes",
        "StreamingMovies":"Yes",
        "Contract":"Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check",
        "MonthlyCharges":85,
        "TotalCharges":2000
    }
])

prediction = model.predict(sample)
print(
    "Churn Prediction:",
    prediction[0]
)