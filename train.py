import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 1. load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
print("Dataset Shape:", df.shape)

# 2. remove customer ID/data cleaning
df.drop("customerID", axis=1, inplace=True)

#convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
#convert target variable
df["Churn"] = df["Churn"].map(
    {"Yes": 1, "No": 0}
)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. column identification
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns

print("\nNumeric Features:")
print(list(numeric_features))
print("\nCategorical Features:")
print(list(categorical_features))


# 4. numeric pipeline
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# 5. categorical pipeline
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# 6. column transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# 7. full pipeline
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression()
        )
    ]
)

# 8. grid search parameters
param_grid = [
    # logistic
    {
        "classifier": [
            LogisticRegression(
                max_iter=1000
            )
        ],

        "classifier__C": [
            0.01,
            0.1,
            1,
            10
        ]
    },
    # random forest
    {
        "classifier": [
            RandomForestClassifier(
                random_state=42
            )
        ],

        "classifier__n_estimators": [
            100,
            200
        ],

        "classifier__max_depth": [
            5,
            10,
            None
        ]
    }
]

# 9. grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)
print("\nTraining Models...\n")
grid_search.fit(
    X_train,
    y_train
)

# 10. best model
print("\nBest Parameters:")
print(grid_search.best_params_)
print("\nBest Cross Validation Score:")
print(grid_search.best_score_)

# 11. test evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    y_pred
)
print("\nTest Accuracy:")
print(round(accuracy, 4))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# 12. save pipeline
joblib.dump(
    best_model,
    "models/churn_pipeline.pkl"
)
print(
    "\nPipeline saved as models/churn_pipeline.pkl"
)

# 13. feature transformation 
transformed_data = preprocessor.fit_transform(X)
print(
    "\nPreprocessing completed successfully."
)
print(
    "Model training completed successfully."
)