import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
file_path = 'data/loan-dataset.csv'
df = pd.read_csv(file_path)

# 2. Safe Data Cleaning & Conditional Filtering
if 'loan_amnt' in df.columns:
    if df['loan_amnt'].dtype == 'object':
        df['loan_amnt'] = df['loan_amnt'].astype(str).str.replace('£', '').str.replace(',', '').astype(float)

if 'customer_age' in df.columns:
    df = df[(df['customer_age'] >= 18) & (df['customer_age'] <= 100)]

if 'historical_default' in df.columns:
    df['historical_default'] = df['historical_default'].fillna('Unknown')

# 3. Feature Selection & Target Separation
# Default to the last column if 'target_column' is not explicitly present in df
target_col = 'target_column'
if target_col not in df.columns:
    target_col = df.columns[-1]
    print(f"Target column 'target_column' not found. Defaulting to last column: '{target_col}'")

X = df.drop(columns=[target_col])
y = df[target_col]

# Define numeric and categorical columns dynamically
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# 4. Preprocessing Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training & Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

print("--- Model Evaluation Results ---")
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

print("Pipeline execution complete!")