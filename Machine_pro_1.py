# -----------------------------------------------------------
# Sonar Mine–Rock Detection Script (Minimal Output Version)
# -----------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------
# File path to your dataset
# -----------------------------------------------------------
data_path = Path(r"C:\Users\Suhas\OneDrive\Desktop\MIT\Guide\Copy of sonar data.csv")

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
sonar_data = pd.read_csv(data_path, header=None)

# -----------------------------------------------------------
# Split features and labels
# -----------------------------------------------------------
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# -----------------------------------------------------------
# Train Logistic Regression model
# -----------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# -----------------------------------------------------------
# Predictive system (single sample)
# -----------------------------------------------------------
input_data = (
    0.0303,0.0374,0.0435,0.0698,0.1183,0.2085,0.3253,0.3545,0.3956,0.4159,
    0.4291,0.4203,0.3954,0.3471,0.3139,0.2938,0.2540,0.2090,0.1880,0.1820,
    0.1450,0.1100,0.0880,0.0675,0.0493,0.0392,0.0325,0.0264,0.0213,0.0178,
    0.0145,0.0120,0.0102,0.0086,0.0069,0.0060,0.0050,0.0042,0.0035,0.0030,
    0.0024,0.0020,0.0016,0.0012,0.0010,0.0008,0.0006,0.0005,0.0004,0.0003,
    0.0003,0.0002,0.0002,0.0001,0.0001,0.0001,0.0001,0.0001,0.0000,0.0000
)
input_np = np.array(input_data).reshape(1, -1)

# Predict label
pred_label = model.predict(input_np)[0]

# Prediction probabilities (optional but useful)
probs = model.predict_proba(input_np)[0]

# -----------------------------------------------------------
# FINAL OUTPUT ONLY
# -----------------------------------------------------------
print("\n======= FINAL PREDICTION =======")
print("Predicted class:", pred_label)

if pred_label == "M":
    print("Result → The object is predicted to be a **MINE**.")
else:
    print("Result → The object is predicted to be a **ROCK**.")

print("Confidence (Mine, Rock):", probs)
print("================================\n")
