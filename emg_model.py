# EMG SIGNAL CLASSIFICATION PROJECT

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FILE = "dataset.csv"

print(f"Loading dataset from: {os.path.abspath(FILE)}")
if not os.path.exists(FILE):
    raise FileNotFoundError(f"Dataset file not found: {FILE}")

# 2. Load Dataset
try:
    data = pd.read_csv(FILE)
except Exception as e:
    raise RuntimeError(f"Failed to read CSV: {e}")

print("Dataset Loaded Successfully ?")
print(data.head())
#ADD THIS LINE EXACTLY HERE
data = data.sample(20000, random_state=42)
print("Reduced Dataset Size:", data.shape)
print(data.columns.tolist())
print(f"Dataset rows: {len(data)}, columns: {data.shape[1]}")

# 3. Preprocessing
if data.isna().any().any():
    print("Found missing values. Dropping rows with NaNs.")
    data = data.dropna()

# Choose label column
label_candidates = [c for c in ["label", "class"] if c in data.columns]
if not label_candidates:
    raise ValueError("No 'label' or 'class' column found. Please set proper label column name.")
label_col = label_candidates[0]
print(f"Using label column: {label_col}")

X = data.drop(label_col, axis=1)
y = data[label_col]

# If there are remaining non-numeric columns, drop or encode
non_numeric = X.select_dtypes(include=[object]).columns.tolist()
if non_numeric:
    print(f"Converting non-numeric columns: {non_numeric}")
    X = pd.get_dummies(X, columns=non_numeric)

print(f"Feature columns: {X.shape[1]}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y.astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]}")

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("RandomForest trained")

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
print("SVM trained")

rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

print("\n=== Accuracy ===")
print("Random Forest Accuracy:", rf_acc)
print("SVM Accuracy:", svm_acc)

best_model, best_pred = (rf_model, rf_pred) if rf_acc >= svm_acc else (svm_model, svm_pred)
print("\nBest Model Selected ?", "RandomForest" if rf_acc >= svm_acc else "SVM")

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
print('Confusion matrix saved to confusion_matrix.png')

print("\n=== Classification Report ===")
print(classification_report(y_test, best_pred))

sample = X_test[0].reshape(1, -1)
prediction = best_model.predict(sample)
gesture = encoder.inverse_transform(prediction)

print("\nSample Prediction:")
print("Predicted Gesture:", gesture[0])
