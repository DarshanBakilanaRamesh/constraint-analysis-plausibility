from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import time
import psutil, os
from z3 import *

# SMT solver of MLDiff modified by adding hand-written constraints in order to generate more plausible results.
# Track execution time
start_time = time.time()
process = psutil.Process(os.getpid())

# Add the ML-DIFF root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mldiff import dt2smt
from mldiff import svm2smt


# Load dataset
data = pd.read_csv("D:/Thesis/Git/ml-diff/constraints/Iris.csv")
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Species"])  # Convert species to numerical values
class_names = label_encoder.classes_

# Train Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X, y)

# Train SVM
svm = LinearSVC()
svm.fit(X, y)

# Convert Decision Tree & SVM to SMT
clDt = Int("classDT")
s = Solver()
s = dt2smt.toSMT(dt, str(clDt))

clSvm = Int("classSVM")
svm2smt.toSMT(svm, str(clSvm), s)

# Define features as SMT variables
features = []
for i in range(X.shape[1]):
    f = Real(f"x{i}")
    features.append(f)
    s.add(f >= 0, f <= 15)  # General range for features

#Train Isolation Forest for Anomaly Detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X)

# Get anomaly scores
scores = iso_forest.decision_function(X)
threshold = np.percentile(scores, 10)  # Set threshold for anomalies

# Identify anomalous points
anomalies = X[scores < threshold]

# Extract feature-wise anomaly thresholds
feature_constraints = {}
for col in X.columns:
    feature_constraints[col] = {
        "min": anomalies[col].min() if not anomalies[col].empty else X[col].min(),
        "max": anomalies[col].max() if not anomalies[col].empty else X[col].max(),
    }

# Convert Isolation Forest Constraints to SMT
for i, col in enumerate(X.columns):
    min_val = feature_constraints[col]["min"]
    max_val = feature_constraints[col]["max"]
    s.add(Not(Or(features[i] < min_val, features[i] > max_val)))

# Find Disagreement Cases Using SMT
dataset = []
s.add(clDt == clSvm)  # Disagreement condition

def all_smt(s, terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if sat == s.check():
            m = s.model()
            yield m
            for i in range(len(terms)):
                s.push()
                block_term(s, m, terms[i])
                for j in range(i):
                    fix_term(s, m, terms[j])
                yield from all_smt_rec(terms[i:])
                s.pop()
    yield from all_smt_rec(list(terms))

if s.check() != sat:
    for m in all_smt(s, features):
        data_instance = []
        for f in features:
            data_instance.append(float(m.eval(f, model_completion=True).as_fraction()))
        data_instance.append(m.eval(clDt, model_completion=True).as_long())  # Decision Tree Label
        dataset.append(data_instance)
        if len(dataset) >= 900:
            break

# Replace class labels with actual species names
for row in dataset:
    row[-1] = class_names[row[-1]]

# Save generated dataset to CSV
header = list(X.columns) + ["Species"]
with open("iris_diff_data2_iforest.csv", mode="w") as file:
    import csv
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(dataset)

# Count Outliers in the Generated Dataset
def is_outlier(row):
    for i, col in enumerate(X.columns):
        min_val = feature_constraints[col]["min"]
        max_val = feature_constraints[col]["max"]
        if row[col] < min_val or row[col] > max_val:
            return True
    return False

outlier_count = sum(is_outlier(dict(zip(header, row))) for row in dataset)

# Print results
print(f"Generated {len(dataset)} disagreement cases")
print(f"Number of Outliers: {outlier_count}")
print(f"Data Generation Time: {time.time() - start_time:.2f} seconds")
print(f"Memory Used: {process.memory_info().rss / 1024**2:.2f} MB")
