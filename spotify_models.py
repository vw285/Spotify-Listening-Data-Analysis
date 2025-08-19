
"""
Authors: Violet Wang, Vanessa Lattes, Michele Maslowski
Converted from R to Python
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# =====================
# Load and Clean Data
# =====================
data = pd.read_csv("spotify-2023.csv")

# Convert streams to numeric and create binary target
data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
mean_streams = data['streams'].mean()
data['average_streams'] = np.where(data['streams'] >= mean_streams, 1, 0)

# Drop NA and irrelevant columns
drop_cols = [
    'key','artist.s._name','track_name','in_deezer_playlists',
    'in_deezer_charts','in_shazam_charts','in_spotify_charts',
    'in_apple_playlists','in_apple_charts'
]
cleaned = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
cleaned = cleaned.dropna()

# Features and target
X = cleaned.drop(columns=['average_streams','streams'], errors='ignore')
y = cleaned['average_streams']

# Encode categorical variables if any
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# KNN Model
# =====================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# ROC for KNN
y_proba_knn = knn.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba_knn)
plt.plot(fpr, tpr, label=f'KNN AUC={roc_auc_score(y_test, y_proba_knn):.2f}')

# =====================
# Logistic Regression
# =====================
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

y_proba_log = log_reg.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba_log)
plt.plot(fpr, tpr, label=f'LogReg AUC={roc_auc_score(y_test, y_proba_log):.2f}')

# =====================
# Support Vector Machine
# =====================
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

y_proba_svm = svm.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba_svm)
plt.plot(fpr, tpr, label=f'SVM AUC={roc_auc_score(y_test, y_proba_svm):.2f}')

# =====================
# Random Forest
# =====================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

y_proba_rf = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_proba_rf)
plt.plot(fpr, tpr, label=f'RF AUC={roc_auc_score(y_test, y_proba_rf):.2f}')

# =====================
# Plot ROC Curves
# =====================
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Models")
plt.legend()
plt.show()
