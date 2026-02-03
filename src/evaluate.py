import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pickle
import json

X_train = pd.read_csv("data/processed/img_validity_X_train.csv")
X_test = pd.read_csv("data/processed/img_validity_X_test.csv")
y_train = pd.read_csv("data/processed/img_validity_y_train.csv")
y_test = pd.read_csv("data/processed/img_validity_y_test.csv")
X_train_without_cloud_cover = pd.read_csv("data/processed/img_validity_X_train_without_cloud_cover.csv")
X_test_without_cloud_cover = pd.read_csv("data/processed/img_validity_X_test_without_cloud_cover.csv")

filepath_model = 'artifacts/models/baseline_logistic_regression_model.pkl'
filepath_model_without_cloud_cover = 'artifacts/models/baseline_logistic_regression_model_with_cloud_cover.pkl'

# Opening the saved baseline logistic regression models
with open(filepath_model, 'rb') as file:
    loaded_model = pickle.load(file)

with open(filepath_model_without_cloud_cover, 'rb') as file:
    loaded_model_without_cloud_cover = pickle.load(file)

# Making predictions on the models trained with and without cloud_cover
y_pred = loaded_model.predict(X_test)
y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]

y_pred_without_cloud_cover  = loaded_model_without_cloud_cover.predict(X_test_without_cloud_cover)
y_pred_proba_without_cloud_cover  = loaded_model_without_cloud_cover.predict_proba(X_test_without_cloud_cover)[:, 1]

#  Evaluating metrics for both models
roc_auc = roc_auc_score(y_test, y_pred_proba)
classification_rep = classification_report(y_test, y_pred, output_dict=True)
print("AUC (baseline logitic regression):", roc_auc)
print(classification_rep)
cm = confusion_matrix(y_test, y_pred)
confusion_matrix_display = ConfusionMatrixDisplay.from_estimator(loaded_model,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=['Not Valid', 'Valid'])
plt.title("Confusion Matrix of the entire data")
plt.show()

roc_auc_without_cloud_cover = roc_auc_score(y_test, y_pred_proba_without_cloud_cover)
classification_rep_without_cloud_cover = classification_report(y_test, y_pred_without_cloud_cover, output_dict=True)
print("AUC (baseline logitic regression without cloud_cover):", roc_auc_without_cloud_cover)
print(classification_rep_without_cloud_cover)
cm_without_cloud_cover = confusion_matrix(y_test, y_pred_without_cloud_cover)
confusion_matrix_display_without_cloud_cover = ConfusionMatrixDisplay.from_estimator(loaded_model_without_cloud_cover,
                      X_test_without_cloud_cover,
                      y_test,
                      values_format='d',
                      display_labels=['Not Valid', 'Valid'])
plt.title("Confusion Matrix without cloud_cover")
plt.show()

# Saving the metrics for both the models
metrics_with_cloud_cover = {
    "roc_auc": float(roc_auc),
    "classification_report": classification_rep,
    "confusion_matrix": cm.tolist(),
    "features": list(X_test.columns),
    "n_samples": int(len(y_test))
}

metrics_without_cloud_cover = {
    "roc_auc": float(roc_auc_without_cloud_cover),
    "classification_report": classification_rep_without_cloud_cover,
    "confusion_matrix": cm_without_cloud_cover.tolist(),
    "features": list(X_test_without_cloud_cover.columns),
    "n_samples": int(len(y_test))
}

with open('artifacts/metrics/baseline_with_cloud_cover.json', 'w') as file:
    json.dump(metrics_with_cloud_cover, file, indent = 2)

with open('artifacts/metrics/baseline_without_cloud_cover.json', 'w') as file:
    json.dump(metrics_without_cloud_cover, file, indent = 2)

# A simple logistic regression using metadata achieves meaningful discrimination of image usability. Removing cloud_cover collapses performance to near-random, confirming it is the dominant signal. This validates the pipeline and motivates auditing the heuristic label.