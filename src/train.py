import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

X_train = pd.read_csv("data/processed/img_validity_X_train.csv")
X_test = pd.read_csv("data/processed/img_validity_X_test.csv")
y_train = pd.read_csv("data/processed/img_validity_y_train.csv")
y_test = pd.read_csv("data/processed/img_validity_y_test.csv")
X_train_without_cloud_cover = pd.read_csv("data/processed/img_validity_X_train_without_cloud_cover.csv")
X_test_without_cloud_cover = pd.read_csv("data/processed/img_validity_X_test_without_cloud_cover.csv")

logisticreg = LogisticRegression(random_state=16, max_iter=1000)
logisticreg_without_cloud_cover = LogisticRegression(random_state=16, max_iter=1000)

# Running the baseline logistic regression model on data with cloud_cover
baseline_log_reg_model = logisticreg.fit(X_train, y_train)

# Running the baseline logistic regression model on data without cloud_cover
baseline_log_reg_model_without_cloud_cover = logisticreg_without_cloud_cover.fit(X_train_without_cloud_cover, y_train)

# Saving the result of the logistic regression models
filepath_model = 'artifacts/models/baseline_logistic_regression_model.pkl'
filepath_model_without_cloud_cover = 'artifacts/models/baseline_logistic_regression_model_with_cloud_cover.pkl'

with open(filepath_model, 'wb') as file:
    pickle.dump(baseline_log_reg_model, file)

with open(filepath_model_without_cloud_cover, 'wb') as file:
    pickle.dump(baseline_log_reg_model_without_cloud_cover, file)

print(f"The output of the baseline logistic regression model is saved in: {filepath_model}")
print(f"The output of the baseline logistic regression model without cloud_cover is saved in: {filepath_model_without_cloud_cover}")


print(X_train_without_cloud_cover.columns)
print(X_train.columns)