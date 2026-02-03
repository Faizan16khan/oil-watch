import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/scenes_clean.csv")

# Defining the features with and without cloud_cover
features = df.drop(['image_validity', 'visual_href', 'thumbnail_href', 'rendered_preview'], axis = 1)
features_without_cloud_cover = df.drop(labels=['image_validity', 'cloud_cover', 'visual_href', 'thumbnail_href', 'rendered_preview'], axis = 1)
target = df['image_validity']

# Splitting into train and test set for with and without cloud_cover
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=16,
    shuffle = False)

X_train_without_cloud_cover, X_test_without_cloud_cover, y_train, y_test = train_test_split(
    features_without_cloud_cover,
    target,
    test_size=0.2,
    random_state=16,
    shuffle = False)

# Only selecting numerical columns from the dataset
X_train_without_cloud_cover = X_train_without_cloud_cover.select_dtypes(include="number")
X_test_without_cloud_cover  = X_test_without_cloud_cover.select_dtypes(include="number")
X_train = X_train.select_dtypes(include="number")
X_test  = X_test.select_dtypes(include="number")

X_train.to_csv("data/processed/img_validity_X_train.csv", index = False)
y_train.to_csv("data/processed/img_validity_y_train.csv", index = False)
X_test.to_csv("data/processed/img_validity_X_test.csv", index = False)
y_test.to_csv("data/processed/img_validity_y_test.csv", index = False)
X_train_without_cloud_cover.to_csv("data/processed/img_validity_X_train_without_cloud_cover.csv", index = False)
X_test_without_cloud_cover.to_csv("data/processed/img_validity_X_test_without_cloud_cover.csv", index = False)
