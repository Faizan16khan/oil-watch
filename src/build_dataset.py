import pandas as pd
from datetime import datetime

df = pd.read_csv("data/raw/scenes.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# Sorting the values by datetime to prepare data for further feature engineering
df = df.sort_values('datetime', ascending=True)

# Extracting distnict meaningful features from datetime 
df['date'] = df['datetime'].dt.date
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['day_of_year'] = df['datetime'].dt.day_of_year

# Engineering revisit_gap_hours feature from the datetime feature
df['revisit_gap_hours'] = (df['datetime'].diff().dt.total_seconds()/3600).fillna(0)

# Engineering image_validity feature from cloud_cover
# Building Label v1 of the image validity score
def image_validity(cloud):
    if cloud <= 30:
        return 1
    else:
        return 0
df['image_validity'] = df['cloud_cover'].apply(image_validity)

# Saving the clean dataset
df.to_csv("data/processed/scenes_clean.csv", index=False)
