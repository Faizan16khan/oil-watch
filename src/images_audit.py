import pandas as pd

df = pd.read_csv('data/processed/scenes_clean.csv')

# Creating a balanced audit set based on image_validity
df_0 = df[df['image_validity'] == 0].sample(n = min(100, (df['image_validity']).sum()), random_state = 16)
df_1 = df[df['image_validity'] == 1].sample(n = min(100, (df['image_validity']).sum()), random_state = 16)

audit = pd.concat([df_0, df_1], ignore_index=True).sample(frac = 1, random_state = 16)
audit = audit[['datetime', 'id', 'cloud_cover', 'thumbnail_href']].rename(columns = {'id': 'scene_id'})

audit.to_csv("audit/audit_sample.csv", index = False)
print(f"Saved {len(audit)} rows to audit/audit_sample.csv")