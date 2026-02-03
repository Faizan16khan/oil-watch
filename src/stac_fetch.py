import pandas as pd
from pystac_client import Client
import planetary_computer as pc

from .config import SETTINGS

def fetch_scene(limit: int = 25) -> pd.DataFrame:
    client = Client.open(SETTINGS.stac_api_url)

    bbox = [-96.804, 35.912, -96.721, 35.989]


    search = client.search(
        collections = [SETTINGS.collection],
        bbox = bbox,
        max_items = limit
    )

    items = list(search.get_items())
    rows = []
    for item in items:
        signed_item = pc.sign(item)
        rows.append(
            {
            'id': item.id,
            'datetime': item.properties.get('datetime'),
            'cloud_cover': item.properties.get('eo:cloud_cover'),
            'bbox': item.bbox,
            'visual_href': signed_item.assets.get("visual"),
            'preview' : signed_item.assets.get("preview"),
            'rendered_preview': signed_item.assets.get("rendered_preview")
            }
        )

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = fetch_scene(limit=16000)
    df.to_csv(SETTINGS.output_csv, index=False)
    print(f"Saved {len(df)} scenes to {SETTINGS.output_csv}")

print(df[['visual_href', 'preview', 'rendered_preview']])

print(df['preview'][1436])