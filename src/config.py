from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    stac_api_url: str = 'https://planetarycomputer.microsoft.com/api/stac/v1'
    collection: str = 'sentinel-2-l2a'
    output_csv: str = 'data/raw/scenes.csv'

SETTINGS = Settings()