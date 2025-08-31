import pandas as pd
import random

# === Config ===
LOCATIONS_FILE = "data/global_locations_cleaned.csv"
WEATHER_FILE = "data/weather_data.csv"
OUTPUT_FILE = "data/processed_data.csv"

# === Load Data ===
df_locations = pd.read_csv(LOCATIONS_FILE)
df_weather = pd.read_csv(WEATHER_FILE)

# Ensure consistent column names
if 'coordinates' in df_locations.columns:
    # Extract latitude and longitude if stored in JSON-like strings
    df_locations['latitude'] = df_locations['coordinates'].str.extract(r"'latitude':\s*([-\d.]+)").astype(float)
    df_locations['longitude'] = df_locations['coordinates'].str.extract(r"'longitude':\s*([-\d.]+)").astype(float)

# === Merge Data ===
df = pd.merge(
    df_locations,
    df_weather,
    left_on="id",
    right_on="location_id",
    how="left"
)

# === Select Relevant Columns ===
columns_to_keep = [
    "id",
    "name",
    "locality",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "wind_speed"
]
df = df[columns_to_keep]

# === Add Dummy Target (for now) ===
sources = ['industrial', 'vehicular', 'agricultural', 'natural']
df['pollution_source'] = [random.choice(sources) for _ in range(len(df))]

# === Save Preprocessed Data ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Preprocessed data saved to {OUTPUT_FILE} (rows: {len(df)})")
