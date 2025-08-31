# scripts/merge_for_labeling.py
import pandas as pd

# Load CSVs
pollution = pd.read_csv('data/pollution_data.csv')
weather = pd.read_csv('data/weather_data.csv')
locations = pd.read_csv('data/global_locations_cleaned.csv')

# Merge pollution + weather on location_id
df = pollution.merge(weather, left_on='location_id', right_on='location_id', how='left')

# Extract latitude and longitude from global_locations_cleaned.csv (the JSON-like 'coordinates' field)
# In your file, 'coordinates' is like "{'latitude': 5.58389, 'longitude': -0.19968}"
# We need to parse out the numbers
import ast

def extract_lat_lon(coord_str):
    try:
        d = ast.literal_eval(coord_str)
        return pd.Series([d.get('latitude'), d.get('longitude')])
    except:
        return pd.Series([None, None])

latlon = locations[['id','coordinates']].copy()
latlon[['latitude','longitude']] = latlon['coordinates'].apply(extract_lat_lon)

df = df.merge(latlon[['id','latitude','longitude']], left_on='location_id', right_on='id', how='left')
df.drop(columns=['id'], inplace=True)

# Save merged file
df.to_csv('data/merged_realtime_data.csv', index=False)
print("✅ Saved data/merged_realtime_data.csv for labeling.")
