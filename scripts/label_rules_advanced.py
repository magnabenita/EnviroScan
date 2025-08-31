# scripts/label_rules_advanced.py
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point

def compute_spatial_features(df):
    """
    Adds distance-to-road, distance-to-factory, distance-to-farmland features
    """
    print("Creating GeoDataFrame for sensor locations...")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    print("Projecting coordinates to UTM for accurate distance calculations...")
    gdf = gdf.to_crs(epsg=32630)  # UTM zone for Accra

    print("Loading OSM road network for Accra...")
    G = ox.graph_from_place("Accra, Ghana", network_type="drive")
    roads = ox.graph_to_gdfs(G, nodes=False, edges=True).to_crs(epsg=32630)
    print(f"Number of road segments loaded: {len(roads)}")

    print("Computing distance to nearest road for each sensor...")
    gdf['dist_to_road_m'] = gdf.geometry.apply(lambda point: roads.distance(point).min())

    print("Fetching industrial areas from OSM...")
    tags = {'industrial': True}
    factories = ox.features_from_place("Accra, Ghana", tags)
    if not factories.empty:
        factories = factories.to_crs(epsg=32630)
        print(f"Number of factories found: {len(factories)}")
        gdf['dist_to_factory_m'] = gdf.geometry.apply(lambda point: factories.distance(point).min())
    else:
        print("No factories found in OSM data.")
        gdf['dist_to_factory_m'] = np.nan

    print("Fetching farmland areas from OSM...")
    tags = {'landuse': 'farmland'}
    farmland = ox.features_from_place("Accra, Ghana", tags)
    if not farmland.empty:
        farmland = farmland.to_crs(epsg=32630)
        print(f"Number of farmland polygons found: {len(farmland)}")
        gdf['dist_to_farmland_m'] = gdf.geometry.apply(lambda point: farmland.distance(point).min())
    else:
        print("No farmland found in OSM data.")
        gdf['dist_to_farmland_m'] = np.nan

    print("Spatial features computed.")
    return gdf.drop(columns='geometry')


def rule_based_labeling(df):
    """
    Labels pollution sources using pollutant + spatial features
    """
    print("Applying rule-based labeling to each row...")
    df = df.copy()

    def label_row(r):
        pm25 = r.get("pm2_5") or 0
        no2  = r.get("no2") or 0
        so2  = r.get("so2") or 0
        co   = r.get("co") or 0
        temp = r.get("temperature") or np.nan
        dist_road = r.get("dist_to_road_m", 999999)
        dist_factory = r.get("dist_to_factory_m", 999999)
        dist_farmland = r.get("dist_to_farmland_m", 999999)

        if no2 > 60 and pm25 > 30 and dist_road < 500:
            return "vehicular"
        if so2 > 40 or (co > 2.0 and dist_factory < 1000):
            return "industrial"
        if pm25 > 80 and temp > 20 and dist_farmland < 2000:
            return "agricultural"
        if pm25 > 100 and co > 1.5 and no2 < 30:
            return "burning"
        if pm25 < 20 and no2 < 20 and so2 < 10:
            return "natural"
        return "unknown"

    df['pollution_source'] = df.apply(label_row, axis=1)
    print("Labeling complete.")
    return df


def main():
    input_path = os.path.join("data", "merged_realtime_data.csv")
    output_path = os.path.join("data", "labeled_features.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Number of rows loaded: {len(df)}")

    df = compute_spatial_features(df)
    df_labeled = rule_based_labeling(df)

    df_labeled.to_csv(output_path, index=False)
    print(f"Labeled data saved to: {output_path}")
    print("Process completed successfully!")


if __name__ == "__main__":
    main()
