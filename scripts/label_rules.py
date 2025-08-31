# scripts/label_rules.py
'''import os
import pandas as pd
import numpy as np

def rule_based_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input DataFrame expected columns:
      pm2_5, pm10, no2, so2, o3, co, temperature, humidity, wind_speed,
      latitude, longitude
    Returns df with new column 'pollution_source' with values:
      'vehicular', 'industrial', 'agricultural', 'burning', 'natural', 'unknown'
    """
    df = df.copy()

    # Calculate thresholds based on percentiles for this dataset
    pm25_75 = df['pm2_5'].quantile(0.75)
    no2_75  = df['no2'].quantile(0.75)
    so2_75  = df['so2'].quantile(0.75)
    co_75   = df['co'].quantile(0.75)

    def label_row(r):
        pm25 = r.get("pm2_5") or 0
        no2  = r.get("no2") or 0
        so2  = r.get("so2") or 0
        co   = r.get("co") or 0
        temp = r.get("temperature") or np.nan

        # Adjusted heuristics using dataset percentiles
        if no2 > no2_75 and pm25 > pm25_75:
            return "vehicular"
        if so2 > so2_75 or (co > co_75 and so2 > so2_75 / 2):
            return "industrial"
        if pm25 > pm25_75 and (temp is not np.nan and temp > 20):
            return "agricultural"
        if pm25 > pm25_75 * 1.2 and co > co_75 * 0.8 and no2 < no2_75 / 2:
            return "burning"
        if pm25 < pm25_75 / 2 and no2 < no2_75 / 2 and so2 < so2_75 / 2:
            return "natural"
        return "unknown"

    df["pollution_source"] = df.apply(label_row, axis=1)
    return df


def main():
    input_path = os.path.join("data", "merged_realtime_data.csv")
    output_path = os.path.join("data", "labeled_features.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    print("Applying rule-based labeling...")
    df_labeled = rule_based_labeling(df)

    df_labeled.to_csv(output_path, index=False)
    print(f"Labeled data saved to: {output_path}")


if __name__ == "__main__":
    main()'''
