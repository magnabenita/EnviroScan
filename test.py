'''import requests

API_KEY = "8ecb7a70686c9052a14f9130d69679e3816fc99ca476a2fee7df4a08dc2e5dae"  # your key

headers = {"X-API-Key": API_KEY}
resp = requests.get("https://api.openaq.org/v3/locations", headers=headers, params={"limit": 10})

print(resp.status_code)
print(resp.json())
'''
import pandas as pd
df = pd.read_csv("data/labeled_features.csv")
print(df.columns)
