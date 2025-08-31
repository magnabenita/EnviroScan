# app.py
import streamlit as st
import pandas as pd
import requests
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
from concurrent.futures import ThreadPoolExecutor

# ----------------- Load Model & Label Encoder -----------------
model = joblib.load("models/pollution_model.pkl")
le = joblib.load("models/label_encoder.pkl")

POLLUTANTS = ['pm2_5','pm10','no2','so2','co','o3']

# ----------------- Config -----------------
OPENAQ_KEY = "8ecb7a70686c9052a14f9130d69679e3816fc99ca476a2fee7df4a08dc2e5dae"
OPENWEATHER_KEY = "e0f3cf0d7c2c77cb00a6e0d258cee192"

# ----------------- Functions -----------------
@st.cache_data(ttl=300)
def fetch_locations(limit=100):
    url = "https://api.openaq.org/v3/locations"
    headers = {"X-API-Key": OPENAQ_KEY}
    try:
        resp = requests.get(url, headers=headers, params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Failed to fetch locations: {e}")
        return pd.DataFrame()
    
    rows = []
    for loc in data.get("results", []):
        if 'coordinates' not in loc:
            continue
        rows.append({
            "location_id": loc.get("id"),
            "location": loc.get("name", "Unknown"),
            "latitude": loc['coordinates']['latitude'],
            "longitude": loc['coordinates']['longitude']
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300)
def fetch_pollution(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"
    try:
        resp = requests.get(url, timeout=10).json()
        if 'list' in resp and len(resp['list']) > 0:
            comp = resp['list'][0].get('components', {})
            aqi = resp['list'][0].get('main', {}).get('aqi')
            data = {p: comp.get(p) for p in POLLUTANTS}
            data['aqi'] = aqi
            return data
    except:
        pass
    return {p: None for p in POLLUTANTS + ['aqi']}

@st.cache_data(ttl=300)
def fetch_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=10).json()
        return {
            "temperature": resp['main']['temp'],
            "humidity": resp['main']['humidity'],
            "wind_speed": resp['wind']['speed']
        }
    except:
        return {"temperature":0, "humidity":0, "wind_speed":0}

def predict_pollution(df):
    features = ["pm2_5","pm10","no2","so2","co","o3","aqi",
                "temperature","humidity","wind_speed",
                "dist_to_road_m","dist_to_factory_m","dist_to_farmland_m"]
    X = df[features].fillna(0)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    df['predicted_source'] = le.inverse_transform(preds)
    df['confidence'] = probs.max(axis=1)
    return df

def plot_map(df, pollutant_filter):
    m = folium.Map(location=[20,0], zoom_start=2)
    colors = {
        "vehicular":"#e74c3c",
        "industrial":"#f39c12",
        "agricultural":"#27ae60",
        "natural":"#2980b9",
        "unknown":"#8e44ad"
    }

    for _, row in df.iterrows():
        radius = 5
        popup_text = f"{row.location}<br>{row.predicted_source} ({row.confidence:.2f})"
        
        if pollutant_filter != "None":
            value = row.get(pollutant_filter, 0) or 0
            radius = max(5, min(15, value/10))
            popup_text += f"<br>{pollutant_filter}: {value}"

        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=radius,
            color=colors.get(row.predicted_source,"#7f8c8d"),
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)
    
    legend_html = """
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 150px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; border-radius:6px;
     padding: 10px; color:black;">
     <b>Source Legend</b><br>
     <i style="background:#e74c3c;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Vehicular<br>
     <i style="background:#f39c12;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Industrial<br>
     <i style="background:#27ae60;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Agricultural<br>
     <i style="background:#2980b9;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Natural<br>
     <i style="background:#8e44ad;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Unknown
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=900, height=500)

def plot_aqi_map(df):
    m = folium.Map(location=[20,0], zoom_start=2)
    aqi_colors = {
        1: "#2ecc71",
        2: "#f1c40f",
        3: "#e67e22",
        4: "#e74c3c",
        5: "#8e44ad"
    }
    for _, row in df.iterrows():
        aqi = row.get("aqi", 0)
        color = aqi_colors.get(aqi, "#7f8c8d")
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{row.location}<br>AQI: {aqi}"
        ).add_to(m)
    legend_html = """
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 150px; 
     background-color: white; z-index:9999; font-size:14px;
     border:2px solid grey; border-radius:6px;
     padding: 10px; color:black;">
     <b>AQI Legend</b><br>
     <i style="background:#2ecc71;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Good<br>
     <i style="background:#f1c40f;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Fair<br>
     <i style="background:#e67e22;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Moderate<br>
     <i style="background:#e74c3c;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Poor<br>
     <i style="background:#8e44ad;width:12px;height:12px;display:inline-block;margin-right:5px;"></i> Very Poor
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=900, height=500)

# ----------------- Parallel Fetch -----------------
def fetch_data_parallel(df_loc):
    spatial_df = pd.read_csv("data/labeled_features.csv")[[ 
        "location_id","dist_to_road_m","dist_to_factory_m","dist_to_farmland_m"
    ]]
    df_loc = pd.merge(df_loc, spatial_df, on="location_id", how="left")
    df_loc.fillna(0, inplace=True)

    def fetch_for_row(row):
        p = fetch_pollution(row.latitude,row.longitude)
        w = fetch_weather(row.latitude,row.longitude)
        return {**row.to_dict(), **p, **w}

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_for_row, [row for _, row in df_loc.iterrows()]))
    
    return pd.DataFrame(results)

# ----------------- Streamlit -----------------
st.title("🌍 Global Real-Time Pollution Tracker")

# Sidebar
refresh_rate = st.sidebar.slider("Refresh every (seconds)", 10, 600, 60)
locations_limit = st.sidebar.slider("Locations to fetch", 50, 1000, 200)

# Filters
st.sidebar.subheader("Filters")
source_filter = st.sidebar.multiselect(
    "Select Pollution Source(s)",
    options=["vehicular","industrial","agricultural","natural","unknown"],
    default=["vehicular","industrial","agricultural","natural","unknown"]
)
pollutant_filter = st.sidebar.selectbox(
    "Highlight by pollutant",
    options=POLLUTANTS + ["aqi", "None"],
    index=len(POLLUTANTS)
)
# Map selection
map_choice = st.sidebar.radio(
    "Choose Map",
    options=["Source Map", "AQI Map", "Both"],
    index=2
)

# Auto-refresh
st_autorefresh(interval=refresh_rate*1000, key="pollution_counter")

st.info("Fetching live locations...")
df_loc = fetch_locations(limit=locations_limit)

if not df_loc.empty:
    df = fetch_data_parallel(df_loc)
    df_pred = predict_pollution(df)
    df_pred = df_pred[df_pred['predicted_source'].isin(source_filter)]

    st.subheader("Live Global Pollution Data")
    st.dataframe(df_pred[['location','latitude','longitude','pm2_5','pm10','no2','so2','co','o3','aqi','predicted_source','confidence']])

    if map_choice in ["Source Map", "Both"]:
        st.subheader("Global Source Map")
        plot_map(df_pred, pollutant_filter)

    if map_choice in ["AQI Map", "Both"]:
        st.subheader("Global AQI Map")
        plot_aqi_map(df_pred)

else:
    st.warning("No locations fetched.")
