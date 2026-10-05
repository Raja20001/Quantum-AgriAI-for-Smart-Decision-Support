# -*- coding: utf-8 -*-
"""
Export optimized datasets into clean JSON for static web deployment
(GitHub Pages & Vercel)
"""
import os
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DATA_DIR = os.path.join(BASE_DIR, "docs", "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DATA_DIR, exist_ok=True)

# ── 1. CROP PROFILES & CENTROIDS ─────────────────────────────────────────────
CROP_PROFILES = {
    "Rice":        {"temp":(20,35),"humidity":(70,90),"rainfall":(150,300),"ph":(5.5,7.0), "sow":"June-July", "harvest":"November-December", "season":"Kharif", "days":"110-140", "emoji":"🌾"},
    "Wheat":       {"temp":(10,25),"humidity":(50,75),"rainfall":(75,150), "ph":(6.0,7.5), "sow":"October-November", "harvest":"March-April", "season":"Rabi", "days":"120-150", "emoji":"🌾"},
    "Maize":       {"temp":(18,35),"humidity":(50,75),"rainfall":(50,150), "ph":(5.8,7.0), "sow":"June-July", "harvest":"September-October", "season":"Kharif", "days":"90-110", "emoji":"🌽"},
    "ChickPea":    {"temp":(10,25),"humidity":(40,70),"rainfall":(30,100), "ph":(6.0,8.0), "sow":"October-November", "harvest":"February-March", "season":"Rabi", "days":"95-115", "emoji":"🌱"},
    "KidneyBeans": {"temp":(18,28),"humidity":(40,70),"rainfall":(40,100), "ph":(6.0,7.5), "sow":"March-May", "harvest":"July-September", "season":"Kharif", "days":"90-120", "emoji":"🫘"},
    "PigeonPeas":  {"temp":(18,35),"humidity":(40,70),"rainfall":(40,100), "ph":(5.5,7.0), "sow":"June-July", "harvest":"December-February", "season":"Kharif", "days":"150-180", "emoji":"🫘"},
    "MothBeans":   {"temp":(25,40),"humidity":(30,65),"rainfall":(25,75),  "ph":(6.0,8.0), "sow":"July-August", "harvest":"October-November", "season":"Kharif", "days":"75-90", "emoji":"🌱"},
    "MungBean":    {"temp":(20,40),"humidity":(50,80),"rainfall":(40,100), "ph":(6.0,7.5), "sow":"March-April", "harvest":"June-July", "season":"Zaid/Kharif", "days":"65-75", "emoji":"🌱"},
    "Blackgram":   {"temp":(25,40),"humidity":(55,80),"rainfall":(40,100), "ph":(5.5,7.0), "sow":"June-July", "harvest":"September-October", "season":"Kharif", "days":"70-85", "emoji":"🫘"},
    "Lentil":      {"temp":(10,25),"humidity":(40,65),"rainfall":(25,75),  "ph":(6.0,8.0), "sow":"October-November", "harvest":"March-April", "season":"Rabi", "days":"110-130", "emoji":"🥣"},
    "Sugarcane":   {"temp":(20,35),"humidity":(70,90),"rainfall":(100,250),"ph":(6.0,7.5), "sow":"October-March", "harvest":"October-March", "season":"Annual", "days":"300-365", "emoji":"🎋"},
    "Cotton":      {"temp":(20,40),"humidity":(40,75),"rainfall":(50,150), "ph":(5.8,8.0), "sow":"April-May", "harvest":"November-January", "season":"Kharif", "days":"150-180", "emoji":"☁️"},
    "Jute":        {"temp":(24,40),"humidity":(60,90),"rainfall":(100,250),"ph":(6.0,7.5), "sow":"March-May", "harvest":"July-August", "season":"Kharif", "days":"120-150", "emoji":"🌾"},
    "Groundnut":   {"temp":(22,35),"humidity":(50,75),"rainfall":(50,150), "ph":(6.0,7.5), "sow":"June-July", "harvest":"October-November", "season":"Kharif", "days":"100-120", "emoji":"🥜"},
    "Soybean":     {"temp":(20,35),"humidity":(50,75),"rainfall":(60,150), "ph":(6.0,7.5), "sow":"June-July", "harvest":"October-November", "season":"Kharif", "days":"90-110", "emoji":"🌱"},
    "Sunflower":   {"temp":(18,35),"humidity":(45,70),"rainfall":(50,120), "ph":(6.0,7.5), "sow":"January-February", "harvest":"April-May", "season":"Rabi", "days":"85-100", "emoji":"🌻"},
    "Mustard":     {"temp":(10,25),"humidity":(40,70),"rainfall":(40,100), "ph":(6.0,7.5), "sow":"September-October", "harvest":"February-March", "season":"Rabi", "days":"105-125", "emoji":"🌾"},
    "Turmeric":    {"temp":(20,35),"humidity":(70,90),"rainfall":(150,300),"ph":(5.5,7.0), "sow":"May-June", "harvest":"January-March", "season":"Kharif", "days":"210-270", "emoji":"🍠"},
    "Ginger":      {"temp":(18,30),"humidity":(70,90),"rainfall":(120,250),"ph":(5.5,6.5), "sow":"April-May", "harvest":"December-January", "season":"Kharif", "days":"210-240", "emoji":"🫚"},
    "Banana":      {"temp":(20,35),"humidity":(60,85),"rainfall":(100,200),"ph":(5.5,7.0), "sow":"June-July", "harvest":"Year-round", "season":"Annual", "days":"300-365", "emoji":"🍌"},
    "Mango":       {"temp":(24,40),"humidity":(40,75),"rainfall":(75,200), "ph":(5.5,7.5), "sow":"June-August", "harvest":"April-June", "season":"Perennial", "days":"365+", "emoji":"🥭"},
    "Coconut":     {"temp":(25,40),"humidity":(60,90),"rainfall":(100,250),"ph":(5.0,8.0), "sow":"May-June", "harvest":"Year-round", "season":"Perennial", "days":"365+", "emoji":"🥥"},
    "Grapes":      {"temp":(15,35),"humidity":(40,65),"rainfall":(25,75),  "ph":(5.5,7.0), "sow":"October-November", "harvest":"March-May", "season":"Perennial", "days":"120-150", "emoji":"🍇"},
    "Apple":       {"temp":(5,25), "humidity":(50,75),"rainfall":(50,150), "ph":(5.5,6.5), "sow":"January-February", "harvest":"August-October", "season":"Perennial", "days":"140-170", "emoji":"🍎"},
    "Tomato":      {"temp":(18,30),"humidity":(50,75),"rainfall":(50,120), "ph":(6.0,7.0), "sow":"June-July", "harvest":"September-November", "season":"Kharif/Rabi", "days":"75-90", "emoji":"🍅"},
    "Potato":      {"temp":(10,25),"humidity":(60,80),"rainfall":(100,200),"ph":(5.0,6.5), "sow":"October-November", "harvest":"January-March", "season":"Rabi", "days":"90-120", "emoji":"🥔"},
    "Onion":       {"temp":(15,30),"humidity":(50,75),"rainfall":(50,120), "ph":(6.0,7.5), "sow":"October-November", "harvest":"February-April", "season":"Rabi", "days":"100-140", "emoji":"🧅"},
}

MARKET_PRICE_INR = {
    "Rice":1850,"Wheat":1975,"Maize":1870,"ChickPea":5230,"KidneyBeans":3500,
    "PigeonPeas":6300,"MothBeans":3900,"MungBean":7755,"Blackgram":6950,
    "Lentil":5500,"Sugarcane":290,"Cotton":6080,"Jute":4500,"Groundnut":5850,
    "Soybean":3880,"Sunflower":5441,"Mustard":5450,"Turmeric":7000,"Ginger":5500,
    "Banana":1200,"Mango":3000,"Coconut":1500,"Grapes":4000,"Apple":5000,
    "Tomato":2500,"Potato":1000,"Onion":2000,
}

crop_csv = os.path.join(BASE_DIR, "datasets", "crop_recommendation_dataset.csv")
df_crop = pd.read_csv(crop_csv)
target_col = 'label' if 'label' in df_crop.columns else 'Crop'
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

crop_database = {}
for crop_name, grp in df_crop.groupby(target_col):
    prof = CROP_PROFILES.get(crop_name, {
        "temp":(15,35),"humidity":(40,80),"rainfall":(50,150),"ph":(5.5,7.5),
        "sow":"Varies","harvest":"Varies","season":"General","days":"90-120","emoji":"🌱"
    })
    means = {f: round(float(grp[f].mean()), 2) for f in features}
    stds  = {f: round(float(grp[f].std()) if grp[f].std() > 0.1 else 1.0, 2) for f in features}
    
    crop_database[crop_name] = {
        "name": crop_name,
        "emoji": prof.get("emoji", "🌱"),
        "price_inr_q": MARKET_PRICE_INR.get(crop_name, 2500),
        "ideal_temp": prof["temp"],
        "ideal_humidity": prof["humidity"],
        "ideal_rainfall": prof["rainfall"],
        "ideal_ph": prof["ph"],
        "sow": prof["sow"],
        "harvest": prof["harvest"],
        "season": prof["season"],
        "duration_days": prof["days"],
        "means": means,
        "stds": stds
    }

with open(os.path.join(DATA_DIR, "crops.json"), "w", encoding="utf-8") as f:
    json.dump(crop_database, f, indent=2)
with open(os.path.join(DOCS_DATA_DIR, "crops.json"), "w", encoding="utf-8") as f:
    json.dump(crop_database, f, indent=2)
print(f"Exported {len(crop_database)} crops to crops.json")

# ── 2. INDIA AGRICULTURE GEO DATA ───────────────────────────────────────────
india_csv = os.path.join(BASE_DIR, "datasets", "india_agriculture_dataset.csv")
df_india = pd.read_csv(india_csv)
states = sorted(df_india['State'].dropna().unique().tolist())
india_summary = {}

for s in states:
    sdf = df_india[df_india['State'] == s]
    districts = sorted(sdf['District'].dropna().unique().tolist())
    top_crops = sdf['Crop'].value_counts().head(8).to_dict()
    avg_yield = sdf.groupby('Crop')['Yield_kg_ha'].mean().round(1).to_dict()
    avg_price = sdf.groupby('Crop')['Price_INR_q'].mean().round(0).to_dict()
    seasons = sdf['Season'].dropna().unique().tolist() if 'Season' in sdf.columns else ["Kharif", "Rabi"]
    india_summary[s] = {
        "state": s,
        "records": int(len(sdf)),
        "district_count": len(districts),
        "districts": districts,
        "top_crops": top_crops,
        "avg_yield": avg_yield,
        "avg_price": avg_price,
        "seasons": seasons
    }

with open(os.path.join(DATA_DIR, "india_agri.json"), "w", encoding="utf-8") as f:
    json.dump(india_summary, f, indent=2)
with open(os.path.join(DOCS_DATA_DIR, "india_agri.json"), "w", encoding="utf-8") as f:
    json.dump(india_summary, f, indent=2)
print(f"Exported {len(india_summary)} Indian states to india_agri.json")

# ── 3. MARKET FORECASTING & TREND DATA ──────────────────────────────────────
market_csv = os.path.join(BASE_DIR, "datasets", "market_price_dataset.csv")
df_m = pd.read_csv(market_csv)
market_crops = df_m['Crop'].unique().tolist()
market_summary = {}

for c in market_crops:
    cdf = df_m[df_m['Crop'] == c]
    mean_p = round(float(cdf['Price_INR_q'].mean()), 1)
    std_p  = round(float(cdf['Price_INR_q'].std()), 1)
    msp_val = round(float(cdf['MSP_INR_q'].mean()), 1) if 'MSP_INR_q' in cdf.columns else round(mean_p * 0.85, 1)
    min_p  = round(float(cdf['Price_INR_q'].min()), 1)
    max_p  = round(float(cdf['Price_INR_q'].max()), 1)
    
    # 12-month synthetic smoothed price trend based on historical mean & std
    np.random.seed(abs(hash(c)) % 10000)
    base_wave = np.sin(np.linspace(0, 2 * np.pi, 12)) * (std_p * 0.7)
    noise = np.random.normal(0, std_p * 0.2, 12)
    trend = [round(float(mean_p + base_wave[i] + noise[i]), 1) for i in range(12)]
    
    market_summary[c] = {
        "crop": c,
        "emoji": CROP_PROFILES.get(c, {}).get("emoji", "🌾"),
        "avg_price": mean_p,
        "std_price": std_p,
        "msp": msp_val,
        "min_price": min_p,
        "max_price": max_p,
        "upper_band": round(mean_p + 1.96 * std_p, 1),
        "lower_band": round(max(0, mean_p - 1.96 * std_p), 1),
        "volatility_pct": round((std_p / mean_p) * 100, 1) if mean_p > 0 else 0,
        "trend_12m": trend,
        "signal": "SELL IMMEDIATELY" if mean_p > msp_val * 1.25 else ("STORE / HOLD" if mean_p < msp_val * 1.05 else "NORMAL TRADING")
    }

with open(os.path.join(DATA_DIR, "market.json"), "w", encoding="utf-8") as f:
    json.dump(market_summary, f, indent=2)
with open(os.path.join(DOCS_DATA_DIR, "market.json"), "w", encoding="utf-8") as f:
    json.dump(market_summary, f, indent=2)
print(f"Exported {len(market_summary)} crops market data to market.json")
print("Data export complete!")
