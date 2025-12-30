import numpy as np, pandas as pd, pathlib

base = pathlib.Path(".")
base.mkdir(exist_ok=True)
(base/"data/profiles").mkdir(parents=True, exist_ok=True)

hours = np.arange(0, 24*12)  # 5-min resolution (288 points)
t = hours/12.0
rng = np.random.default_rng(42)

base_load = 50 + 20*np.sin(2*np.pi*(t-7)/24) + 5*rng.normal(size=len(t))
load_A = np.clip(base_load + 10*np.sin(2*np.pi*(t)/8), 10, None)
load_B = np.clip(base_load + 5*np.sin(2*np.pi*(t)/6+0.8) + 8, 10, None)
load_C = np.clip(base_load + 7*np.sin(2*np.pi*(t)/12+1.5) + 5, 10, None)
ev_chg = np.clip(15*np.exp(-((t-18)/3)**2) + 2*rng.normal(size=len(t)), 0, None)
pv = np.clip(80*np.exp(-((t-13)/4)**2) + 5*rng.normal(size=len(t)), 0, None)

price = 0.08 + 0.04*((t%24>=7)&(t%24<=16)) + 0.12*((t%24>=17)&(t%24<=21))
price = np.clip(price + 0.005*rng.normal(size=len(t)), 0.05, 0.30)

co2 = 0.45 - 0.20*np.exp(-((t-13)/4)**2) + 0.08*((t%24>=17)&(t%24<=22))
co2 = np.clip(co2 + 0.01*rng.normal(size=len(t)), 0.12, 0.7)

profiles = pd.DataFrame({
    "slot": np.arange(len(t)),
    "hour_of_day": t%24,
    "load_A_kw": load_A.round(3),
    "load_B_kw": load_B.round(3),
    "load_C_kw": load_C.round(3),
    "ev_kw": ev_chg.round(3),
    "pv_kw": pv.round(3),
    "price_usd_per_kwh": price.round(4),
    "grid_co2_kg_per_kwh": co2.round(4)
})
profiles.to_csv(base/"data/profiles/smartcity_24h_profiles.csv", index=False)
print("saved:", base/"data/profiles/smartcity_24h_profiles.csv")
