#!/usr/bin/env python3
"""Stage 7: home-medication count (n_medrecon analog) and ED-administered
medication count (n_med analog).

meds.csv is patient-level (MRN, no CSN) with Entry_date/Start_date/End_date per
medication and an ambiguous Active flag not tied to any specific visit (the data
dictionary's "at the time of the visit" wording doesn't square with the table
having no visit key). Start_date is 66% null and End_date 52% null, so rather
than trust Active, a medication is counted as active for a given visit if
Entry_date <= Arrival_time <= End_date (End_date missing treated as ongoing).
This is a proxy for MIMIC's ed/medrecon.csv, which is a genuine per-ED-visit
reconciliation — not the same measurement, and documented as such.
"""
import time, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

m = pd.read_pickle(OUT / "_stage6.pkl")

log("loading meds.csv (home medications)")
me = pd.read_csv(ROOT / "data/meds.csv", usecols=["MRN", "Generic_name", "Entry_date", "End_date"],
                 parse_dates=["Entry_date", "End_date"])
for c in ["Entry_date", "End_date"]:
    me[c] = pd.to_datetime(me[c], utc=True).dt.tz_localize(None)

pairs = m[["subject_id", "stay_id", "intime"]].merge(me, left_on="subject_id", right_on="MRN", how="inner")
active = (pairs["Entry_date"] <= pairs["intime"]) & (pairs["End_date"].isna() | (pairs["End_date"] >= pairs["intime"]))
pairs = pairs.loc[active]
n_medrecon = pairs.groupby("stay_id")["Generic_name"].nunique()
m["n_medrecon"] = m["stay_id"].map(n_medrecon).fillna(0).astype("int16")

log("loading orders.csv for ED-administered medication count")
orders = pd.read_csv(ROOT / "data/orders.csv", usecols=["CSN", "Order_type", "Procedure_name"])
meds_ordered = orders[orders["Order_type"] == "Medications"]
n_med = meds_ordered.groupby("CSN")["Procedure_name"].nunique()
m["n_med"] = m["stay_id"].map(n_med).fillna(0).astype("int16")

m.to_pickle(OUT / "_stage7.pkl")
log(f"stage 7 complete -> {m.shape}")
print(f"n_medrecon: mean {m.n_medrecon.mean():.2f}, median {m.n_medrecon.median():.0f}")
print(f"n_med:      mean {m.n_med.mean():.2f}, median {m.n_med.median():.0f}")
