#!/usr/bin/env python3
"""Stage 5: the 5 benchmark outcomes (ACS, sepsis, AHF, stroke, PE), from
visits.csv's Dx_ICD9/Dx_ICD10, plus ventilation timing (kept as a predictor).

CRITICAL CAVEAT, load-bearing for everything in this stage: Dx_ICD9/Dx_ICD10 is
the visit's PRIMARY diagnosis only (occasionally a short comma-separated list
when multiple diagnoses are equally primary — 4.2% of visits), not a full coded
diagnosis list the way MIMIC's diagnoses_icd + ed/diagnosis are. A patient who
ruled in for a secondary condition captured nowhere in the primary-diagnosis
field will be a false negative here. Every outcome rate computed in this stage
should be read as a FLOOR, not a true prevalence, and is not comparable in
absolute terms to the MIMIC benchmark's numbers for the same outcome.

ACS and PE use the same cp_codes.py PREFIX-match definitions as the MIMIC
benchmark (outcome_cp_acs / outcome_cp_pe, same column names, same codes).
Sepsis, AHF and stroke use outcome_codes_exact.py EXACT match.
"""
import sys, time, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
sys.path.insert(0, str(ROOT / "scripts"))
from cp_codes import CP_9, CP_10
from outcome_codes_exact import EXACT_OUTCOMES

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

m = pd.read_pickle(OUT / "_stage4.pkl").copy()

log("splitting Dx_ICD9 / Dx_ICD10 into per-visit code lists")
def split_codes(s):
    if not isinstance(s, str):
        return []
    return [c.strip().upper().replace(".", "") for c in s.split(",") if c.strip()]

codes9 = m["Dx_ICD9"].map(split_codes)
codes10 = m["Dx_ICD10"].map(split_codes)
m["n_primary_dx_codes"] = codes9.map(len) + codes10.map(len)

# ---------------------------------------------------------------- sepsis / AHF / stroke
log("flagging sepsis / AHF / stroke (exact match against the primary diagnosis)")
for name, (label, c9, c10) in EXACT_OUTCOMES.items():
    s9, s10 = set(c9), set(c10)
    hit = np.array([bool(set(a) & s9) or bool(set(b) & s10)
                    for a, b in zip(codes9, codes10)])
    m[f"outcome_{name}"] = hit.astype("int8")

# ---------------------------------------------------------------- ACS / PE (prefix)
log("flagging ACS / PE (prefix match, cp_codes.py)")
for k in ("acs", "pe"):
    p9 = tuple(CP_9[k]); p10 = tuple(CP_10[k])
    hit = np.array([any(c.startswith(p9) for c in a) or any(c.startswith(p10) for c in b)
                    for a, b in zip(codes9, codes10)])
    m[f"outcome_cp_{k}"] = hit.astype("int8")

# ---------------------------------------------------------------- ventilation timing
log("timing invasive ventilation / intubation from orders.csv (predictor only; "
    "too sparse in MC-MED for a time-resolved outcome -- see README)")
orders = pd.read_csv(ROOT / "data/orders.csv", usecols=["CSN", "Order_time", "Procedure_name"],
                     parse_dates=["Order_time"])
orders["Order_time"] = pd.to_datetime(orders["Order_time"], utc=True).dt.tz_localize(None)
vent = orders[orders["Procedure_name"].isin(["INTUBATION", "RESP - VENTILATOR SETTINGS"])]
first_vent = vent.groupby("CSN")["Order_time"].min().rename("first_vent_time")
m = m.merge(first_vent, left_on="stay_id", right_index=True, how="left")
m["time_to_invasive_vent_hours"] = (m["first_vent_time"] - m["outtime"]).dt.total_seconds() / 3600

m.to_pickle(OUT / "_stage5.pkl")
log(f"stage 5 complete -> {m.shape}")
for name in EXACT_OUTCOMES:
    c = f"outcome_{name}"
    print(f"  {c:16s} n={m[c].sum():7,d}  rate={m[c].mean()*100:5.2f}%")
for c in ["outcome_cp_acs", "outcome_cp_pe"]:
    print(f"  {c:16s} n={m[c].sum():7,d}  rate={m[c].mean()*100:5.2f}%")
