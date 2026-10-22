#!/usr/bin/env python3
"""Stage 1: MC-MED cohort spine, outcomes, and utilization history (one row per CSN).

MC-MED has no separate ED-stay/admission/patient tables like MIMIC-IV — visits.csv
already carries demographics, triage, disposition and admission fields in one row per
ED visit (CSN). MRN is the patient key, analogous to subject_id.
"""
import sys, time, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
OUT.mkdir(exist_ok=True)
SEC = 10_000_000_000

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)


def epoch(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[s]").astype("int64")


def _key(codes, secs):
    return codes.astype("int64") * SEC + secs


def prior_counts(base_code, base_t, ev_code, ev_t, windows):
    order = np.lexsort((ev_t, ev_code))
    ev_keys = _key(ev_code[order], ev_t[order])
    now = _key(base_code, base_t)
    upto = np.searchsorted(ev_keys, now, "left")
    out = {}
    for w in windows:
        back = _key(base_code, base_t - np.int64(w * 86400))
        out[w] = upto - np.searchsorted(ev_keys, back, "left")
    return out


def next_event(base_code, base_t, ev_code, ev_t):
    order = np.lexsort((ev_t, ev_code))
    ev_code_s, ev_t_s = ev_code[order], ev_t[order]
    ev_keys = _key(ev_code_s, ev_t_s)
    pos = np.searchsorted(ev_keys, _key(base_code, base_t), "right")
    ok = (pos < len(ev_keys)) & (ev_code_s[np.clip(pos, 0, len(ev_keys) - 1)] == base_code)
    res = np.full(len(base_t), np.iinfo("int64").min, dtype="int64")
    res[ok] = ev_t_s[pos[ok]]
    return pd.to_datetime(np.where(ok, res, np.nan), unit="s")


log("loading visits.csv")
v = pd.read_csv(ROOT / "data/visits.csv",
                parse_dates=["Arrival_time", "Roomed_time", "Dispo_time", "Admit_time", "Departure_time"])
v["Arrival_time"] = v["Arrival_time"].dt.tz_localize(None)
v["Roomed_time"] = v["Roomed_time"].dt.tz_localize(None)
v["Dispo_time"] = v["Dispo_time"].dt.tz_localize(None)
v["Admit_time"] = v["Admit_time"].dt.tz_localize(None)
v["Departure_time"] = v["Departure_time"].dt.tz_localize(None)
log(f"{len(v):,} visits, {v.MRN.nunique():,} patients")

m = v.rename(columns={"MRN": "subject_id", "CSN": "stay_id", "Age": "age", "Gender": "gender",
                      "Race": "ethnicity", "Means_of_arrival": "arrival_transport",
                      "Arrival_time": "intime", "Departure_time": "outtime"})

# ---------------------------------------------------------------- race_group
def race_group(x):
    if not isinstance(x, str):
        return "Other"
    xl = x.lower()
    if "white" in xl:
        return "White"
    if "black" in xl or "african" in xl:
        return "Black/African American"
    if "asian" in xl:
        return "Asian"
    return "Other"


m["race_group"] = m["ethnicity"].map(race_group).astype("string")
# MC-MED separates Hispanic ethnicity from race (US OMB convention); fold Hispanic/Latino
# into race_group the same way the MIMIC build did, so the two datasets' race_group values
# are comparable.
m.loc[m["Ethnicity"] == "Hispanic/Latino", "race_group"] = "Hispanic/Latino"

# ---------------------------------------------------------------- LOS, timing
m["ed_los_hours"] = m["ED_LOS"]
m["ed_los_hours_from_ts"] = (m["outtime"] - m["intime"]).dt.total_seconds() / 3600
m["roomed_to_dispo_hours"] = (m["Dispo_time"] - m["Roomed_time"]).dt.total_seconds() / 3600
m["dispo_to_departure_hours"] = (m["outtime"] - m["Dispo_time"]).dt.total_seconds() / 3600

# ---------------------------------------------------------------- outcomes
log("deriving outcomes")
# ED_dispo has 4 levels: Discharge / Observation / Inpatient / ICU. There is no
# 'Expired' ED_dispo level and no ED-bay-death signal at all (verified: of the 800
# visits whose DC_dispo is 'Expired', 799 already carry ED_dispo in {Inpatient,
# ICU, Observation} — death occurred after leaving the ED, not in it; 1 residual
# 'Discharge' case is noise). Unlike MIMIC-IV-ED, MC-MED cannot support an
# `ed_death` column.
m["outcome_hospitalization"] = (m["ED_dispo"] != "Discharge").astype("int8")
m["outcome_admitted_inpatient"] = m["ED_dispo"].isin(["Inpatient", "ICU"]).astype("int8")
m["outcome_icu_transfer"] = (m["ED_dispo"] == "ICU").astype("int8")
m["outcome_inhospital_mortality"] = (m["DC_dispo"] == "Expired").astype("int8")
m["outcome_critical"] = ((m["outcome_icu_transfer"] == 1)
                         | (m["outcome_inhospital_mortality"] == 1)).astype("int8")

# ---------------------------------------------------------------- utilization
log("computing prior-utilization counts and next-visit timing")
b_code = m["subject_id"].to_numpy()
b_t = epoch(m["intime"])
assert b_t.min() > 0

ed_ev = (b_code, b_t)
hosp_ev = (m.loc[m.outcome_hospitalization == 1, "subject_id"].to_numpy(),
           epoch(m.loc[m.outcome_hospitalization == 1, "intime"]))
icu_ev = (m.loc[m.outcome_icu_transfer == 1, "subject_id"].to_numpy(),
          epoch(m.loc[m.outcome_icu_transfer == 1, "intime"]))

for label, ev in [("ed", ed_ev), ("hosp", hosp_ev), ("icu", icu_ev)]:
    cnt = prior_counts(b_code, b_t, ev[0], ev[1], [30, 90, 365])
    for w, val in cnt.items():
        m[f"n_{label}_{w}d"] = val.astype("int32")

nxt = next_event(b_code, b_t, ed_ev[0], ed_ev[1])
m["next_ed_visit_time"] = nxt
m["next_ed_visit_time_diff"] = m["next_ed_visit_time"] - m["outtime"]
m["next_ed_visit_time_diff_days"] = m["next_ed_visit_time_diff"].dt.total_seconds() / 86400
m["outcome_ed_revisit_3d"] = ((m["next_ed_visit_time_diff_days"] >= 0)
                              & (m["next_ed_visit_time_diff_days"] <= 3)).astype("int8")

m.to_pickle(OUT / "_stage1.pkl")
log(f"stage 1 complete -> {m.shape}")
for c in ["outcome_hospitalization", "outcome_admitted_inpatient", "outcome_icu_transfer",
          "outcome_inhospital_mortality", "outcome_critical", "outcome_ed_revisit_3d"]:
    print(f"  {c:28s} n={m[c].sum():7,d}  rate={m[c].mean()*100:5.2f}%")

# sanity check vs the file's own Hours_to_next_visit
cmp = m.dropna(subset=["next_ed_visit_time_diff_days"])
hrs_from_file = m.loc[cmp.index, "Hours_to_next_visit"] / 24
diff = (cmp["next_ed_visit_time_diff_days"] - hrs_from_file).abs()
print(f"  cross-check vs Hours_to_next_visit: median abs diff = {diff.median():.4f} days "
      f"(n={len(cmp):,})")
