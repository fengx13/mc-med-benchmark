#!/usr/bin/env python3
"""Stage 8: order columns (mirroring the MIMIC benchmark's naming convention
where the underlying concept matches), write outputs and dictionary.

Outcome block reduced to exactly 5 columns: outcome_sepsis, outcome_hf_acute,
outcome_stroke, outcome_cp_acs, outcome_cp_pe.
"""
import sys, time, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
sys.path.insert(0, str(ROOT / "scripts"))
from codemaps import CCI_CATS, ECI_CATS

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

m = pd.read_pickle(OUT / "_stage7.pkl").copy()
VIT = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
cols = set(m.columns)

BENCHMARK_OUTCOMES = ["outcome_sepsis", "outcome_hf_acute", "outcome_stroke",
                      "outcome_cp_acs", "outcome_cp_pe"]

ORDER = [c for c in (
    ["subject_id", "stay_id", "Visit_no", "Visits", "intime", "outtime",
     "age", "gender", "ethnicity", "race_group", "Ethnicity", "arrival_transport",
     "Roomed_time", "Dispo_time", "Admit_time",
     "ed_los_hours", "ed_los_hours_from_ts", "roomed_to_dispo_hours", "dispo_to_departure_hours",
     "Hosp_LOS", "Payor_class", "Admit_service", "ED_dispo", "DC_dispo",
     "cohort_chest_pain",
     "outcome_hospitalization", "outcome_admitted_inpatient", "outcome_icu_transfer",
     "outcome_inhospital_mortality", "outcome_critical",
     "first_vent_time", "time_to_invasive_vent_hours",
     "n_ed_30d", "n_ed_90d", "n_ed_365d",
     "next_ed_visit_time", "next_ed_visit_time_diff", "next_ed_visit_time_diff_days",
     "outcome_ed_revisit_3d", "Hours_to_next_visit", "Dispo_class_next_visit",
     "n_hosp_30d", "n_hosp_90d", "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d"]
    + [f"triage_{v}" for v in VIT] + [f"triage_{v}_missing" for v in VIT]
    + [f"triage_{v}_obs_first" for v in VIT] + [f"triage_{v}_obs_last" for v in VIT]
    + ["triage_map_chart_obs_first", "triage_map_chart_obs_last",
       "triage_pain_first", "triage_pain_last",
       "triage_gcs_first", "triage_gcs_last",
       "triage_acuity", "triage_acuity_text", "triage_MAP", "triage_MAP_missing",
       "n_vitals_imputed", "n_vitals_obs_ed", "max_lpm_o2", "on_supplemental_o2",
       "triage_sofa", "CC"]
    + [f"chiefcom_{c}" for c in ["chest_pain", "abdominal_pain", "headache",
       "shortness_of_breath", "back_pain", "cough", "nausea_vomiting", "fever_chills",
       "syncope", "dizziness"]]
    + [f"cci_{c}" for c in CCI_CATS] + [f"eci_{c}" for c in ECI_CATS]
    + ["n_pmh_codes_5y", "n_pmh_codes_any", "has_pmh_5y",
       "n_med", "n_medrecon",
       "score_CCI", "score_CART", "score_REMS", "score_NEWS", "score_NEWS2", "score_MEWS",
       "Dx_ICD9", "Dx_ICD10", "Dx_name", "n_primary_dx_codes"]
    + BENCHMARK_OUTCOMES
    + ["has_labs_ed", "n_labs_ed", "n_lab_assays_ed", "time_to_first_lab_min"]
    + sorted(c for c in cols if c.startswith("lab_"))
) if c in cols]

extra = [c for c in m.columns if c not in ORDER]
if extra:
    log(f"appending {len(extra)} unlisted columns: {extra}")
    ORDER += extra
m = m[ORDER]
log(f"final shape {m.shape}")

m.to_pickle(OUT / "master_benchmark5.pkl")
log("writing master_benchmark5.csv.gz")
m.to_csv(OUT / "master_benchmark5.csv.gz", index=False, compression="gzip")

log("writing column dictionary")
rows = []
for c in m.columns:
    s = m[c]; k = s.dtype.kind
    rows.append({"column": c, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
                "pct_missing": round(float(s.isna().mean()) * 100, 2),
                "min": s.min() if k in "iuf" else "",
                "median": round(float(s.median()), 3) if k in "iuf" and s.notna().any() else "",
                "max": s.max() if k in "iuf" else ""})
pd.DataFrame(rows).to_csv(OUT / "column_dictionary.csv", index=False)
log("done")
print(f"\nFinal dataset: {m.shape[0]:,} rows x {m.shape[1]} columns")
print("\nOutcome prevalence:")
for c in BENCHMARK_OUTCOMES:
    print(f"  {c:16s} n={m[c].sum():7,d}  rate={m[c].mean()*100:5.2f}%")
