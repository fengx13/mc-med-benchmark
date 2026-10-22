#!/usr/bin/env python3
"""Stage 2: serial ED vitals from numerics-001.csv, cleaning, and median imputation.

numerics-001.csv (2.6 GB, 47.8M rows) carries two families of readings per visit:
Source=='Chart' (nursing-documented spot checks, sparse — the direct analog of
MIMIC's ed/vitalsign.csv) and Source=='Monitor' (continuous bedside-monitor output,
minute-level, including two HRV channels). Only Chart is used for the *_obs_first /
*_obs_last columns, to stay a fair analog of the MIMIC pipeline; Monitor volume is
kept as a coverage count only. Nothing here touches the raw continuous streams —
they remain in numerics-001.csv for future work.
"""
import sys, time, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
MIMIC_SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(MIMIC_SCRIPTS))
import scores as sc

_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

m = pd.read_pickle(OUT / "_stage1.pkl")

# ---------------------------------------------------------------- triage vitals
# visits.csv already provides one triage-time reading per vital; MC-MED's own
# columns are already named Triage_*, unlike MIMIC's ed/triage.csv. Temperature
# here is native Celsius (data dictionary states this explicitly), unlike MIMIC-IV-ED
# which stores triage temperature in Fahrenheit — kept as Celsius throughout this
# build rather than reproducing MIMIC's unit choice.
RANGES = {"triage_temperature": (21.0, 46.0), "triage_heartrate": (20.0, 300.0),
          "triage_resprate": (2.0, 80.0), "triage_o2sat": (30.0, 100.0),
          "triage_sbp": (30.0, 350.0), "triage_dbp": (10.0, 250.0)}
m = m.rename(columns={"Triage_Temp": "triage_temperature", "Triage_HR": "triage_heartrate",
                      "Triage_RR": "triage_resprate", "Triage_SpO2": "triage_o2sat",
                      "Triage_SBP": "triage_sbp", "Triage_DBP": "triage_dbp"})
for c, (lo, hi) in RANGES.items():
    v = pd.to_numeric(m[c], errors="coerce")
    m[c] = v.where((v >= lo) & (v <= hi))

# Triage_acuity ships as text ("3-Urgent"); keep both the numeric ESI level (for
# scoring/validation) and the original text.
m["triage_acuity_text"] = m["Triage_acuity"]
m["triage_acuity"] = pd.to_numeric(m["Triage_acuity"].astype("string").str.extract(r"^(\d)")[0],
                                   errors="coerce")

# ---------------------------------------------------------------- serial vitals
log("scanning numerics-001.csv for Chart-source serial vitals")
VIT_MEASURES = ["HR", "RR", "SpO2", "SBP", "DBP", "MAP", "Temp", "Pain"]
kept = []
n_seen = n_chart = n_lpm = 0
for i, chunk in enumerate(pd.read_csv(ROOT / "numerics-001.csv", chunksize=4_000_000)):
    n_seen += len(chunk)
    chart = chunk[(chunk["Source"] == "Chart") & (chunk["Measure"].isin(VIT_MEASURES))]
    n_chart += len(chart)
    lpm = chunk[chunk["Measure"] == "LPM_O2"]
    n_lpm += len(lpm)
    if len(chart) or len(lpm):
        kept.append(pd.concat([chart, lpm]))
    if (i + 1) % 3 == 0:
        log(f"  chunk {i+1}: {n_seen:,} rows scanned, {n_chart:,} chart-vital rows kept, "
            f"{n_lpm:,} LPM_O2 rows kept")
num = pd.concat(kept, ignore_index=True)
log(f"DONE: {n_seen:,} numerics rows scanned -> {len(num):,} kept")
num["Time"] = pd.to_datetime(num["Time"], utc=True).dt.tz_localize(None)
num["Value"] = pd.to_numeric(num["Value"], errors="coerce")
num = num.dropna(subset=["Value"]).sort_values(["CSN", "Measure", "Time"])

VIT_RANGES = {"HR": (20, 300), "RR": (2, 80), "SpO2": (30, 100), "SBP": (30, 350),
             "DBP": (10, 250), "Temp": (70, 115), "Pain": (0, 10)}
for meas, (lo, hi) in VIT_RANGES.items():
    sel = num["Measure"] == meas
    num.loc[sel & ~num["Value"].between(lo, hi), "Value"] = np.nan
num = num.dropna(subset=["Value"])

chart = num[num["Measure"] != "LPM_O2"]
g = chart.groupby(["CSN", "Measure"])["Value"]
first = g.first().unstack()
last = g.last().unstack()
n_obs = g.size().groupby("CSN").sum() if False else chart.groupby("CSN").size()
RENAME = {"HR": "heartrate", "RR": "resprate", "SpO2": "o2sat", "SBP": "sbp",
         "DBP": "dbp", "MAP": "map_chart", "Temp": "temperature", "Pain": "pain"}
first = first.rename(columns=RENAME).add_suffix("_obs_first")
last = last.rename(columns=RENAME).add_suffix("_obs_last")
# Temp in numerics.csv is Fahrenheit (data dictionary + verified against
# Triage_Temp: e.g. 97.7F row == 36.5C triage row for the same visit) — convert to
# Celsius so every temperature column in this dataset shares one unit.
for df in (first, last):
    tc = [c for c in df.columns if c.startswith("temperature")]
    for c in tc:
        df[c] = sc.f_to_c(df[c])
first.columns = [f"triage_{c}" for c in first.columns]
last.columns = [f"triage_{c}" for c in last.columns]
# Pain follows the MIMIC-build naming convention (triage_pain_first/_last, no
# "_obs_" infix) since, unlike the other vitals, MC-MED has no separate
# single-reading "Triage_Pain" column to disambiguate from.
first = first.rename(columns={"triage_pain_obs_first": "triage_pain_first"})
last = last.rename(columns={"triage_pain_obs_last": "triage_pain_last"})

m = m.merge(first, on="stay_id" if False else None, left_on="stay_id", right_index=True, how="left")
m = m.merge(last, left_on="stay_id", right_index=True, how="left")
m["n_vitals_obs_ed"] = m["stay_id"].map(n_obs).fillna(0).astype("int32")

lpm = num[num["Measure"] == "LPM_O2"] if "LPM_O2" in num["Measure"].unique() else num.iloc[0:0]
if len(lpm) == 0:
    lpm = pd.concat(kept, ignore_index=True)
    lpm = lpm[lpm["Measure"] == "LPM_O2"]
    lpm["Value"] = pd.to_numeric(lpm["Value"], errors="coerce")
max_lpm = lpm.groupby("CSN")["Value"].max()
m["max_lpm_o2"] = m["stay_id"].map(max_lpm)
m["on_supplemental_o2"] = (m["max_lpm_o2"].fillna(0) > 0).astype("int8")

m["triage_MAP"] = (m["triage_sbp"] + 2 * m["triage_dbp"]) / 3
# No GCS/AVPU anywhere in MC-MED either (Measure list has 12 entries — HR, RR,
# SpO2, SBP, DBP, MAP, Temp, Perf, Pain, LPM_O2, 1min_HRV, 5min_HRV — none of
# them a consciousness level), same structural gap as MIMIC-IV-ED.
m["triage_gcs_first"] = np.nan
m["triage_gcs_last"] = np.nan

# ---------------------------------------------------------------- imputation
log("median-imputing triage vitals")
VCOLS = ["triage_temperature", "triage_heartrate", "triage_resprate", "triage_o2sat",
        "triage_sbp", "triage_dbp"]
medians = {}
for c in VCOLS:
    med = float(m[c].median())
    medians[c] = med
    m[f"{c}_missing"] = m[c].isna().astype("int8")
    m[c] = m[c].fillna(med)
m["n_vitals_imputed"] = m[[f"{c}_missing" for c in VCOLS]].sum(axis=1).astype("int8")
m["triage_MAP_missing"] = ((m["triage_sbp_missing"] == 1) | (m["triage_dbp_missing"] == 1)).astype("int8")
m["triage_MAP"] = (m["triage_sbp"] + 2 * m["triage_dbp"]) / 3
# triage_acuity left un-imputed on purpose (same rationale as the MIMIC build):
# its missingness may itself be informative, and imputing it would erase that.
import json
(OUT / "imputation_medians.json").write_text(json.dumps(medians, indent=1))

log("computing severity scores")
ALERT_GCS = pd.Series(15.0, index=m.index)
m["score_CART"] = sc.score_cart(m["triage_resprate"], m["triage_heartrate"], m["triage_dbp"], m["age"])
m["score_REMS"] = sc.score_rems(m["age"], m["triage_MAP"], m["triage_heartrate"],
                                m["triage_resprate"], m["triage_o2sat"], ALERT_GCS)
alert_series = pd.Series(1.0, index=m.index)
supp_o2 = m["on_supplemental_o2"].astype(float)
m["score_NEWS"] = sc.score_news(m["triage_resprate"], m["triage_o2sat"], m["triage_temperature"],
                                m["triage_sbp"], m["triage_heartrate"], alert_series,
                                pd.Series(0.0, index=m.index))
# Unlike the MIMIC build (no supplemental-O2 data -> NEWS2 identical to NEWS),
# MC-MED's LPM_O2 channel lets NEWS2 genuinely differ: its SpO2 scoring band
# shifts when the patient is on supplemental oxygen.
m["score_NEWS2"] = sc.score_news2(m["triage_resprate"], m["triage_o2sat"], m["triage_temperature"],
                                  m["triage_sbp"], m["triage_heartrate"], alert_series, supp_o2)
m["score_MEWS"] = sc.score_mews(m["triage_sbp"], m["triage_heartrate"], m["triage_resprate"],
                                m["triage_temperature"], pd.Series(0.0, index=m.index))
m["triage_sofa"] = sc.score_sofa_vitals(m["triage_MAP"], ALERT_GCS, m["triage_o2sat"])

m.to_pickle(OUT / "_stage2.pkl")
log(f"stage 2 complete -> {m.shape}")
print("\nserial-vitals coverage (Chart source):")
for c in ["triage_heartrate_obs_first", "triage_sbp_obs_first", "triage_temperature_obs_first",
         "triage_pain_first"]:
    print(f"  {c:32s} {m[c].notna().mean()*100:5.1f}%")
print(f"\non_supplemental_o2 rate: {m.on_supplemental_o2.mean()*100:.2f}%")
print(f"NEWS vs NEWS2 identical: {(m.score_NEWS==m.score_NEWS2).mean()*100:.1f}% "
      f"(differ where on_supplemental_o2==1: n={m.on_supplemental_o2.sum():,})")
print(f"vitals imputed: {(m.n_vitals_imputed>0).mean()*100:.1f}% of visits had >=1 imputed")
