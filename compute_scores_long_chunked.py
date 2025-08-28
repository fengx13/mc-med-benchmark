
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_scores_long_chunked.py  (FINAL: includes ALL visits columns in output)

What’s new in this version?
- The final CSV now contains **every column from visits.csv**, plus triage_* columns and scores.
- Merge order: all visits columns first, then triage_* and score_* columns.

Other features (unchanged):
- Streams a long-format numerics file (CSN, Source, Measure, Value, Time) in chunks.
- Extracts earliest (triage) vitals per CSN.
- Computes qSOFA, MEWS, NEWS2. Optionally REMS (needs Age) and CCI (needs dx with cci_* flags).

Usage
-----
python3 compute_scores_long_chunked.py --visits visits.csv --numerics numerics.csv --out group3_scores_plus.csv
# optional CCI:
# python3 compute_scores_long_chunked.py --visits visits.csv --numerics numerics.csv --dx dx_history_dataset.csv --out group3_scores_plus.csv
"""

import argparse
import pandas as pd
import numpy as np


import re as _re_for_keys  # safe alias to avoid clashes
import pandas as _pd_for_keys

def _align_keys(vis: _pd_for_keys.DataFrame, out: _pd_for_keys.DataFrame):
    """
    Make merge keys consistent between `vis` and `out`.
    Tries common aliases for MRN/CSN-like identifiers.
    Returns: (out_aligned, keys)
    """
    candidates = {
        "MRN": ["MRN", "mrn", "subject_id", "patient_id", "patient", "patient_mrn"],
        "CSN": ["CSN", "csn", "encounter_id", "visit_id", "visit_occurrence_id", "hadm_id", "stay_id"],
    }
    rename_map = {}
    # If vis uses MRN/CSN, rename out's aliases to match
    for std_name, alts in candidates.items():
        if std_name in vis.columns:
            for alt in alts:
                if alt in out.columns and alt != std_name:
                    rename_map[alt] = std_name
                    break
    if rename_map:
        out = out.rename(columns=rename_map)

    # Build merge keys from intersecting standard names
    keys = [k for k in ["MRN", "CSN"] if (k in vis.columns and k in out.columns)]

    # Fallback: pick likely ID columns from the intersection
    if not keys:
        inter = list(set(vis.columns) & set(out.columns))
        likely = [c for c in inter if _re_for_keys.search(r"(mrn|csn|subject|patient|encounter|visit|hadm|stay)", c, _re_for_keys.I)]
        if not likely:
            raise KeyError(
                "No common key columns between visits and computed output. "
                "Ensure identifiers align (e.g., MRN/CSN or subject_id/encounter_id)."
            )
        keys = likely
    return out, keys


# ---------- helpers (same as last clean version) ----------

def normalize_measure(x: str):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("hr","heart rate","pulse"): return "hr"
    if s in ("rr","resp","resp rate","respiratory rate"): return "rr"
    if s in ("spo2","sp02","sao2","oxygen sat","o2 sat","o2sat","o2 saturation"): return "spo2"
    if s in ("sbp","systolic","nibp systolic","bp_systolic"): return "sbp"
    if s in ("dbp","diastolic","nibp diastolic","bp_diastolic"): return "dbp"
    if s in ("map","mean arterial pressure","mean art pressure","nibp map","bp_map"): return "map"
    if s in ("temp","temperature","body temp","temp c","tempc","temp f","tempf"): return "temp"
    if s in ("gcs","glasgow","glasgow coma scale"): return "gcs"
    if "temp" in s: return "temp"
    if "spo2" in s or ("o2" in s and "sat" in s): return "spo2"
    if "systolic" in s: return "sbp"
    if "diastolic" in s: return "dbp"
    if ("map" in s and "nibp" in s) or s == "map": return "map"
    if "resp" in s: return "rr"
    return None

def f_to_c(x):
    return (x - 32.0) * 5.0/9.0

def estimate_map(sbp, dbp):
    if pd.notna(sbp) and pd.notna(dbp):
        return dbp + (sbp - dbp)/3.0
    return np.nan

def score_qsofa(rr, sbp, gcs):
    pts = 0
    if pd.notna(rr) and rr >= 22: pts += 1
    if pd.notna(sbp) and sbp <= 100: pts += 1
    if pd.notna(gcs) and gcs < 15: pts += 1
    return np.nan if (pd.isna(rr) and pd.isna(sbp) and pd.isna(gcs)) else pts

def score_mews(rr, hr, sbp, temp_c):
    pts = 0; miss = True
    if pd.notna(rr):
        miss=False
        if rr <= 8: pts += 3
        elif 9 <= rr <= 14: pts += 0
        elif 15 <= rr <= 20: pts += 1
        elif 21 <= rr <= 29: pts += 2
        elif rr >= 30: pts += 3
    if pd.notna(hr):
        miss=False
        if hr <= 40: pts += 2
        elif 41 <= hr <= 50: pts += 1
        elif 51 <= hr <= 100: pts += 0
        elif 101 <= hr <= 110: pts += 1
        elif 111 <= hr <= 129: pts += 2
        elif hr >= 130: pts += 3
    if pd.notna(sbp):
        miss=False
        if sbp <= 70: pts += 3
        elif 71 <= sbp <= 80: pts += 2
        elif 81 <= sbp <= 100: pts += 1
        elif 101 <= sbp <= 199: pts += 0
        elif sbp >= 200: pts += 2
    if pd.notna(temp_c):
        miss=False
        if temp_c <= 35.0: pts += 2
        elif 35.1 <= temp_c <= 38.4: pts += 0
        elif temp_c >= 38.5: pts += 2
    return np.nan if miss else pts

def score_news2(rr, spo2, temp_c, sbp, hr, avpu=None):
    pts = 0; miss=True
    if pd.notna(rr):
        miss=False
        if rr <= 8: pts += 3
        elif 9 <= rr <= 11: pts += 1
        elif 12 <= rr <= 20: pts += 0
        elif 21 <= rr <= 24: pts += 2
        elif rr >= 25: pts += 3
    if pd.notna(spo2):
        miss=False
        if spo2 <= 91: pts += 3
        elif 92 <= spo2 <= 93: pts += 2
        elif 94 <= spo2 <= 95: pts += 1
        elif spo2 >= 96: pts += 0
    if pd.notna(temp_c):
        miss=False
        if temp_c <= 35.0: pts += 3
        elif 35.1 <= temp_c <= 36.0: pts += 1
        elif 36.1 <= temp_c <= 38.0: pts += 0
        elif 38.1 <= temp_c <= 39.0: pts += 1
        elif temp_c >= 39.1: pts += 2
    if pd.notna(sbp):
        miss=False
        if sbp <= 90: pts += 3
        elif 91 <= sbp <= 100: pts += 2
        elif 101 <= sbp <= 110: pts += 1
        elif 111 <= sbp <= 219: pts += 0
        elif sbp >= 220: pts += 3
    if pd.notna(hr):
        miss=False
        if hr <= 40: pts += 3
        elif 41 <= hr <= 50: pts += 1
        elif 51 <= hr <= 90: pts += 0
        elif 91 <= hr <= 110: pts += 1
        elif 111 <= hr <= 130: pts += 2
        elif hr >= 131: pts += 3
    if avpu is not None and isinstance(avpu, str):
        miss=False
        avpu_l = avpu.strip().lower()[:1]
        pts += 0 if avpu_l=='a' else 3
    return np.nan if miss else pts

def rems_points_age(age):
    if pd.isna(age): return np.nan
    age = float(age)
    if age < 45: return 0
    if age <= 54: return 2
    if age <= 64: return 3
    if age <= 74: return 5
    return 6

def rems_points_map(mapv):
    if pd.isna(mapv): return np.nan
    x = float(mapv)
    if x > 159: return 4
    if 130 <= x <= 159: return 3
    if 110 <= x <= 129: return 2
    if 70 <= x <= 109: return 0
    if 50 <= x <= 69: return 2
    if x < 50: return 4
    return np.nan

def rems_points_hr(hr):
    if pd.isna(hr): return np.nan
    x = float(hr)
    if x > 179: return 4
    if 140 <= x <= 179: return 3
    if 110 <= x <= 139: return 2
    if 70 <= x <= 109: return 0
    if 55 <= x <= 69: return 2
    if 40 <= x <= 54: return 3
    if x < 40: return 4
    return np.nan

def rems_points_rr(rr):
    if pd.isna(rr): return np.nan
    x = float(rr)
    if x > 49: return 4
    if 35 <= x <= 49: return 3
    if 25 <= x <= 34: return 2
    if 12 <= x <= 24: return 0
    if 10 <= x <= 11: return 1
    if 6 <= x <= 9: return 2
    if x < 6: return 4
    return np.nan

def rems_points_spo2(sp):
    if pd.isna(sp): return np.nan
    x = float(sp)
    if x > 89: return 0
    if 86 <= x <= 89: return 2
    if 75 <= x <= 85: return 3
    if x < 75: return 4
    return np.nan

def rems_points_gcs(g):
    if pd.isna(g): return np.nan
    x = float(g)
    if x > 13: return 0
    if 11 <= x <= 13: return 1
    if 8 <= x <= 10: return 2
    if 5 <= x <= 7: return 3
    if x < 5: return 4
    return np.nan

def compute_age(vis):
    v = vis.copy()
    for c in v.columns:
        if c.lower()=="age":
            v = v.rename(columns={c:"Age"})
            return v
    dob_cols = [c for c in v.columns if c.lower() in ("dob","birth_date","birthdate","date_of_birth")]
    t_cols = [c for c in v.columns if c.lower() in ("visit_time","arrival_time","charttime","time","timestamp")]
    if dob_cols and t_cols:
        v[dob_cols[0]] = pd.to_datetime(v[dob_cols[0]], errors="coerce")
        v[t_cols[0]] = pd.to_datetime(v[t_cols[0]], errors="coerce")
        v["Age"] = ((v[t_cols[0]] - v[dob_cols[0]]).dt.days/365.25).astype("float")
        return v
    v["Age"] = np.nan
    return v

def compute_cci_from_flags(dx):
    weights = {
        "cci_MI": 1, "cci_CHF": 1, "cci_PVD": 1, "cci_Stroke": 1, "cci_Dementia": 1,
        "cci_Pulmonary": 1, "cci_Rheumatic": 1, "cci_PUD": 1, "cci_Liver1": 1, "cci_DM1": 1,
        "cci_DM2": 2, "cci_Paralysis": 2, "cci_Renal": 2, "cci_Cancer1": 2,
        "cci_Liver2": 3, "cci_Cancer2": 6, "cci_HIV": 6
    }
    for k in list(weights.keys()):
        if k not in dx.columns:
            dx[k] = 0
    s = None
    for col, w in weights.items():
        add = dx[col].fillna(0).astype(float) * w
        s = add if s is None else s + add
    return s

def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--visits", required=True)
    ap.add_argument("--numerics", required=True)
    ap.add_argument("--out", default="group3_scores_plus.csv")
    ap.add_argument("--dx", default=None, help="dx_history_dataset.csv with cci_* flags (optional)")
    ap.add_argument("--chunksize", type=int, default=500000)
    args = ap.parse_args()

    # ----- visits & keys -----
    vis = pd.read_csv(args.visits)
    vcols = list(vis.columns)
    def pick(colnames, cands):
        low = {c.lower(): c for c in colnames}
        for x in cands:
            if x in low: return low[x]
        return None
    mrn_v = pick(vcols, ["mrn","patient_id","patient_mrn","patientid"])
    csn_v = pick(vcols, ["csn","encounter_id","visit_id","encounter","encounterid","visitid"])
    if csn_v is None:
        raise SystemExit("ERROR: visits.csv must contain CSN/encounter key.")
    vis = vis.rename(columns={mrn_v:"MRN"} if mrn_v else {}).rename(columns={csn_v:"CSN"})
    vis_age = compute_age(vis)

    # ----- numerics header mapping -----
    num_head = pd.read_csv(args.numerics, nrows=0)
    ncols = list(num_head.columns)
    low = {c.lower(): c for c in ncols}
    col_csn = low.get("csn")
    col_measure = low.get("measure") or low.get("name") or low.get("parameter") or low.get("label")
    col_value = low.get("value") or low.get("val") or low.get("result")
    col_time = low.get("time") or low.get("charttime") or low.get("timestamp") or low.get("recorded_time") or low.get("event_time")
    if not (col_csn and col_measure and col_value):
        raise SystemExit("ERROR: numerics.csv must have CSN, Measure(name/parameter/label), and Value(val/result).")
    usecols = [col_csn, col_measure, col_value] + ([col_time] if col_time else [])

    keep_vitals = {"temp","hr","rr","spo2","sbp","dbp","map","gcs"}
    collected = []
    total_rows = 0
    kept_rows = 0

    # ----- stream & keep earliest per CSN+vital -----
    reader = pd.read_csv(args.numerics, usecols=usecols, chunksize=args.chunksize)
    for chunk in reader:
        total_rows += len(chunk)
        chunk = chunk.rename(columns={col_csn:"CSN", col_measure:"Measure", col_value:"Value"})
        if col_time: chunk = chunk.rename(columns={col_time:"Time"})
        chunk["Measure"] = chunk["Measure"].astype(str).str.strip()
        chunk["_vital"] = chunk["Measure"].apply(normalize_measure)
        ch = chunk[chunk["_vital"].isin(keep_vitals)].copy()
        kept_rows += len(ch)
        if "Time" in ch.columns:
            t = pd.to_datetime(ch["Time"], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
            if t.isna().any():
                t2 = pd.to_datetime(ch.loc[t.isna(),"Time"], errors="coerce")
                t.loc[t.isna()] = t2
            ch["Time"] = t
            ch = ch.sort_values(["CSN","_vital","Time"]).drop_duplicates(subset=["CSN","_vital"], keep="first")
        else:
            ch = ch.drop_duplicates(subset=["CSN","_vital"], keep="first")
        collected.append(ch[["CSN","_vital","Value"] + (["Time"] if "Time" in ch.columns else [])])

    # ----- assemble wide triage table -----
    if collected:
        data = pd.concat(collected, ignore_index=True)
        if "Time" in data.columns:
            data = data.sort_values(["CSN","_vital","Time"]).drop_duplicates(subset=["CSN","_vital"], keep="first")
        wide = data.pivot_table(index="CSN", columns="_vital", values="Value", aggfunc="first").reset_index()
    else:
        wide = pd.DataFrame({"CSN": vis["CSN"].unique()})

    # ----- build triage + scores -----
    out = vis[["MRN","CSN"]].drop_duplicates() if "MRN" in vis.columns else vis[["CSN"]].drop_duplicates()
    out = out.merge(wide, on="CSN", how="left")
    for c in ["temp","hr","rr","sbp","dbp","spo2","gcs","map"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan

    temp_c = out["temp"]
    if temp_c.notna().sum() > 0:
        med = temp_c.median(skipna=True)
        if 50 <= med <= 120:
            temp_c = temp_c.apply(lambda x: f_to_c(x) if pd.notna(x) else np.nan)
    out["triage_temp_c"] = temp_c
    out["triage_hr"] = out["hr"]
    out["triage_rr"] = out["rr"]
    out["triage_sbp"] = out["sbp"]
    out["triage_dbp"] = out["dbp"]
    out["triage_spo2"] = out["spo2"]
    out["triage_gcs"] = out["gcs"]
    out["triage_map"] = out["map"]
    need_map = out["triage_map"].isna()
    out.loc[need_map, "triage_map"] = [estimate_map(s,d) for s,d in zip(out.loc[need_map,"triage_sbp"], out.loc[need_map,"triage_dbp"])]

    # score columns
    out["score_qSOFA"] = [score_qsofa(r,s,g) for r,s,g in zip(out["triage_rr"], out["triage_sbp"], out["triage_gcs"])]
    out["score_MEWS"]  = [score_mews(r,h,s,t) for r,h,s,t in zip(out["triage_rr"], out["triage_hr"], out["triage_sbp"], out["triage_temp_c"])]
    avpu_proxy = out["triage_gcs"].apply(lambda g: 0 if pd.notna(g) and g==15 else (3 if pd.notna(g) and g<15 else None))
    out["score_NEWS2"] = [score_news2(r,sp,t,s,h,a) for r,sp,t,s,h,a in zip(out["triage_rr"], out["triage_spo2"], out["triage_temp_c"], out["triage_sbp"], out["triage_hr"], avpu_proxy)]

    # Optional REMS
    if "Age" in vis_age.columns:
        age_map = vis_age[["CSN","Age"]].drop_duplicates()
        out = out.merge(age_map, on="CSN", how="left")
        a = out["Age"].apply(rems_points_age)
        m = out["triage_map"].apply(rems_points_map)
        h = out["triage_hr"].apply(rems_points_hr)
        r = out["triage_rr"].apply(rems_points_rr)
        s = out["triage_spo2"].apply(rems_points_spo2)
        g = out["triage_gcs"].apply(rems_points_gcs)
        comp = pd.concat([a,m,h,r,s,g], axis=1)
        out["score_REMS"] = comp.sum(axis=1, min_count=1)

    # Optional CCI
    if args.dx:
        dx = pd.read_csv(args.dx)
        dx["score_CCI"] = compute_cci_from_flags(dx)
        keys = ["CSN"]
        if "MRN" in dx.columns and "MRN" in out.columns:
            keys = ["MRN","CSN"] if "CSN" in dx.columns else ["MRN"]
        out = out.merge(dx[keys + ["score_CCI"]].drop_duplicates(), on=keys, how="left")

    # ----- FINAL OUTPUT: all visits columns + triage & scores -----
    # Merge on CSN (and MRN if present on both)
    keys = ["CSN"]
    if "MRN" in vis.columns and "MRN" in out.columns:
        keys = ["MRN","CSN"]
# Align keys between vis and out, and perform a safe left merge.
out, keys = _align_keys(vis, out)
_cols_to_drop = [c for c in out.columns if c in ("MRN", "CSN") and c in vis.columns]
final_df = vis.merge(out.drop(columns=_cols_to_drop, errors="ignore"), on=keys, how="left")
# Ensure all visits columns come first.
_ordered_cols = list(vis.columns) + [c for c in final_df.columns if c not in vis.columns]
final_df = final_df.reindex(columns=_ordered_cols)
    # Order: all visits columns first, then triage_* and score_* (already arranged by merge)
final_df.to_csv(args.out, index=False)

    # Print summary
    miss = lambda c: (final_df[c].isna().mean()*100 if c in final_df.columns else 100.0)
    print(f"✅ Saved {args.out}  | rows={len(final_df)}")
    print(f"Missing rates -> RR:{miss('triage_rr'):.2f}% SBP:{miss('triage_sbp'):.2f}% Temp:{miss('triage_temp_c'):.2f}% SpO2:{miss('triage_spo2'):.2f}%")

if __name__ == "__main__":
    main()