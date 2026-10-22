#!/usr/bin/env python3
"""Stage 3: Charlson/Elixhauser comorbidity flags from pmh.csv, 5-year lookback.

pmh.csv is patient-level (keyed by MRN) with one row per historical diagnosis and
its Noted_date — the MC-MED analog of MIMIC's diagnoses_icd, except there is no
admission wrapper: a code is just a (date, code) pair on the patient's problem
list. CodeType is 'Dx9'/'Dx10' and Code is already dot-less, so the same Quan
(2005) ICD-9-CM/ICD-10 maps used for the MIMIC build apply unchanged.
"""
import sys, time, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
MIMIC_SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(MIMIC_SCRIPTS))
from codemaps import CCI_9, CCI_10, ECI_9, ECI_10, CCI_CATS, ECI_CATS

LOOKBACK = pd.Timedelta(days=365.25 * 5)
_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)


def or_reduce(group_ids, masks):
    starts = np.flatnonzero(np.r_[True, group_ids[1:] != group_ids[:-1]])
    return group_ids[starts], np.bitwise_or.reduceat(masks, starts)


m = pd.read_pickle(OUT / "_stage1.pkl")
# stage2 columns aren't needed here but keep everything downstream chained on stage2
m2 = pd.read_pickle(OUT / "_stage2.pkl")
assert (m["stay_id"] == m2["stay_id"]).all()
m = m2

log("loading pmh.csv")
pmh = pd.read_csv(ROOT / "data/pmh.csv", usecols=["MRN", "Noted_date", "CodeType", "Code"],
                  parse_dates=["Noted_date"])
pmh["Noted_date"] = pd.to_datetime(pmh["Noted_date"], utc=True).dt.tz_localize(None)
pmh["Code"] = pmh["Code"].astype("string").str.strip().str.upper()
pmh["icd_version"] = pmh["CodeType"].map({"Dx9": 9, "Dx10": 10})
pmh = pmh.dropna(subset=["icd_version", "Code"])
log(f"{len(pmh):,} pmh rows, {pmh.MRN.nunique():,} patients")

log("building comorbidity bitmask on distinct codes")
uniq = pmh[["Code", "icd_version"]].drop_duplicates().reset_index(drop=True)
is9 = uniq["icd_version"].to_numpy() == 9
code = uniq["Code"]
FLAG_COLS, mask = [], np.zeros(len(uniq), dtype="int64")
for prefix, m9, m10, cats in [("cci", CCI_9, CCI_10, CCI_CATS), ("eci", ECI_9, ECI_10, ECI_CATS)]:
    for cat in cats:
        hit = np.where(is9, code.str.startswith(tuple(m9[cat]), na=False),
                       code.str.startswith(tuple(m10[cat]), na=False))
        mask |= hit.astype("int64") << len(FLAG_COLS)
        FLAG_COLS.append(f"{prefix}_{cat}")
uniq["mask"] = mask
log(f"{len(uniq):,} distinct codes -> {len(FLAG_COLS)} flags packed into int64")

pmh = pmh.merge(uniq[["Code", "icd_version", "mask"]], on=["Code", "icd_version"], how="left")
pmh["mask"] = pmh["mask"].fillna(0).astype("int64")

log("building visit x prior-pmh-code pairs (5-year lookback, strictly before arrival)")
pairs = m[["subject_id", "stay_id", "intime"]].merge(
    pmh[["MRN", "Noted_date", "mask"]], left_on="subject_id", right_on="MRN", how="inner")
keep = (pairs["Noted_date"] < pairs["intime"]) & (pairs["Noted_date"] >= pairs["intime"] - LOOKBACK)
pairs = pairs.loc[keep, ["stay_id", "mask"]]
log(f"{len(pairs):,} (visit, prior pmh code) pairs in the {LOOKBACK.days}-day window")

ps = pairs.sort_values("stay_id")
s_id, s_mask = or_reduce(ps["stay_id"].to_numpy(), ps["mask"].to_numpy())
stay_mask = m["stay_id"].map(pd.Series(s_mask, index=s_id)).fillna(0).astype("int64").to_numpy()
for i, c in enumerate(FLAG_COLS):
    m[c] = ((stay_mask >> i) & 1).astype("int8")

# Any-time history (no 5-year cutoff), for reference / sensitivity analysis.
pairs_any = m[["subject_id", "stay_id", "intime"]].merge(
    pmh[["MRN", "Noted_date", "mask"]], left_on="subject_id", right_on="MRN", how="inner")
pairs_any = pairs_any.loc[pairs_any["Noted_date"] < pairs_any["intime"], ["stay_id", "mask"]]
pa = pairs_any.sort_values("stay_id")
if len(pa):
    a_id, a_mask = or_reduce(pa["stay_id"].to_numpy(), pa["mask"].to_numpy())
    any_mask = m["stay_id"].map(pd.Series(a_mask, index=a_id)).fillna(0).astype("int64")
else:
    any_mask = pd.Series(0, index=m.index)
m["n_pmh_codes_5y"] = m["stay_id"].map(pairs.groupby("stay_id").size()).fillna(0).astype("int32")
m["n_pmh_codes_any"] = m["stay_id"].map(pairs_any.groupby("stay_id").size()).fillna(0).astype("int32")
m["has_pmh_5y"] = (m["n_pmh_codes_5y"] > 0).astype("int8")

log("computing score_CCI")
import scores as sc
m["score_CCI"] = sc.score_cci(m, m["age"])

m.to_pickle(OUT / "_stage3.pkl")
log(f"stage 3 complete -> {m.shape}")
has = m.has_pmh_5y == 1
print(f"\nvisits with any 5y pmh history: {has.sum():,} ({has.mean()*100:.1f}%)")
cci_cols = [c for c in FLAG_COLS if c.startswith("cci_")]
print("\nCharlson prevalence among visits WITH history:")
print((m.loc[has, cci_cols].mean() * 100).round(1).sort_values(ascending=False).to_string())
