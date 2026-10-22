#!/usr/bin/env python3
"""Stage 4: chief-complaint flags from CC, and the T0-only chest-pain cohort flag."""
import time, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

m = pd.read_pickle(OUT / "_stage3.pkl")

# CC is a free-text, comma-separated list of triage complaints (e.g. "ALTERED
# MENTAL STATUS,SHORTNESS OF BREATH"), the MC-MED analog of MIMIC's
# ed/triage.chiefcomplaint. Same ten regex categories as the MIMIC build.
CC = {
    "chest_pain":          r"chest\s*pain|\bcp\b",
    "abdominal_pain":      r"abd(ominal)?\s*pain",
    "headache":            r"headache|h/?a\b",
    "shortness_of_breath": r"\bsob\b|short(ness)?\s*of\s*breath|dyspnea|difficulty breathing",
    "back_pain":           r"back\s*pain",
    "cough":               r"cough",
    "nausea_vomiting":     r"nausea|vomit|\bn/?v\b|emesis",
    "fever_chills":        r"fever|chills|febrile",
    "syncope":             r"syncope|fainting|passed out|loc\b",
    "dizziness":           r"dizz|vertigo|light\s*head",
}
cctext = m["CC"].astype("string").fillna("").str.lower()
for name, pat in CC.items():
    m[f"chiefcom_{name}"] = cctext.str.contains(pat, regex=True, na=False).astype("int8")

# Chest-pain cohort: chief complaint ONLY, not any ED/discharge diagnosis. This
# mirrors the fix made to the MIMIC build after the diagnosis-based OR was found
# to enroll patients via R07.9/786.50 ("chest pain, unspecified") — an
# exclusionary code assigned once ACS etc. have already been ruled out, which
# inflates the cohort with pre-screened negatives and is not T0 information
# regardless. Using the triage complaint alone keeps the cohort definition
# available at the moment of triage and free of that bias.
m["cohort_chest_pain"] = m["chiefcom_chest_pain"]

m.to_pickle(OUT / "_stage4.pkl")
log(f"stage 4 complete -> {m.shape}")
cc_cols = [c for c in m.columns if c.startswith("chiefcom_")]
print((m[cc_cols].mean() * 100).round(2).sort_values(ascending=False).to_string())
print(f"\ncohort_chest_pain: {m.cohort_chest_pain.sum():,} ({m.cohort_chest_pain.mean()*100:.2f}%)")
