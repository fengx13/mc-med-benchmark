#!/usr/bin/env python3
"""Stage 6: ED laboratory panel from labs.csv, one static value per assay.

labs.csv is already scoped to CSN (the ED visit), unlike MIMIC's labevents.csv
which spans a patient's whole hospitalization and needed a time-window join —
so this is a straight groupby, no searchsorted matching required. Per the same
instruction as the MIMIC build, each lab is a single static value: if the assay
resulted at all, the visit's value is its earliest result within the window
[Arrival_time, max(Departure_time, Arrival_time+24h)] — widened past ED
departure for the same reason as MIMIC (a troponin drawn in the ED often
results after the patient is moved to a floor bed).
"""
import re, time, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "derived"
_t0 = time.time()
def log(m): print(f"[{time.time()-_t0:7.1f}s] {m}", flush=True)

m = pd.read_pickle(OUT / "_stage5.pkl")
N = len(m)

log("loading labs.csv")
labs = pd.read_csv(ROOT / "data/labs.csv",
                   usecols=["CSN", "Order_time", "Component_name", "Component_value"],
                   parse_dates=["Order_time"])
labs["Order_time"] = pd.to_datetime(labs["Order_time"], utc=True).dt.tz_localize(None)
labs["valuenum"] = pd.to_numeric(labs["Component_value"], errors="coerce")
labs = labs.dropna(subset=["valuenum"])
log(f"{len(labs):,} numeric-parseable lab component results")

win = m[["stay_id", "intime", "outtime"]].rename(columns={"stay_id": "CSN"})
win["cap"] = np.maximum(win["outtime"].to_numpy().astype("datetime64[s]"),
                        (win["intime"] + pd.Timedelta(hours=24)).to_numpy().astype("datetime64[s]"))
labs = labs.merge(win[["CSN", "intime", "cap"]], on="CSN", how="inner")
labs["offset_min"] = (labs["Order_time"] - labs["intime"]).dt.total_seconds() / 60
labs = labs[(labs["offset_min"] >= 0) & (labs["Order_time"] <= labs["cap"])]
log(f"{len(labs):,} results inside the capture window")

cov = labs.groupby("Component_name")["CSN"].nunique()
CP_ASSAYS = ["TROPONIN I", "HIGH SENSITIVITY TROPONIN", "D-DIMER", "D-DIMER, ELISA",
            "BNP, NT-PRO", "CK-MB", "CK-MB RELATIVE INDEX", "CK, TOTAL", "MYOGLOBIN",
            "C-REACTIVE PROTEIN (CRP)", "PROCALCITONIN", "LACTATE.WHOLE BLD",
            "POC:LACTATE, ISTAT", "POC:TROPONIN I, POCT"]
panel = sorted(set(cov[cov >= 0.05 * N].index) | (set(CP_ASSAYS) & set(cov.index)))
labs = labs[labs["Component_name"].isin(panel)]
log(f"panel: {len(panel)} assays")


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return re.sub(r"_+", "_", s)


name_map = {c: slug(c) for c in panel}
assert len(set(name_map.values())) == len(name_map), "slug collision in MC-MED lab panel"

log("pivoting to one static value per assay")
labs = labs.sort_values(["CSN", "Component_name", "Order_time"])
first = labs.groupby(["CSN", "Component_name"])["valuenum"].first().reset_index()
first["col"] = first["Component_name"].map(name_map)
W = first.pivot(index="CSN", columns="col", values="valuenum")
W.columns = [f"lab_{c}" for c in W.columns]
W = W.astype("float32")
m = m.merge(W, left_on="stay_id", right_index=True, how="left")

allb = labs
m["n_labs_ed"] = m["stay_id"].map(allb.groupby("CSN").size()).fillna(0).astype("int32")
m["n_lab_assays_ed"] = m["stay_id"].map(allb.groupby("CSN")["Component_name"].nunique()).fillna(0).astype("int16")
m["time_to_first_lab_min"] = m["stay_id"].map(allb.groupby("CSN")["offset_min"].min()).astype("float32")
m["has_labs_ed"] = (m["n_labs_ed"] > 0).astype("int8")

meta = pd.DataFrame({"component_name": list(name_map), "column": [f"lab_{name_map[c]}" for c in name_map]})
meta["n_results"] = meta["component_name"].map(labs.groupby("Component_name").size())
meta["n_stays"] = meta["component_name"].map(cov)
meta.sort_values("n_stays", ascending=False).to_csv(OUT / "lab_panel_dictionary.csv", index=False)

m.to_pickle(OUT / "_stage6.pkl")
log(f"stage 6 complete -> {m.shape}")
print("\nchest-pain lab coverage (% of all visits):")
for c in ["TROPONIN I", "HIGH SENSITIVITY TROPONIN", "D-DIMER", "BNP, NT-PRO", "CK-MB",
         "LACTATE.WHOLE BLD"]:
    col = f"lab_{name_map.get(c,'')}"
    if col in m.columns:
        print(f"  {col:30s} {m[col].notna().mean()*100:5.1f}%")
