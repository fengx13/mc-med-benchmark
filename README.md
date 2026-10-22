# MC-MED 5-Outcome Benchmark

Reproduces the **MultiAutoScore** MC-MED master dataset — the same cohort,
the same 256 predictor columns (demographics, triage and serial vital signs,
five-year comorbidity history from pmh.csv, laboratory panel, severity
scores) — trimmed to five clinical outcomes instead of the full project's
forty-nine. This is the MC-MED companion to the
[MIMIC-IV-ED 5-outcome benchmark](../mimic-iv-ed-benchmark) and uses the
same outcome definitions wherever the two datasets overlap.

| Outcome | Column | Definition |
|---|---|---|
| Acute coronary syndrome | `outcome_cp_acs` | ICD-9/10 prefix match (same `cp_codes.py` definition as the MIMIC benchmark) |
| Sepsis | `outcome_sepsis` | ICD-9/10 exact match |
| Acute heart failure (AHF) | `outcome_hf_acute` | ICD-9/10 exact match (acute / acute-on-chronic only) |
| Stroke | `outcome_stroke` | ICD-9/10 exact match (non-traumatic cerebrovascular events) |
| Pulmonary embolism | `outcome_cp_pe` | ICD-9/10 prefix match (same `cp_codes.py` definition as the MIMIC benchmark) |

**This repository's output was verified column-by-column against the full
49-outcome MC-MED master dataset: all 256 shared predictor columns and all 5
outcome columns are value-identical.**

## Read this before you trust any outcome rate here

MC-MED's `visits.csv` carries only the visit's **PRIMARY diagnosis**
(`Dx_ICD9`/`Dx_ICD10` — occasionally 2–4 comma-separated codes when several
are equally primary, in about 4.2% of visits), not a full coded diagnosis
list the way MIMIC-IV's `diagnoses_icd` + `ed/diagnosis` are. A patient whose
sepsis, AHF, stroke, ACS or PE was captured only as a *secondary* diagnosis
is a false negative here. **Every rate below is a floor, not a true
prevalence**, and is not comparable in absolute terms to the MIMIC
benchmark's rate for the same outcome using identical code logic:

| Outcome | MC-MED rate | MIMIC-IV-ED rate (same codes) |
|---|---:|---:|
| Sepsis | 0.71% | 2.68% |
| AHF | 0.42% | 2.99% |
| Stroke | 0.99% | 1.90% |
| ACS | 0.78% | 2.21% |
| PE | 0.39% | 0.72% |

What *does* still hold is relative enrichment — e.g. ACS and PE rates inside
the chest-pain-complaint subgroup versus outside it — because the same
undercounting mechanism applies to both groups equally.

## What you get

One row per MC-MED ED visit (CSN), **261 columns**: demographics and timing,
core ED outcomes (hospitalization, ICU transfer, in-hospital mortality,
3-day revisit — note MC-MED structurally cannot support an ED-bay-death
column, see Limitations), prior utilization counts, triage vitals plus
serial nursing-chart vitals from `numerics-001.csv` (cleaned, median-imputed,
with missingness flags), supplemental-oxygen flag, ten chief-complaint
flags, 48 Charlson/Elixhauser comorbidity flags from `pmh.csv`, six bedside
severity scores, a laboratory panel, home and ED-administered medication
counts, and the five outcomes above.

## Data access

You need **credentialed PhysioNet access** to MC-MED — this repository
contains no patient data, only code:

**MC-MED v1.0** (Multimodal Clinical Monitoring in the Emergency Department)
— <https://physionet.org/content/mc-med/1.0.0/>

This requires a signed PhysioNet Data Use Agreement and a completed human
subjects research training course. Row-level output from this pipeline is
equally restricted — do not share it outside your own DUA, and do not send
it to any third-party service.

## Setup

```
mcmed-benchmark/
├── data/                 <- MC-MED's data/ directory contents go here
│   ├── visits.csv
│   ├── labs.csv
│   ├── pmh.csv
│   ├── orders.csv
│   └── meds.csv
├── numerics-001.csv      <- MC-MED's numerics file, at the repo ROOT (not under data/)
├── derived/              <- pipeline writes its outputs here
└── scripts/
```

```bash
ln -s /path/to/your/mcmed/data             data
ln -s /path/to/your/mcmed/numerics-001.csv numerics-001.csv
pip install -r requirements.txt
```

## Running the pipeline

```bash
python3 scripts/01_cohort.py          # cohort spine, core outcomes, utilization
python3 scripts/02_vitals.py          # triage + serial vitals from numerics-001.csv (~25s, scans 47.8M rows)
python3 scripts/03_comorbidity.py     # Charlson/Elixhauser from pmh.csv
python3 scripts/04_chiefcomplaint.py  # chief-complaint flags, T0-only chest-pain cohort
python3 scripts/05_outcomes.py        # the 5 benchmark outcomes
python3 scripts/06_labs.py            # lab panel, one static value per assay
python3 scripts/07_meds.py            # home + ED medication counts
python3 scripts/08_assemble.py        # final column order, write outputs
```

Stages must run in this order (each reads the previous stage's checkpoint).
Total runtime is roughly 1 minute on a laptop.

**Output**: `derived/master_benchmark5.csv.gz`, `master_benchmark5.pkl`
(preserves dtypes), and `column_dictionary.csv`.

## Outcome definitions

### ACS and PE — prefix match (`scripts/cp_codes.py`)

Identical codes to the MIMIC-IV-ED benchmark — see that repository's README
for the full prefix tables and the clinical audit behind them.

### Sepsis, AHF, stroke — exact match (`scripts/outcome_codes_exact.py`)

Identical codes to the MIMIC-IV-ED benchmark (sepsis: 21 ICD-9 / 29 ICD-10
codes; AHF: 6 ICD-9 / 8 ICD-10, acute-only; stroke: 15 ICD-9 / 161 ICD-10).
Matching here is against the visit's *primary* diagnosis code list, not a
full admission diagnosis table — see the caveat above.

## Known limitations specific to MC-MED

- **No ED-bay-death signal is possible.** `ED_dispo` has four levels
  (Discharge/Observation/Inpatient/ICU) and none of them is death. Checked
  directly: of the 800 visits whose final hospital disposition is
  'Expired', 799 already carry a non-Discharge `ED_dispo` — the death
  happened after admission, not in the ED. `outcome_inhospital_mortality`
  here means "died anywhere during the encounter," not "died in the ED,"
  unlike MIMIC's `ed_death`.
- **GCS does not exist in MC-MED either.** The `numerics-001.csv` Measure
  list has exactly 12 entries (HR, RR, SpO2, SBP, DBP, MAP, Temp, Perf,
  Pain, LPM_O2, 1min_HRV, 5min_HRV) — no consciousness level.
- **NEWS2 genuinely differs from NEWS here**, unlike the MIMIC benchmark
  (which has no supplemental-oxygen data and makes the two numerically
  identical). MC-MED's `LPM_O2` channel lets NEWS2's oxygen-dependent
  scoring band activate for the 6.78% of visits recorded on supplemental O2.
- **Troponin / D-dimer / BNP are sparse**, same clinical reasoning as the
  MIMIC benchmark — note two non-interchangeable assay variants exist for
  troponin (conventional vs. high-sensitivity, different diagnostic
  thresholds, zero overlapping visits) and for lactate (central-lab vs.
  point-of-care, 1,342 visits have both). Both pairs are kept as separate
  `lab_*` columns rather than merged.
- **Ventilation timing (`first_vent_time`) is too sparse to anchor any
  time-resolved outcome** — only 8–9 visits total show a timestamped
  intubation/ventilator order against 2,766 ICU admissions. It is kept as a
  predictor column only; no outcome in this benchmark depends on it.
- **`numerics-001.csv` covers 86.1%** of visits (101,893 of 118,385) — the
  remaining 13.9% have no serial-vitals data, only the one triage-time
  reading.

## Reproducibility

This repository's output was verified **column-by-column against the full
49-outcome master dataset**: all 256 shared predictor columns and all 5
outcome columns are value-identical (0 mismatches) — but only against the
`numerics-001.csv` this was built with.

**The single biggest risk to reproducing this exactly is numerics
completeness.** `numerics-001.csv` is a single part file covering 86.1% of
visits (101,893 of 118,385) — its filename suggests it may be one part of a
multi-file download, and the official MC-MED distribution could ship
additional `numerics-NNN.csv` parts or a single complete file. If you obtain
more complete numerics data than was used here, every column derived from
it will differ: `triage_*_obs_first/last`, `n_vitals_obs_ed`,
`on_supplemental_o2`, `max_lpm_o2`, `score_NEWS2` (wherever it differs from
NEWS), and the vital-imputation rate. This is expected — more complete input
data should produce a more complete (and more correct) dataset — but it
means your numbers will not match the figures quoted in this README if your
numerics file differs from the one used to build them.

Everything else carries the same version-pinning and sort-stability
guarantees documented in the
[MIMIC-IV-ED benchmark's README](../mimic-iv-ed-benchmark/README.md#reproducibility):
`requirements.txt` pins exact tested versions, and the tie-sensitive sorts
in `02_vitals.py` and `06_labs.py` (5.6% of MC-MED lab rows share an exact
tie key) are multi-column and therefore stable via `numpy.lexsort`,
independent of pandas version.

## Citation

Use the exact citation given on the dataset's own PhysioNet page —
<https://physionet.org/content/mc-med/1.0.0/> — this README does not
reproduce it to avoid attributing the wrong author list.
