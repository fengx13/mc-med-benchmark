MC-MED Benchmark
=========================
Python workflow for generating benchmark datasets and machine learning models from the MC-MED database.

## General Information

Making timely decisions in the emergency department is essential for both patient care and efficient use of scarce resources, which has driven the growing adoption of machine learning prediction models in emergency medicine.

The Multimodal Clinical Monitoring in the Emergency Department (MC-MED) dataset, recently released by Stanford Health Care on PhysioNet, addresses this gap by being the first publicly available ED dataset to combine structured EHR data, continuous vital signs, high-frequency waveforms, and free-text clinical documentation. With over 118,000 ED visits from more than 70,000 patients, MC-MED provides an unprecedented resource for developing and evaluating predictive models that leverage the complementary strengths of multiple data modalities. [MC-MED](https://physionet.org/content/mc-med/1.0.0/)

Here, we will present a workflow that generates a benchmark dataset from the MC-MED database.

## Structure

The structure of this repository is detailed as follows:

- `benchmark_scripts/...` contains the scripts for benchmark dataset generation (master_data.csv).

## Requirements and Setup
MC-MED database is not provided with this repository and is **required** for this workflow. MC-MED can be downloaded from [https://physionet.org/content/mc-med/1.0.0/](https://physionet.org/content/mc-med/1.0.0/)

## Workflow 

Before proceeding, download and set up the MC-MED repository locally. 

### 1. Benchmark Data Generation
~~~
python extract_master_dataset.py 
~~~

**Arguements**:
- `VISITS_PATH` : Path to the directory containing the patient's ED visit data.
- `MEDS_PATH ` : Path to the directory containing the patient's home medication records.
- `PMH_PATH` : Path to the directory containing the patient's past medical history.
- `LABS_PATH` : Path to the directory containing the patient's laboratory tests and results.
- `output_path` : Path to output directory

**Output**:

`master_dataset.csv` output to `output_path`

**Details**:

A total of **216** variables are included in `master_dataset.csv`
