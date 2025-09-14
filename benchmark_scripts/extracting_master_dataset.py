import pandas as pd
import numpy as np
from benchmark_scripts.helpers import *
from benchmark_scripts.util import *
from benchmark_scripts.dataset_path import *
from mappers import *
from benchmark_scripts.disease_defs import DISEASE_DEFS

df_visits = pd.read_csv(VISITS_PATH)
df_meds = pd.read_csv(MEDS_PATH)
df_pmh = pd.read_csv(PMH_PATH)
df_labs = pd.read_csv(LABS_PATH)

visits_history = df_visits.copy()
meds_history = df_meds.copy()
pmh_history = df_pmh.copy()

patient_visit_columns = ['subject_id', 'stay_id',
                         'age', 'gender', 'race', 'ethnicity',
                         'intime', 'outtime', 'in_year', 'edregtime', 'edouttime', 'admittime', 'dischtime', 'intime_icu', 'outtime_icu',
                         'ed_los', 'ed_los_hours', 'next_ed_visit_time', 'next_ed_visit_time_diff', 'next_ed_visit_time_diff_days',
                         'time_to_icu_transfer_hours',
                         'triage_temperature', 'triage_heartrate', 'triage_resprate', 'triage_o2sat', 'triage_sbp', 'triage_dbp', 'triage_acuity',
                         'triage_MAP']
df_master = pd.DataFrame(columns=patient_visit_columns)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


basic_column_mapping = {
    'MRN': 'subject_id',
    'CSN': 'stay_id', 
    'Age': 'age',
    'Gender': 'gender',
    'Race': 'race',
    'Ethnicity': 'ethnicity'
}

df_master[list(basic_column_mapping.values())] = df_visits[list(basic_column_mapping.keys())]

triage_mapping = {
    'Triage_Temp': 'triage_temperature',
    'Triage_HR': 'triage_heartrate', 
    'Triage_RR': 'triage_resprate',
    'Triage_SpO2': 'triage_o2sat',
    'Triage_SBP': 'triage_sbp',
    'Triage_DBP': 'triage_dbp'
}

df_master[list(triage_mapping.values())] = df_visits[list(triage_mapping.keys())]

time_column_mapping = {
    'Arrival_time': 'intime',
    'Departure_time': 'outtime', 
    'Roomed_time': 'edregtime',
    'Dispo_time': 'edouttime'
}

for source_col, target_col in time_column_mapping.items():
    df_master[target_col] = adjust_time_by_subtracting_year(df_visits[source_col])

df_master['admittime'] = adjust_time_by_subtracting_year(df_visits[df_visits['ED_dispo'] == 'Inpatient']['Admit_time'])
df_master['intime_icu'] = adjust_time_by_subtracting_year(df_visits[df_visits['ED_dispo'] == 'ICU']['Admit_time'])

hosp_los_timedelta = pd.to_timedelta(df_visits['Hosp_LOS'], unit='D')
derived_time_operations = {
    'dischtime': lambda: df_master['admittime'] + hosp_los_timedelta,
    'outtime_icu': lambda: df_master['intime_icu'] + hosp_los_timedelta,
    'in_year': lambda: df_visits['Arrival_time'].str[:4].astype(int)
}

for col_name, operation in derived_time_operations.items():
    df_master[col_name] = operation()

visit_interval = pd.to_timedelta(df_visits['Hours_to_next_visit'], unit='h')
visit_metrics = {
    'ed_los': round(df_visits['ED_LOS'] / 24, 1),
    'ed_los_hours': df_visits['ED_LOS'],
    'next_ed_visit_time': df_master['intime'] + visit_interval,
    'next_ed_visit_time_diff': df_visits['Hours_to_next_visit'], 
    'next_ed_visit_time_diff_days': round(df_visits['Hours_to_next_visit'] / 24, 1)
}

for col_name, values in visit_metrics.items():
    df_master[col_name] = values

time_diff_to_icu = df_master['intime_icu'] - df_master['intime']
df_master['time_to_icu_transfer_hours'] = round(time_diff_to_icu.dt.total_seconds() / 3600, 2)

df_master['triage_acuity'] = df_visits['Triage_acuity'].str[0].replace('n', np.nan).astype('Int64')
df_master['triage_MAP'] = round((df_master['triage_sbp'] + 2 * df_master['triage_dbp']) / 3, 1)

complaint_dict = {
    "chiefcom_chest_pain": "chest pain|tight chest|pressure chest|angina|retrosternal pain",
    "chiefcom_abdominal_pain": "abdominal pain|abd pain|stomach ache|belly pain|tummy ache",
    "chiefcom_headache": "headache|lightheaded|migraine|cephalgia|pressure head",
    "chiefcom_shortness_of_breath": "breath",
    "chiefcom_back_pain": "back pain|lower back pain|lumbago|upper back pain",
    "chiefcom_cough": "cough|dry cough|productive cough|hacking cough",
    "chiefcom_nausea_vomiting": "nausea|vomit|vomiting|emesis|throwing up",
    "chiefcom_fever_chills": "fever|chill|temperature|feverish|shivering",
    "chiefcom_syncope": "syncope|faint|fainted|blackout|loss of consciousness",
    "chiefcom_dizziness": "dizz|dizzy|vertigo|lightheaded|spinning|off balance"
}

df_master['CC'] = df_visits['CC']
df_master = encode_chief_complaints(df_master, complaint_dict)

visits_history['Arrival_time'] = adjust_time_by_subtracting_year(visits_history['Arrival_time'])
meds_history[['Start_date','End_date']] = meds_history[['Start_date','End_date']].apply(adjust_time_by_subtracting_year)
pmh_history['Noted_date'] = adjust_time_by_subtracting_year(pmh_history['Noted_date'])

visits_history['Admit_service'] = visits_history['Admit_service'].astype(str).fillna('')

pmh_history['Code'] = pmh_history['Code'].astype(str)
pmh_history['Code'] = pmh_history['Code'].str.replace('.', '', regex=False)

comorbidity_codes = build_comorbidity_code_dictionary()

visits_history = visits_history.sort_values(by=['MRN', 'Arrival_time']).reset_index(drop=True)
meds_history.set_index('MRN', inplace=True)
pmh_history = pmh_history.sort_values(by=['MRN', 'Noted_date'])

visit_history_columns = [
    'n_ed_30d', 'n_ed_90d', 'n_ed_365d',
    'n_hosp_30d', 'n_hosp_90d', 'n_hosp_365d',
    'n_icu_30d', 'n_icu_90d', 'n_icu_365d'
]
visit_history_mapping = calculate_visit_history(visits_history)

df_master = pd.merge(
    df_master,
    visit_history_mapping,
    left_on=['subject_id', 'stay_id'],
    right_on=['MRN', 'CSN'],
    how='left'
).drop(columns=['MRN', 'CSN'])

med_history_columns = ['n_med', 'n_medrecon']
med_history_results = calculate_medication_history(visits_history, meds_history)

df_master = pd.merge(
    df_master,
    med_history_results,
    left_on=['subject_id', 'stay_id'],
    right_on=['MRN', 'CSN'],
    how='left'
).drop(columns=['MRN', 'CSN'])

comorbidity_results = calculate_comorbidities_batch(visits_history, pmh_history, comorbidity_codes)

df_master = pd.merge(
    df_master,
    comorbidity_results,
    left_on=['subject_id', 'stay_id'],
    right_on=['MRN', 'CSN'],
    how='left'
).drop(columns=['MRN', 'CSN'])

add_triage_MAP(df_master)
add_score_REMS(df_master)
add_score_CART(df_master)
add_score_NEWS(df_master)
add_score_NEWS2(df_master)
add_score_MEWS(df_master)
add_score_CCI(df_master)

score_columns = ['stay_id', 'score_CCI', 'score_CART','score_REMS',  'score_NEWS', 'score_NEWS2', 'score_MEWS']
final_score_columns = [col for col in score_columns if col in df_master.columns]

df_labs_processed = df_labs.copy()
test_column_name = 'Component_name'
csn_column_name = 'CSN'
value_column_name = 'Component_value'

df_labs_filtered = process_lab_data(df_labs, min_frequency=10000)

df_labs_aggregated = (df_labs_filtered
                      .groupby(['CSN', 'normalized_test'])['Component_value']
                      .mean()
                      .reset_index())

df_labs_wide = (df_labs_aggregated
                .pivot(index='CSN', columns='normalized_test', values='Component_value')
                .reset_index())

df_labs_wide.columns.name = None

df_master = merge_lab_data_with_visits(df_master, df_labs_wide)

print(f"Final dataset shape: {df_master.shape}")

visits_history_copy = df_visits.copy()

time_cols = ["Arrival_time", "Roomed_time", "Departure_time", "Dispo_time", "Admit_time"]

for col in time_cols:
    year = (
        visits_history_copy[col].str.extract(r"(\d{4})")[0]
        .dropna()
        .astype(int) - 200
    )
    
    decrypted_time = (
        year.astype(str) + 
        visits_history_copy[col].str[4:]
    )

    visits_history_copy[f"{col}_200"] = pd.to_datetime(
        decrypted_time,
        format="%Y-%m-%dT%H:%M:%SZ",  
        errors="coerce" 
    )

pmh_history_copy = df_pmh.copy()

time_cols = ["Noted_date"]

for col in time_cols:
    year = (
        pmh_history_copy[col].str.extract(r"(\d{4})")[0]
        .dropna()
        .astype(int) - 200
    )
    
    decrypted_time = (
        year.astype(str) + 
        pmh_history_copy[col].str[4:]
    )
    
    pmh_history_copy[f"{col}_200"] = pd.to_datetime(
        decrypted_time,
        format="%Y-%m-%dT%H:%M:%SZ", 
        errors="coerce" 
    )

pmh_history_copy["Noted_date_200"] = pd.to_datetime(pmh_history_copy["Noted_date_200"])

pmh_history_copy['icd_version'] = pmh_history_copy['CodeType'].apply(lambda x: 9 if 'Dx9' in x else 10)

pmh_history_copy['ICD_10_Code'] = pmh_history_copy.apply(
    lambda row: convert_9to10(row['Code']) if row['icd_version'] == 9 else row['Code'],
    axis=1
)
pmh_history_copy['icd_version'] = 10

pmh_history_copy["ICD_9_Code"] = pmh_history_copy.apply(
    lambda row: convert_10to9(row["ICD_10_Code"]), axis=1
)

visits_history_copy['Dx_ICD9_list']  = visits_history_copy['Dx_ICD9'].apply(split_clean_icd)
visits_history_copy['Dx_ICD10_list'] = visits_history_copy['Dx_ICD10'].apply(split_clean_icd)

visits_history_copy['Dx_name_clean'] = visits_history_copy['Dx_name'].fillna('')

visits_history_copy['Dx_name_clean'] = visits_history_copy['Dx_name_clean'].str.lower()

visits_history_copy['Dx_name_clean'] = visits_history_copy['Dx_name_clean'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

for disease_name, disease_def in DISEASE_DEFS.items():
    codes = [c.upper() for c in disease_def['codes']]
    keywords = disease_def['keywords']

    match_icd9 = visits_history_copy['Dx_ICD9_list'].apply(lambda lst: any(any(code and c.startswith(code) for code in codes) for c in lst))
    match_icd10 = visits_history_copy['Dx_ICD10_list'].apply(lambda lst: any(any(code and c.startswith(code) for code in codes) for c in lst))

    if keywords:
        keyword_pattern = '|'.join(keywords)
        match_kw = visits_history_copy['Dx_name_clean'].str.contains(keyword_pattern, case=False, na=False, regex=True)
    else:
        match_kw = False

    match_any = match_icd9 | match_icd10 | match_kw

    hit_ids = set(visits_history_copy.loc[match_any, 'CSN'])

    visits_history_copy[f'outcome_{disease_name}'] = visits_history_copy['CSN'].isin(hit_ids).astype(int)

visits_history_copy["outcome_hospitalization"] = (
    visits_history_copy["Admit_time"].notna() & 
    (visits_history_copy["ED_dispo"] == "Inpatient")
).astype(int)
visits_history_copy["outcome_critical"] = (visits_history_copy["ED_dispo"] == "ICU").astype(int)

visits_history_copy["outcome_discharge"] = (visits_history_copy["ED_dispo"] == "Discharge").astype(int)

visits_history_copy["Hours_to_next_visit"] = pd.to_numeric(visits_history_copy["Hours_to_next_visit"])

visits_history_copy["outcome_ed_revisit_72h"] = ((visits_history_copy["Hours_to_next_visit"] < 72)).astype(int)
visits_history_copy["outcome_ed_revisit_168h"] = ((visits_history_copy["Hours_to_next_visit"] < 168)).astype(int)
visits_history_copy["outcome_ed_revisit_720h"] = ((visits_history_copy["Hours_to_next_visit"] < 720)).astype(int)

outcome_cols_regular = [
    'outcome_sepsis',
    'outcome_copd_exac', 
    'outcome_acs_mi',
    'outcome_stroke',
    'outcome_ards',
    'outcome_aki',
    'outcome_pe',
    'outcome_pneumonia_bacterial',
    'outcome_pneumonia_viral',
    'outcome_pneumonia_all',
    'outcome_asthma_acute_exac',
    'outcome_ahf',
    'outcome_hospitalization',
    'outcome_critical',
    'outcome_discharge',
    'outcome_ed_revisit_72h',
    'outcome_ed_revisit_168h',
    'outcome_ed_revisit_720h'
]

existing_regular_cols = [col for col in outcome_cols_regular if col in visits_history_copy.columns]

merge_columns_regular = ['MRN', 'CSN'] + existing_regular_cols

df_master['subject_id'] = df_master['subject_id'].astype(str)
df_master['stay_id'] = df_master['stay_id'].astype(str)
visits_history_copy['MRN'] = visits_history_copy['MRN'].astype(str)
visits_history_copy['CSN'] = visits_history_copy['CSN'].astype(str)

df_master = df_master.merge(
    visits_history_copy[merge_columns_regular],
    left_on=['subject_id', 'stay_id'],
    right_on=['MRN', 'CSN'],
    how='left'
).drop(columns=['MRN', 'CSN'])

visits_result = apply_outcome_detection(visits_history_copy, pmh_history_copy, DISEASE_DEFS, method='vectorized')

outcome_cols_3d = [col for col in visits_result.columns if col.startswith("outcome_") and col.endswith("_3d")]

conflicting_3d_cols = [col for col in outcome_cols_3d if col in df_master.columns]
if conflicting_3d_cols:
    for col in conflicting_3d_cols:
        print(f"  - {col}")
    df_master = df_master.drop(columns=conflicting_3d_cols)

merge_columns_3d = ['MRN', 'CSN'] + outcome_cols_3d

visits_result['MRN'] = visits_result['MRN'].astype(str)
visits_result['CSN'] = visits_result['CSN'].astype(str)

df_master = df_master.merge(
    visits_result[merge_columns_3d],
    left_on=['subject_id', 'stay_id'],
    right_on=['MRN', 'CSN'],
    how='left'
).drop(columns=['MRN', 'CSN'])

df_master['outcome_icu_transfer_12h'] = (df_master['time_to_icu_transfer_hours'] <= 12).astype(int)

outcome_types = [
    'sepsis', 'copd_exac', 'acs_mi', 'stroke', 'ards', 'aki', 'pe',
    'pneumonia_bacterial', 'pneumonia_viral', 'pneumonia_all',
    'asthma_acute_exac', 'ahf', 'copd_asthma'
]

for outcome_type in outcome_types:
    ed_outcome = f'outcome_{outcome_type}'
    three_day_outcome = f'outcome_{outcome_type}_3d'
    new_outcome = f'outcome_{outcome_type}_3d_new'

    if ed_outcome in df_master.columns and three_day_outcome in df_master.columns:
        df_master[new_outcome] = (df_master[ed_outcome] + df_master[three_day_outcome]).astype(int)

df_master.to_csv(os.path.join(output_path, 'master_dataset.csv'), index=False)


