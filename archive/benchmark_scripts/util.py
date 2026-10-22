from datetime import timedelta
import pandas as pd
import re
import numpy as np
from mappers.icd9to10_dict import icd9to10dict
from mappers.icd10to9_dict import icd10to9dict
from mappers.elixhauser_charlson import *

def split_clean_icd(cell):
  if pd.isna(cell):
    return []
  parts = re.split(r'[,;\s]+', str(cell))
  return [p.replace('.', '').strip().upper() for p in parts if p.strip()]

def adjust_time_by_subtracting_year(time_series):
  s = time_series.astype("string")
  year = pd.to_numeric(s.str[:4], errors='coerce')
  month_day_time = s.str[4:]
  new_year = year - 200
  new_time_str = new_year.astype('Int64').astype('string') + month_day_time
  new_time_str = new_time_str.where(s.notna(), pd.NA)
  return pd.to_datetime(new_time_str, errors='coerce').dt.tz_localize(None)

def encode_chief_complaints(df_master, complaint_dict):
    holder_list = []
    complaint_colnames_list = list(complaint_dict.keys())
    complaint_regex_list = list(complaint_dict.values())

    for _, row in df_master.iterrows():
        curr_patient_complaint = str(row['CC'])
        curr_patient_complaint_list = [0] * len(complaint_regex_list)

        for idx, complaint in enumerate(complaint_regex_list):
            if re.search(complaint, curr_patient_complaint, re.IGNORECASE):
                curr_patient_complaint_list[idx] = 1

        holder_list.append(curr_patient_complaint_list)

    df_encoded_complaint = pd.DataFrame(holder_list, columns=complaint_colnames_list)
    df_master = pd.concat([df_master, df_encoded_complaint], axis=1)
    return df_master

def build_comorbidity_code_dictionary():
  
    comorbidity_codes = {}
    
    code_sources = [
        (charlson_codes_v9, 'icd9'),
        (charlson_codes_v10, 'icd10'),
        (elixhauser_codes_v9, 'icd9'),
        (elixhauser_codes_v10, 'icd10')
    ]
    
    for code_dict, icd_version in code_sources:
        for condition_name, code_list in code_dict.items():
            if condition_name not in comorbidity_codes:
                comorbidity_codes[condition_name] = {}
            comorbidity_codes[condition_name][icd_version] = {'starts': code_list}
    
    return comorbidity_codes

def calculate_visit_history(visits_df):
    visits_sorted = visits_df.sort_values(['MRN', 'Arrival_time']).reset_index(drop=True)
    results = []
    
    for mrn, patient_visits in visits_sorted.groupby('MRN'):
        patient_visits = patient_visits.sort_values('Arrival_time').reset_index(drop=True)
        
        for i, (_, current_visit) in enumerate(patient_visits.iterrows()):
            previous_visits = patient_visits.iloc[:i]
            
            n_ed_30d = n_ed_90d = n_ed_365d = 0
            n_hosp_30d = n_hosp_90d = n_hosp_365d = 0
            n_icu_30d = n_icu_90d = n_icu_365d = 0
            
            if len(previous_visits) > 0:
                time_diffs = (current_visit['Arrival_time'] - previous_visits['Arrival_time']).dt.days
                
                n_ed_30d = (time_diffs <= 30).sum()
                n_ed_90d = (time_diffs <= 90).sum()
                n_ed_365d = (time_diffs <= 365).sum()
                
                hospital_mask = previous_visits['ED_dispo'] == 'Inpatient'
                if hospital_mask.any():
                    hosp_time_diffs = time_diffs[hospital_mask]
                    n_hosp_30d = (hosp_time_diffs <= 30).sum()
                    n_hosp_90d = (hosp_time_diffs <= 90).sum()
                    n_hosp_365d = (hosp_time_diffs <= 365).sum()
                
                icu_mask = previous_visits['ED_dispo'] == 'ICU'
                if icu_mask.any():
                    icu_time_diffs = time_diffs[icu_mask]
                    n_icu_30d = (icu_time_diffs <= 30).sum()
                    n_icu_90d = (icu_time_diffs <= 90).sum()
                    n_icu_365d = (icu_time_diffs <= 365).sum()
            
            results.append({
                'MRN': current_visit['MRN'],
                'CSN': current_visit['CSN'],
                'n_ed_30d': n_ed_30d,
                'n_ed_90d': n_ed_90d,
                'n_ed_365d': n_ed_365d,
                'n_hosp_30d': n_hosp_30d,
                'n_hosp_90d': n_hosp_90d,
                'n_hosp_365d': n_hosp_365d,
                'n_icu_30d': n_icu_30d,
                'n_icu_90d': n_icu_90d,
                'n_icu_365d': n_icu_365d
            })
    
    return pd.DataFrame(results)


def calculate_medication_history(visits_df, meds_df):
    results = []
    
    for _, visit in visits_df.iterrows():
        current_mrn = visit['MRN']
        current_arrival_time = visit['Arrival_time']
        
        n_med = n_medrecon = 0

        if current_mrn in meds_df.index:
            patient_meds = meds_df.loc[current_mrn]
            
            if isinstance(patient_meds, pd.Series):
                patient_meds = patient_meds.to_frame().T
            
            five_years_prior = current_arrival_time - timedelta(days=5*365)
            
            start_mask = (patient_meds['Start_date'] >= five_years_prior) & (patient_meds['Start_date'] <= current_arrival_time)
            end_mask = (patient_meds['End_date'] >= five_years_prior) & (patient_meds['End_date'] <= current_arrival_time)
            
            if start_mask.any():
                n_med = patient_meds.loc[start_mask, 'Generic_name'].nunique()
            
            n_medrecon = start_mask.sum() + end_mask.sum()
        
        results.append({
            'MRN': visit['MRN'],
            'CSN': visit['CSN'],
            'n_med': n_med,
            'n_medrecon': n_medrecon
        })
    
    return pd.DataFrame(results)

def calculate_comorbidities_batch(visits_df, pmh_df, comorbidity_codes):
    pmh_processed = pmh_df.copy()
    pmh_processed['Code'] = pmh_processed['Code'].astype(str)
    pmh_processed = pmh_processed.sort_values(['MRN', 'Noted_date']).reset_index(drop=True)
    
    pmh_grouped = pmh_processed.groupby('MRN')
    
    compiled_patterns = {}
    for condition, codes in comorbidity_codes.items():
        compiled_patterns[condition] = {
            'icd9_starts': tuple(codes['icd9']['starts']) if 'icd9' in codes and 'starts' in codes['icd9'] else (),
            'icd10_starts': tuple(codes['icd10']['starts']) if 'icd10' in codes and 'starts' in codes['icd10'] else ()
        }
    
    results = {condition: np.zeros(len(visits_df), dtype=np.int8) for condition in comorbidity_codes.keys()}
    results['MRN'] = visits_df['MRN'].values
    results['CSN'] = visits_df['CSN'].values
    
    visits_sorted = visits_df.sort_values(['MRN', 'Arrival_time']).reset_index(drop=True)
    
    for mrn, patient_visits in visits_sorted.groupby('MRN'):
        try:
            patient_pmh = pmh_grouped.get_group(mrn)
        except KeyError:
            continue
        
        patient_icd9 = patient_pmh[patient_pmh['CodeType'] == 'Dx9']
        patient_icd10 = patient_pmh[patient_pmh['CodeType'] == 'Dx10']
        
        for visit_idx, (_, visit) in enumerate(patient_visits.iterrows()):
            current_arrival_time = visit['Arrival_time']
            original_idx = visit.name
            
            valid_icd9_mask = patient_icd9['Noted_date'] <= current_arrival_time
            valid_icd10_mask = patient_icd10['Noted_date'] <= current_arrival_time
            
            if not valid_icd9_mask.any() and not valid_icd10_mask.any():
                continue
            
            valid_icd9_codes = patient_icd9.loc[valid_icd9_mask, 'Code'].values if valid_icd9_mask.any() else np.array([])
            valid_icd10_codes = patient_icd10.loc[valid_icd10_mask, 'Code'].values if valid_icd10_mask.any() else np.array([])
            
            for condition, patterns in compiled_patterns.items():
                condition_found = False
                
                if patterns['icd9_starts'] and len(valid_icd9_codes) > 0:
                    for code in valid_icd9_codes:
                        if code.startswith(patterns['icd9_starts']):
                            condition_found = True
                            break
                
                if not condition_found and patterns['icd10_starts'] and len(valid_icd10_codes) > 0:
                    for code in valid_icd10_codes:
                        if code.startswith(patterns['icd10_starts']):
                            condition_found = True
                            break
                
                results[condition][original_idx] = 1 if condition_found else 0
    
    return pd.DataFrame(results)

def convert_9to10(code):
    if code in icd9to10dict.keys():
        return icd9to10dict[code]
    else:
        return code
    
def convert_10to9(code, digit3=False):
    if code in icd10to9dict.keys():
        output = icd10to9dict[code]
    else:
        output = code
    if digit3:
        output = output[:3]
    return output
    
def normalize_lab_test_names(df, test_col):
    """
    Normalize lab test names by combining similar tests into standardized groups.
    """
    df = df.copy()
    df['normalized_test'] = df[test_col]
    
    normalization_rules = {
        'PCO2, Arterial (Combined)': ['PCO2 (A)', 'POC:PCO2 (A), ISTAT', 'PCO2.ARTERIAL'],
        'PCO2, Venous (Combined)': ['PCO2.VENOUS', 'POC:PCO2 (V), ISTAT'],
        'pH, Arterial (Combined)': ['PH.ARTERIAL', 'POC:PH (A), ISTAT'],
        'pH, Venous (Combined)': ['PH.VENOUS', 'POC:PH (V), ISTAT'],
        'PO2, Arterial (Combined)': ['PO2 (A)', 'POC:PO2 (A), ISTAT', 'PO2.ARTERIAL'],
        'PO2, Venous (Combined)': ['PO2.VENOUS', 'POC:PO2 (V), ISTAT'],
        
        'Potassium (Combined)': ['POTASSIUM', 'POC:POTASSIUM, ISTAT', 'POTASSIUM.WHOLE BLD'],
        'Sodium (Combined)': ['SODIUM', 'POC:SODIUM, ISTAT', 'SODIUM.WHOLE BLD'],
        
        'Calcium (Combined)': r'CALCIUM',
        'eGFR (Combined)': r'EGFR',
        'LDL (Combined)': r'LDL',
        'Cholesterol, HDL (Combined)': r'CHOLESTEROL.*HDL',
        'Bilirubin, Unconjugated (Combined)': r'BILIRUBIN, UNCONJ',
        
        'HCT & Hemoglobin (Combined)': r'^HCT|HEMOGLOBIN',
        'ESR (Combined)': r'ESR',
        'NRBC (Combined)': r'NRBC',
        'Blasts (Combined)': r'BLASTS',
        
        'Basophil % (Combined)': r'BASOPHILS? \s* % \s* \( (AUTO|MANUAL) DIFF \)',
        'Basophil Absolute (Combined)': r'BASOPHILS?.*ABSOLUTE',
        'Eosinophil % (Combined)': r'EOSINOPHILS? \s* % \s* \( (AUTO|MANUAL) DIFF \)',
        'Eosinophil Absolute (Combined)': r'EOSINOPHILS?.*ABSOLUTE',
        'Monocyte % (Combined)': r'MONOCYTES? \s* % \s* \( (AUTO|MANUAL) DIFF \)',
        'Monocyte Absolute (Combined)': r'MONOCYTE.*ABSOLUTE',
        'Lymphocyte Absolute (Combined)': r'LYMPHOCYTES?.*ABSOLUTE',
        
        'Blood Typing & RH (Combined)': r'abo/rh|blood type verification',
        'Antibody Screen (Combined)': r'ANTIBODY SCREEN',
        'Hepatitis B Surface Antigen (Combined)': r'HBSAG|HEP B SURFACE',
        'Anti-HAV IgG (Combined)': r'ANTI-HAV IGG',
        'Anti-HBc (Combined)': r'ANTI-HBC',
        'Anti-HCV (Combined)': r'ANTI-HCV',
        
        'Adenovirus (Combined)': r'ADENOVIRUS',
        'Rhinovirus (Combined)': r'RHINOVIRUS',
        'Escherichia Coli (Combined)': r'ESCHERICHIA COLI',
        
        'Normal RBC Count (Combined)': r'NORMAL RBC ON .* COUNT',
        "RBC's in Small Square (Combined)": r"RBC'S IN SMALL SQUARE",
        'Basophil CSF (Combined)': r'BASOPHILS?.*CSF'
    }
    
    for normalized_name, patterns in normalization_rules.items():
        if isinstance(patterns, list):
            mask = df[test_col].isin(patterns)
        else:
            mask = df[test_col].str.contains(patterns, case=False, na=False, regex=True)
        
        df.loc[mask, 'normalized_test'] = normalized_name
    
    return df

def process_lab_data(df_labs, min_frequency=10000):
    """
    Process lab data: clean values, normalize test names, and filter by frequency.
    """
    df_processed = df_labs.copy()
    
    df_processed['Component_name'] = df_processed['Component_name'].str.strip()
    df_processed['Component_value'] = (df_processed['Component_value']
                                       .astype(str)
                                       .str.extract(r'(\d+\.?\d*)', expand=False))
    df_processed['Component_value'] = pd.to_numeric(df_processed['Component_value'], errors='coerce')
    
    df_processed = df_processed.dropna(subset=['Component_value'])
    
    df_processed = normalize_lab_test_names(df_processed, 'Component_name')
    
    test_counts = df_processed['normalized_test'].value_counts()
    high_frequency_tests = test_counts[test_counts > min_frequency].index.tolist()
    
    print(f"Found {len(high_frequency_tests)} tests with > {min_frequency} occurrences.")
    
    return df_processed[df_processed['normalized_test'].isin(high_frequency_tests)]

def merge_lab_data_with_visits(df_patient_visits, df_labs_wide, join_col='CSN'):
    """
    Merge lab data with patient visits, handling column conflicts.
    """
    lab_cols_to_add = df_labs_wide.columns.drop(join_col)
    
    conflicting_cols = [col for col in lab_cols_to_add if col in df_patient_visits.columns]
    if conflicting_cols:
        print(f"Warning: Dropping {len(conflicting_cols)} conflicting columns from master dataset.")
        df_patient_visits = df_patient_visits.drop(columns=conflicting_cols)
    
    df_merged = pd.merge(
        df_patient_visits, 
        df_labs_wide, 
        left_on='stay_id', 
        right_on=join_col, 
        how='left'
    )
    
    return df_merged.drop(columns=[join_col], errors='ignore')

def check_visit_outcome(visit_row, pmh_indexed, codes, keywords, discharge_col):
    discharge_time = visit_row[discharge_col]
    mrn = str(visit_row['MRN'])
    
    if pd.isna(discharge_time):
        return 0
    
    if mrn not in pmh_indexed.index:
        return 0

    try:
        patient_pmh = pmh_indexed.loc[[mrn]]
    except KeyError:
        return 0
    
    if patient_pmh.empty:
        return 0
    
    time_diff_hours = (patient_pmh['Noted_date_200'] - discharge_time).dt.total_seconds() / 3600
    
    records_in_window = patient_pmh[(time_diff_hours >= 0) & (time_diff_hours <= 72)]
    
    if records_in_window.empty:
        return 0
    
    code_match = False
    if codes:
        if 'ICD_9_Code' in records_in_window.columns:
            icd9_matches = records_in_window['ICD_9_Code'].astype(str).str.upper().str.startswith(tuple(codes), na=False)
            if icd9_matches.any():
                code_match = True
        
        if 'ICD_10_Code' in records_in_window.columns and not code_match:
            icd10_matches = records_in_window['ICD_10_Code'].astype(str).str.upper().str.startswith(tuple(codes), na=False)
            if icd10_matches.any():
                code_match = True
    
    keyword_match = False
    if keywords and 'Desc10' in records_in_window.columns:
        keyword_pattern = '|'.join(keywords)
        keyword_matches = records_in_window['Desc10'].astype(str).str.lower().str.contains(
            keyword_pattern, regex=True, na=False
        )
        if keyword_matches.any():
            keyword_match = True
    
    return int(code_match or keyword_match)

def preprocess_data(visits_df, pmh_df):
   
    visits_processed = visits_df.copy()
    pmh_processed = pmh_df.copy()
    
    discharge_time_candidates = ['Departure_time_200', 'discharge_time', 'outtime', 'end_time', 'Arrival_time']
    discharge_col = None
    for candidate in discharge_time_candidates:
        if candidate in visits_processed.columns:
            discharge_col = candidate
            break
    
    if discharge_col is None:
        raise KeyError(f"Discharge time column not found. Please check column names. Available columns: {list(visits_processed.columns)}")
    
    visits_processed[discharge_col] = pd.to_datetime(visits_processed[discharge_col])
    pmh_processed['Noted_date_200'] = pd.to_datetime(pmh_processed['Noted_date_200'])
    
    visits_processed['MRN'] = visits_processed['MRN'].astype(str)
    pmh_processed['MRN'] = pmh_processed['MRN'].astype(str)
    
    pmh_indexed = pmh_processed.set_index('MRN')
    
    diagnosis_columns = ['ICD_9_Code', 'ICD_10_Code', 'Desc10']
    for col in diagnosis_columns:
        if col in pmh_indexed.columns:
            pmh_indexed[col] = pmh_indexed[col].fillna('').astype(str)
    
    return visits_processed, pmh_indexed, discharge_col

def vectorized_outcome_detection(visits_df, pmh_indexed, disease_defs, discharge_col):
    
    results = {}
    total_visits = len(visits_df)

    available_patients = set(pmh_indexed.index)
    visits_patients = set(visits_df['MRN'].astype(str))
    overlap_patients = available_patients.intersection(visits_patients)
    
    for disease, disease_def in disease_defs.items():
    
        codes = [str(c).upper() for c in disease_def.get("codes", [])] if disease_def.get("codes") else []
        keywords = [str(k).lower() for k in disease_def.get("keywords", [])] if disease_def.get("keywords") else []
        
        disease_results = []
        positive_count = 0
        
        for idx, visit_row in visits_df.iterrows():
            result = check_visit_outcome(visit_row, pmh_indexed, codes, keywords, discharge_col)
            disease_results.append(result)
            if result == 1:
                positive_count += 1
        
        results[disease] = disease_results
    
    return results

def merge_based_approach(visits_df, pmh_indexed, disease_defs, discharge_col):
    pmh_reset = pmh_indexed.reset_index()
    
    visits_work = visits_df[['MRN', discharge_col]].copy().reset_index()
    visits_work['visit_index'] = visits_work.index
    visits_work['MRN'] = visits_work['MRN'].astype(str)
    
    merged = visits_work.merge(pmh_reset, on='MRN', how='left')
    
    merged['time_diff_hours'] = (merged['Noted_date_200'] - merged[discharge_col]).dt.total_seconds() / 3600
    
    valid_records = merged[
        (merged['time_diff_hours'] >= 0) & 
        (merged['time_diff_hours'] <= 72) &
        (merged['time_diff_hours'].notna())
    ].copy()
    
    results = {}
 
    for disease, disease_def in disease_defs.items():
        
        codes = [str(c).upper() for c in disease_def.get("codes", [])] if disease_def.get("codes") else []
        keywords = [str(k).lower() for k in disease_def.get("keywords", [])] if disease_def.get("keywords") else []
        
        if valid_records.empty:
            results[disease] = [0] * len(visits_df)
            continue
        
        code_match_mask = pd.Series(False, index=valid_records.index)
        if codes:
            if 'ICD_9_Code' in valid_records.columns:
                icd9_match = valid_records['ICD_9_Code'].astype(str).str.upper().str.startswith(tuple(codes), na=False)
                code_match_mask |= icd9_match
            
            if 'ICD_10_Code' in valid_records.columns:
                icd10_match = valid_records['ICD_10_Code'].astype(str).str.upper().str.startswith(tuple(codes), na=False)
                code_match_mask |= icd10_match
        
        keyword_match_mask = pd.Series(False, index=valid_records.index)
        if keywords and 'Desc10' in valid_records.columns:
            keyword_pattern = '|'.join(keywords)
            keyword_match_mask = valid_records['Desc10'].astype(str).str.lower().str.contains(
                keyword_pattern, regex=True, na=False
            )
        
        overall_match = code_match_mask | keyword_match_mask
        
        visit_matches = valid_records.loc[overall_match, 'visit_index'].unique() if overall_match.any() else []
        
        final_results = [0] * len(visits_df)
        for visit_idx in visit_matches:
            final_results[visit_idx] = 1
        
        results[disease] = final_results
    
    return results

def apply_outcome_detection(visits_df, pmh_df, disease_defs, method='vectorized'):
    
    visits_processed, pmh_indexed, discharge_col = preprocess_data(visits_df, pmh_df)
    
    if method == 'merge':
        results = merge_based_approach(visits_processed, pmh_indexed, disease_defs, discharge_col)
    else:
        results = vectorized_outcome_detection(visits_processed, pmh_indexed, disease_defs, discharge_col)
    
    visits_result = visits_df.copy()
    for disease, disease_results in results.items():
        column_name = f"outcome_{disease}_3d"
        visits_result[column_name] = disease_results
    
    return visits_result

