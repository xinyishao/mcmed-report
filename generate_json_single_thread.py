#!/usr/bin/env python3
"""
Single-threaded JSON generator for affected MRNs
Reads affected_mrns.txt and regenerates JSON files only
"""

import os
import json
import pandas as pd
import numpy as np
import wfdb
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any, Set
import sys

# Import functions from data_preprocessing
sys.path.append('.')
from data_preprocessing import safe_parse_timestamp, process_patient

# Configuration
SEGMENT_DURATION = timedelta(minutes=5)
OUTPUT_DIR = "outputs"
OUTPUT_JSON_DIR = "outputs_json"

# Statistics tracking
stats = {
    "total_mrns": 0,
    "processed_mrns": 0,
    "failed_mrns": 0,
    "total_segments": 0,
    "generated_json": 0,
    "empty_json": 0,
    "failed_json": 0
}

def load_affected_mrns(filename: str = "affected_mrns.txt") -> Set[str]:
    """Load affected MRNs from file"""
    if not os.path.exists(filename):
        print(f"[ERROR] File {filename} not found!")
        return set()
    
    try:
        with open(filename, 'r') as f:
            mrns = {line.strip() for line in f if line.strip()}
        
        print(f"[INFO] Loaded {len(mrns)} affected MRNs from {filename}")
        return mrns
    except Exception as e:
        print(f"[ERROR] Failed to load MRNs from {filename}: {e}")
        return set()

def load_data_files():
    """Load all required data files"""
    print("[INFO] Loading data files...")
    
    try:
        visits = pd.read_csv('data/visits.csv')
        orders = pd.read_csv('data/orders.csv')
        meds = pd.read_csv('data/meds.csv')
        pmh = pd.read_csv('data/pmh.csv')
        labs = pd.read_csv('data/labs.csv')
        rads = pd.read_csv('data/rads.csv')
        numerics = pd.read_csv('data/numerics.csv')
        waveform_summary = pd.read_csv('data/waveform_summary.csv')
        
        print("[SUCCESS] All data files loaded successfully")
        return visits, orders, meds, pmh, labs, rads, numerics, waveform_summary
    except Exception as e:
        print(f"[ERROR] Failed to load data files: {e}")
        return None, None, None, None, None, None, None, None

def process_patient_json_only(mrn, dataframes):
    """Process patient and generate JSON files only (single-threaded)"""
    visits, orders, meds, pmh, labs, rads, numerics, waveform_summary = dataframes
    
    print(f"[INFO] Processing MRN {mrn}...")
    
    try:
        # Filter patient data
        patient_meds = meds[meds['MRN'] == mrn]
        patient_pmh = pmh[pmh['MRN'] == mrn]
        patient_visits = visits[visits['MRN'] == mrn]
        csns = patient_visits['CSN'].unique()
        patient_orders = orders[orders['CSN'].isin(csns)]
        patient_labs = labs[labs['CSN'].isin(csns)]
        patient_rads = rads[rads['CSN'].isin(csns)]
        patient_numerics = numerics[numerics['CSN'].isin(csns)]
        patient_waveforms = waveform_summary[waveform_summary['CSN'].isin(csns)]

        # Sort data
        patient_orders = patient_orders.sort_values('Order_time')
        patient_labs = patient_labs.sort_values('Order_time')
        patient_rads = patient_rads.sort_values('Order_time')
        patient_numerics = patient_numerics.sort_values('Time')
        patient_meds = patient_meds.sort_values('Entry_date')
        patient_pmh = patient_pmh.sort_values('Noted_date')
        patient_visits = patient_visits.sort_values('Roomed_time')

        # Build events
        events = []
        for _, row in patient_labs.iterrows():
            data = {
                'Display_name': row['Display_name'], 'Abnormal': row['Abnormal'],
                'Component_name': row['Component_name'], 'Component_result': row['Component_result'],
                'Component_value': row['Component_value'], 'Component_units': row['Component_units'],
                'Component_abnormal': row['Component_abnormal'], 'Component_nml_low': row['Component_nml_low'],
                'Component_nml_high': row['Component_nml_high']
            }
            events.append({'type': 'lab', 'timestamp': row['Result_time'], 'data': data})
            
        for _, row in patient_numerics.iterrows():
            data = {
                'Source': row['Source'], 'Measure': row['Measure'], 'Value': row['Value']
            }
            events.append({'type': 'numeric', 'timestamp': row['Time'], 'data': data})
            
        for _, row in patient_meds.iterrows():
            data = {
                'Med_ID': row['Med_ID'], 'NDC': row['NDC'], 'Name': row['Name'],
                'Generic_name': row['Generic_name'], 'Med_class': row['Med_class'],
                'Med_subclass': row['Med_subclass'], 'Active': row['Active']
            }
            events.append({'type': 'med', 'timestamp': row['Entry_date'], 'data': data,
                          'parsed_end_timestamp': safe_parse_timestamp(row['End_date'])})
            
        for _, row in patient_pmh.iterrows():
            data = {
                'CodeType': row['CodeType'], 'Code': row['Code'], 'Desc10': row['Desc10'],
                'CCS': row['CCS'], 'DescCCS': row['DescCCS']
            }
            events.append({'type': 'pmh', 'timestamp': row['Noted_date'], 'data': data})
            
        for _, row in patient_visits.iterrows():
            data = {
                "Visit_no": row["Visit_no"], "Visits": row["Visits"], "Age": row["Age"],
                "Gender": row["Gender"], "Race": row["Race"], "Ethnicity": row["Ethnicity"],
                "Means_of_arrival": row["Means_of_arrival"], "Triage_Temp": row["Triage_Temp"],
                "Triage_HR": row["Triage_HR"], "Triage_RR": row["Triage_RR"],
                "Triage_SpO2": row["Triage_SpO2"], "Triage_SBP": row["Triage_SBP"],
                "Triage_DBP": row["Triage_DBP"], "Triage_acuity": row["Triage_acuity"],
                "CC": row["CC"], "ED_dispo": row["ED_dispo"], "ED_LOS": row["ED_LOS"],
                "Hosp_LOS": row["Hosp_LOS"], "DC_dispo": row["DC_dispo"],
                "Payor_class": row["Payor_class"], "Admit_service": row["Admit_service"],
                "Dx_ICD9": row["Dx_ICD9"], "Dx_ICD10": row["Dx_ICD10"],
                "Dx_name": row["Dx_name"], "Hours_to_next_visit": row["Hours_to_next_visit"],
                "Dispo_class_next_visit": row["Dispo_class_next_visit"]
            }
            events.append({'type': 'visit', 'timestamp': row['Roomed_time'], 'data': data,
                          'parsed_end_timestamp': safe_parse_timestamp(row['Departure_time'])})

        # Parse timestamps
        for event in events:
            event['parsed_timestamp'] = safe_parse_timestamp(event['timestamp'])
        events_sorted = sorted(events, key=lambda x: (pd.isna(x['parsed_timestamp']), x['parsed_timestamp']))
        
        print(f"[DEBUG] MRN {mrn}: Built {len(events)} total events")

        # Process waveform segments
        pleth_segments = patient_waveforms[patient_waveforms['Type'] == 'Pleth']
        all_segments = []
        
        print(f"[DEBUG] MRN {mrn}: Found {len(pleth_segments)} Pleth segments in waveform_summary")
        
        if len(pleth_segments) == 0:
            print(f"[WARNING] MRN {mrn}: No Pleth segments found in waveform_summary")
            stats["processed_mrns"] += 1
            return True
        
        for _, row in pleth_segments.iterrows():
            csn = str(row['CSN'])
            segment_number = int(row['Segments'])
            last3 = csn[-3:]
            wfdb_path = os.path.join("data/waveforms", last3, csn, "Pleth", f"{csn}_{segment_number}")
            
            if not os.path.exists(wfdb_path + ".hea") or not os.path.exists(wfdb_path + ".dat"):
                print(f"[WARNING] Missing file: {wfdb_path}, skipping.")
                continue
                
            try:
                record = wfdb.rdrecord(wfdb_path)
                if 'Pleth' not in record.sig_name:
                    print(f"[WARNING] No Pleth channel in: {wfdb_path}, skipping.")
                    continue
                    
                pleth_idx = record.sig_name.index('Pleth')
                fs = record.fs
                ppg_values = record.p_signal[:, pleth_idx]
                scaler = MinMaxScaler()
                ppg_values_normalized = scaler.fit_transform(ppg_values.reshape(-1, 1)).flatten()
                base_datetime = datetime.combine(record.base_date, record.base_time)
                
                # Apply consistent timestamp conversion
                if base_datetime.year >= 2100:
                    base_datetime = base_datetime.replace(year=2000 + (base_datetime.year % 100))
                    
                timestamps = [base_datetime + timedelta(seconds=i / fs) for i in range(len(ppg_values))]
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'ppg_value': ppg_values_normalized
                })
                all_segments.append(df)
                print(f"[DEBUG] MRN {mrn}: Successfully loaded segment {segment_number} from CSN {csn}")
            except Exception as e:
                print(f"[ERROR] Failed to read {wfdb_path}: {e}")

        # Sort segments by timestamp
        all_segments_sorted = sorted(all_segments, key=lambda df: df['timestamp'].iloc[0])
        print(f"[DEBUG] MRN {mrn}: Loaded {len(all_segments_sorted)} valid waveform segments")
        
        # Generate JSON files for each segment
        segment_count = 0
        print(f"[DEBUG] MRN {mrn}: Starting segment generation from {len(all_segments_sorted)} segments")
        
        for segment_df in all_segments_sorted:
            df = segment_df.sort_values("timestamp").reset_index(drop=True)
            earliest_ts = df['timestamp'].min()
            latest_ts = df['timestamp'].max()

            print(f"[DEBUG] MRN {mrn}: Processing segment from {earliest_ts} to {latest_ts}")

            segment_start = earliest_ts
            while segment_start < latest_ts:
                segment_end = segment_start + SEGMENT_DURATION
                window_df = df[(df['timestamp'] >= segment_start) & (df['timestamp'] < segment_end)]
                
                if not window_df.empty:
                    segment_count += 1
                    stats["total_segments"] += 1
                    
                    print(f"[DEBUG] MRN {mrn}: Generating JSON for segment {segment_count} ({segment_start} to {segment_end})")
                    
                    # Generate JSON for this segment
                    success = generate_json_for_segment(mrn, segment_count, segment_start, segment_end, window_df, events_sorted)
                    if success:
                        stats["generated_json"] += 1
                    else:
                        stats["failed_json"] += 1
                
                segment_start = segment_end

        print(f"[SUCCESS] MRN {mrn}: Generated {segment_count} JSON segments")
        stats["processed_mrns"] += 1
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed processing MRN {mrn}: {e}")
        stats["failed_mrns"] += 1
        return False

def generate_json_for_segment(mrn, segment_id, segment_start, segment_end, window_df, events_sorted):
    """Generate JSON file for a single segment"""
    try:
        timestamp_str = segment_start.strftime("%Y-%m-%d-%H-%M")
        
        # Build JSON events
        json_events = {}
        events_in_window = 0
        
        for event in events_sorted:
            event_type = event.get("type")
            ts = event.get("parsed_timestamp")
            ts_end = event.get("parsed_end_timestamp") or ts
            
            if ts is None:
                continue
                
            if event_type == "pmh":
                if ts <= segment_end:
                    json_events.setdefault(event_type, []).append(event['data'])
                    events_in_window += 1
            elif event_type in ["med", "visit"]:
                if ts <= segment_end and ts_end >= segment_start:
                    json_events.setdefault(event_type, []).append(event['data'])
                    events_in_window += 1
            else:
                if segment_start <= ts < segment_end:
                    json_events.setdefault(event_type, []).append(event['data'])
                    events_in_window += 1
        
        # Write JSON file
        json_filename = f"p{mrn}-{timestamp_str}_ppg_{segment_id}.json"
        json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
        
        with open(json_path, "w") as f:
            json.dump(json_events, f, indent=4, default=str)
        
        # Verify JSON file was created correctly
        if verify_json_file(json_path, json_events):
            if len(json_events) == 0:
                print(f"[WARNING] Empty JSON generated: {json_filename}")
                stats["empty_json"] += 1
            else:
                print(f"[SUCCESS] JSON generated: {json_filename} ({events_in_window} events)")
            return True
        else:
            print(f"[ERROR] JSON verification failed: {json_filename}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to generate JSON for segment {segment_id}: {e}")
        return False

def verify_json_file(json_path, expected_data):
    """Verify that JSON file was created correctly"""
    try:
        if not os.path.exists(json_path):
            return False
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check if data matches expected
        return len(data) == len(expected_data)
    except Exception:
        return False

def main():
    print("=== Single-Threaded JSON Generator ===")
    print("This script will regenerate JSON files for affected MRNs only")
    print()
    
    # Load affected MRNs
    affected_mrns = load_affected_mrns()
    if not affected_mrns:
        print("[ERROR] No affected MRNs found!")
        return
    
    stats["total_mrns"] = len(affected_mrns)
    
    # Load data files
    dataframes = load_data_files()
    if any(df is None for df in dataframes):
        print("[ERROR] Failed to load data files!")
        return
    
    # Process each MRN sequentially
    print(f"\n[INFO] Processing {len(affected_mrns)} MRNs sequentially...")
    for i, mrn in enumerate(sorted(affected_mrns), 1):
        print(f"\n--- Processing MRN {i}/{len(affected_mrns)}: {mrn} ---")
        process_patient_json_only(mrn, dataframes)
    
    # Print final statistics
    print("\n=== FINAL STATISTICS ===")
    print(f"Total MRNs: {stats['total_mrns']}")
    print(f"Processed MRNs: {stats['processed_mrns']}")
    print(f"Failed MRNs: {stats['failed_mrns']}")
    print(f"Total segments: {stats['total_segments']}")
    print(f"Generated JSON files: {stats['generated_json']}")
    print(f"Empty JSON files: {stats['empty_json']}")
    print(f"Failed JSON files: {stats['failed_json']}")
    
    # Print statistics (no file generation)
    print(f"\n=== JSON Generation Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if stats['generated_json'] > 0:
        print(f"[SUCCESS] Generated {stats['generated_json']} JSON files!")
    if stats['empty_json'] > 0:
        print(f"[WARNING] {stats['empty_json']} JSON files are empty")
    if stats['failed_json'] > 0:
        print(f"[ERROR] {stats['failed_json']} JSON files failed to generate")

if __name__ == "__main__":
    main()
