import os
import json
import asyncio
import httpx
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import wfdb
import shutil
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Disable proxy for localhost connections
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

SEGMENT_DURATION = timedelta(minutes=5)
OUTPUT_DIR = "outputs"
OUTPUT_JSON_DIR = "outputs_json"
OUTPUT_LLM_DIR = "outputs_Llama"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_LLM_DIR, exist_ok=True)

# vLLM constants
ENDPOINTS = [
    "http://localhost:8000"
]
MODEL_NAME = "/gemini/platform/public/aigc/Lirui/chengding/Meta-Llama-3.1-8B-Instruct"
MAX_CONCURRENT = 4  # Optimized for tensor-parallel 4-GPU setup
GENERATION_WORKERS = 8  # threads for NPZ/JSON generation

DEBUG = True

# Control flags
TEST_VLLM_CONNECTION = True   # Set to False to skip vLLM connection test
USE_SYNC_LLM_PROCESSING = True  # Set to True for reliable synchronous processing
USE_SINGLE_THREAD = False      # Set to True to use single-threaded processing (no ThreadPoolExecutor)

# Background LLM execution state
_ASYNC_LOOP = None
_LOOP_THREAD = None
_LLM_CLIENT = None
_LLM_FUTURES = []
_FUT_LOCK = threading.Lock()

def safe_parse_timestamp(ts):
    try:
        if pd.isna(ts) or ts == '':
            if DEBUG:
                print(f"[DEBUG] safe_parse_timestamp: Invalid input: {ts}")
            return pd.NaT
            
        # Handle timezone-aware timestamps (with 'Z' suffix)
        if isinstance(ts, str) and ts.endswith('Z'):
            # Remove 'Z' and parse
            ts_clean = ts[:-1]
            # Apply consistent anonymization: convert ALL future timestamps (21xx, 22xx, 23xx) to 20xx
            if ts_clean.startswith(('21', '22', '23')):
                # Assume this is anonymized data - convert 21xx/22xx/23xx to 20xx
                ts_adjusted = '20' + ts_clean[2:]
                result = pd.to_datetime(ts_adjusted)
                if DEBUG:
                    print(f"[DEBUG] safe_parse_timestamp: Converted {ts} -> {ts_adjusted} -> {result}")
                return result
            else:
                result = pd.to_datetime(ts_clean)
                if DEBUG:
                    print(f"[DEBUG] safe_parse_timestamp: Parsed {ts} -> {ts_clean} -> {result}")
                return result
        else:
            # For non-Z timestamps, also check for future years
            if isinstance(ts, str) and ts.startswith(('21', '22', '23')):
                ts_adjusted = '20' + ts[2:]
                result = pd.to_datetime(ts_adjusted)
                if DEBUG:
                    print(f"[DEBUG] safe_parse_timestamp: Converted {ts} -> {ts_adjusted} -> {result}")
                return result
            else:
                result = pd.to_datetime(ts)
                if DEBUG:
                    print(f"[DEBUG] safe_parse_timestamp: Parsed {ts} -> {result}")
                return result
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] safe_parse_timestamp: Failed to parse {ts}: {e}")
        return pd.NaT

def process_patient(mrn, dataframes):
    visits, orders, meds, pmh, labs, rads, numerics, waveform_summary = dataframes
    patient_meds = meds[meds['MRN'] == mrn]
    patient_pmh = pmh[pmh['MRN'] == mrn]
    patient_visits = visits[visits['MRN'] == mrn]
    csns = patient_visits['CSN'].unique()
    patient_orders = orders[orders['CSN'].isin(csns)]
    patient_labs = labs[labs['CSN'].isin(csns)]
    patient_rads = rads[rads['CSN'].isin(csns)]
    patient_numerics = numerics[numerics['CSN'].isin(csns)]
    patient_waveforms = waveform_summary[waveform_summary['CSN'].isin(csns)]

    # Sort
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
            "Visit_no": row["Visit_no"],
            "Visits": row["Visits"],
            "Age": row["Age"],
            "Gender": row["Gender"],
            "Race": row["Race"],
            "Ethnicity": row["Ethnicity"],
            "Means_of_arrival": row["Means_of_arrival"],
            "Triage_Temp": row["Triage_Temp"],
            "Triage_HR": row["Triage_HR"],
            "Triage_RR": row["Triage_RR"],
            "Triage_SpO2": row["Triage_SpO2"],
            "Triage_SBP": row["Triage_SBP"],
            "Triage_DBP": row["Triage_DBP"],
            "Triage_acuity": row["Triage_acuity"],
            "CC": row["CC"],
            "ED_dispo": row["ED_dispo"],
            "ED_LOS": row["ED_LOS"],
            "Hosp_LOS": row["Hosp_LOS"],
            "DC_dispo": row["DC_dispo"],
            "Payor_class": row["Payor_class"],
            "Admit_service": row["Admit_service"],
            "Dx_ICD9": row["Dx_ICD9"],
            "Dx_ICD10": row["Dx_ICD10"],
            "Dx_name": row["Dx_name"],
            "Hours_to_next_visit": row["Hours_to_next_visit"],
            "Dispo_class_next_visit": row["Dispo_class_next_visit"]
        }
        events.append({'type': 'visit', 'timestamp': row['Roomed_time'], 'data': data,
                      'parsed_end_timestamp': safe_parse_timestamp(row['Departure_time'])})
    for event in events:
        event['parsed_timestamp'] = safe_parse_timestamp(event['timestamp'])
    events_sorted = sorted(events, key=lambda x: (pd.isna(x['parsed_timestamp']), x['parsed_timestamp']))
    
    # Debug: Log event summary
    if DEBUG:
        print(f"[DEBUG] MRN {mrn}: Built {len(events)} total events")
        event_counts = {}
        for event in events_sorted:
            event_type = event.get("type")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        print(f"[DEBUG] MRN {mrn}: Event breakdown: {event_counts}")
    
    # Debug: Log timestamp ranges
    if DEBUG:
        valid_events = [e for e in events_sorted if not pd.isna(e['parsed_timestamp'])]
        if valid_events:
            min_ts = min(e['parsed_timestamp'] for e in valid_events)
            max_ts = max(e['parsed_timestamp'] for e in valid_events)
            print(f"[DEBUG] MRN {mrn}: Event timestamp range: {min_ts} to {max_ts}")
        else:
            print(f"[WARNING] MRN {mrn}: No events with valid timestamps!")

    # Waveform processing
    pleth_segments = patient_waveforms[patient_waveforms['Type'] == 'Pleth']
    all_segments = []
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
            # Apply consistent timestamp conversion to waveform timestamps
            if base_datetime.year >= 2100:
                # Convert ALL future years (21xx, 22xx, 23xx) to 20xx for consistency with event timestamps
                base_datetime = base_datetime.replace(year=2000 + (base_datetime.year % 100))
            timestamps = [base_datetime + timedelta(seconds=i / fs) for i in range(len(ppg_values))]
            df = pd.DataFrame({
                'timestamp': timestamps,
                'ppg_value': ppg_values_normalized
            })
            all_segments.append(df)
        except Exception as e:
            print(f"[ERROR] Failed to read {wfdb_path}: {e}")
    
    
    # Build reports
    all_segments_sorted = sorted(all_segments, key=lambda df: df['timestamp'].iloc[0])

    def _write_segment_files(mrn_local, seg_id_local, seg_start_local, seg_end_local, window_df_local, events_sorted_local):
        try:
            timestamp_str_local = seg_start_local.strftime("%Y-%m-%d-%H-%M")
            # Save NPZ
            npz_filename_local = f"p{mrn_local}-{timestamp_str_local}_ppg_{seg_id_local}.npz"
            np.savez(os.path.join(OUTPUT_DIR, npz_filename_local), **window_df_local.to_dict('list'))
            
            # Build JSON events 
            json_events_local = {}
            events_in_window = 0
            total_events = len(events_sorted_local)
            
            for event in events_sorted_local:
                event_type = event.get("type")
                ts = event.get("parsed_timestamp")
                ts_end = event.get("parsed_end_timestamp") or ts
                
                if ts is None:
                    continue
                    
                if event_type == "pmh":
                    if ts <= seg_end_local:
                        json_events_local.setdefault(event_type, []).append(event['data'])
                        events_in_window += 1
                elif event_type in ["med", "visit"]:
                    if ts <= seg_end_local and ts_end >= seg_start_local:
                        json_events_local.setdefault(event_type, []).append(event['data'])
                        events_in_window += 1
                else:
                    if seg_start_local <= ts < seg_end_local:
                        json_events_local.setdefault(event_type, []).append(event['data'])
                        events_in_window += 1
            
            # Check if JSON is empty and log details
            if not json_events_local:
                print(f"[WARNING] Empty JSON for MRN {mrn_local}, seg {seg_id_local}!")
                print(f"[WARNING] Window: {seg_start_local} to {seg_end_local}")
                print(f"[WARNING] Total events available: {total_events}")
                print(f"[WARNING] Events in window: {events_in_window}")
                
            else:
                print(f"[INFO] MRN {mrn_local}, seg {seg_id_local}: Added {events_in_window} events to JSON")
            
            # Write JSON
            json_filename_local = f"p{mrn_local}-{timestamp_str_local}_ppg_{seg_id_local}.json"
            with open(os.path.join(OUTPUT_JSON_DIR, json_filename_local), "w") as f:
                json.dump(json_events_local, f, indent=4, default=str)
            
            # Process LLM report (always generate reports)
            if json_events_local:
                if USE_SYNC_LLM_PROCESSING:
                    # Use synchronous processing for reliability
                    success = process_llm_report_sync(json_events_local, json_filename_local)
                    if not success:
                        print(f"[ERROR] Failed to generate LLM report synchronously: {json_filename_local}")
                else:
                    # Use async processing (original method)
                    future_local = submit_llm_job(json_events_local, json_filename_local)
                    if future_local is not None:
                        with _FUT_LOCK:
                            _LLM_FUTURES.append(future_local)
            else:
                print(f"[WARNING] Skipping LLM job for empty JSON: {json_filename_local}")
                
            return True
        except Exception as e:
            print(f"[ERROR] Segment write failed for MRN {mrn_local}, seg {seg_id_local}: {e}")
            import traceback
            traceback.print_exc()
            return False

    # First pass: count windows
    total_windows = 0
    for segment_df in all_segments_sorted:
        df = segment_df.sort_values("timestamp").reset_index(drop=True)
        segment_start = df['timestamp'].min()
        segment_end_all = df['timestamp'].max()
        while segment_start < segment_end_all:
            segment_end = segment_start + SEGMENT_DURATION
            window_df = df[(df['timestamp'] >= segment_start) & (df['timestamp'] < segment_end)]
            if not window_df.empty:
                total_windows += 1
            segment_start = segment_end

    # Second pass: submit tasks
    if USE_SINGLE_THREAD:
        # Single-threaded processing
        segment_id = 1
        for segment_df in all_segments_sorted:
            df = segment_df.sort_values("timestamp").reset_index(drop=True)
            earliest_ts = df['timestamp'].min()
            latest_ts = df['timestamp'].max()

            segment_start = earliest_ts
            while segment_start < latest_ts:
                segment_end = segment_start + SEGMENT_DURATION
                window_df = df[(df['timestamp'] >= segment_start) & (df['timestamp'] < segment_end)]
                if not window_df.empty:
                    _write_segment_files(
                        mrn,
                        segment_id,
                        segment_start,
                        segment_end,
                        window_df,
                        events_sorted,
                    )
                segment_start = segment_end
                segment_id += 1
    else:
        # Multi-threaded processing (original)
        with ThreadPoolExecutor(max_workers=GENERATION_WORKERS) as executor:
            futures = []
            segment_id = 1
            for segment_df in all_segments_sorted:
                df = segment_df.sort_values("timestamp").reset_index(drop=True)
                earliest_ts = df['timestamp'].min()
                latest_ts = df['timestamp'].max()

                segment_start = earliest_ts
                while segment_start < latest_ts:
                    segment_end = segment_start + SEGMENT_DURATION
                    window_df = df[(df['timestamp'] >= segment_start) & (df['timestamp'] < segment_end)]
                    if not window_df.empty:
                        futures.append(
                            executor.submit(
                                _write_segment_files,
                                mrn,
                                segment_id,
                                segment_start,
                                segment_end,
                                window_df,
                                events_sorted,
                            )
                        )
                    segment_start = segment_end
                    segment_id += 1

            for fut in as_completed(futures):
                fut.result()
    
    # Summary for this patient
    if USE_SINGLE_THREAD:
        print(f"[INFO] MRN {mrn}: Completed single-threaded processing")
    else:
        print(f"[INFO] MRN {mrn}: Completed processing with {len(futures)} segments")

def build_prompt(patient_json: Dict[str, Any]) -> str:
    prompt = (
        "You are a clinical assistant. "
        "Convert the following medical data into a clear and structured plain-language clinical report. "
        "Only describe observations as they appear; do not interpret, explain, or recommend treatment. "
        "Group related data (Vitals, Labs, Medications, Past Medical History, Visits) under appropriate headings. "
        "When mentioning measurements or field names, strictly preserve their original capitalization and spacing as in the input data."
    )

    if "pmh" in patient_json and patient_json["pmh"]:
        prompt += "\n\n== Past Medical History =="
        for entry in patient_json["pmh"]:
            desc = entry.get("Desc10") or "Unknown condition"
            codetype = entry.get("CodeType") or "Unknown"
            code = entry.get("Code") or "Unknown"
            ccs = entry.get("DescCCS") or "Unclassified"
            prompt += f"\n- {desc} ({codetype} {code}, Category: {ccs})"

    if "med" in patient_json and patient_json["med"]:
        prompt += "\n\n== Medications =="
        for med in patient_json["med"]:
            name = med.get("Name", "Unknown")
            generic = med.get("Generic_name", "Unknown")
            med_class = med.get("Med_class", "Unknown")
            subclass = med.get("Med_subclass", "Unknown")
            active_status = "Active" if med.get("Active", "N") == "Y" else "Inactive or Unknown"
            prompt += (
                f"\n- {name} ({generic}), Class: {med_class}, Subclass: {subclass}, Status: {active_status}"
            )

    if "visit" in patient_json and patient_json["visit"]:
        prompt += "\n\n== Visits =="
        for visit in patient_json["visit"]:
            prompt += (
                f"\nVisit #{visit.get('Visit_no', 'Unknown')}: "
                f"Age {visit.get('Age', 'Unknown')} years, Gender: {visit.get('Gender', 'Unknown')}, "
                f"Race: {visit.get('Race', 'Unknown')}, Ethnicity: {visit.get('Ethnicity', 'Unknown')}"
                f"\n- Chief Complaint: {visit.get('CC', 'N/A')}"
                f"\n- Triage: Temp {visit.get('Triage_Temp', 'N/A')}°C, "
                f"HR {visit.get('Triage_HR', 'N/A')} bpm, RR {visit.get('Triage_RR', 'N/A')} breaths/min, "
                f"SpO2 {visit.get('Triage_SpO2', 'N/A')}%, BP {visit.get('Triage_SBP', 'N/A')}/"
                f"{visit.get('Triage_DBP', 'N/A')} mmHg"
                f"\n- ED Disposition: {visit.get('ED_dispo', 'Unknown')}, "
                f"Hospital LOS: {visit.get('Hosp_LOS', 'N/A')} days, "
                f"Diagnosis: {visit.get('Dx_name', 'Unknown')} (ICD10: {visit.get('Dx_ICD10', 'Unknown')})"
            )

    if "lab" in patient_json and patient_json["lab"]:
        prompt += "\n\n== Laboratory Results =="
        for lab in patient_json["lab"]:
            abnormal_flag = f", Flag: {lab.get('Abnormal', 'Unknown') }"
            units = lab.get('Component_units') or ""
            normal_range = ""
            if lab.get("Component_nml_low") and lab.get("Component_nml_high"):
                normal_range = f" (Normal Range: {lab['Component_nml_low']} - {lab['Component_nml_high']} {units})"
            prompt += (
                f"\n- {lab.get('Display_name', 'Unknown')}: "
                f"{lab.get('Component_name', 'Unknown')} = {lab.get('Component_value', 'N/A')} {units}"
                f"{abnormal_flag}{normal_range}"
            )

    if "numeric" in patient_json and patient_json["numeric"]:
        prompt += "\n\n== Vital Signs =="
        for num in patient_json["numeric"]:
            prompt += (
                f"\n- {num.get('Source', 'Unknown')} - {num.get('Measure', 'Unknown')}: {num.get('Value', 'N/A')}"
            )

    prompt += "\n\nWrite a concise, plain-language clinical description of the data above."
    return prompt

# ================= vLLM client =================
class MultiGPUVLLMClient:
    def __init__(self, endpoints: List[str], model_name: str, max_concurrent: int = 10):
        self.endpoints = endpoints
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.endpoint_index = 0
        self.http_client = None

    def _get_next_endpoint(self) -> str:
        endpoint = self.endpoints[self.endpoint_index]
        self.endpoint_index = (self.endpoint_index + 1) % len(self.endpoints)
        return endpoint

    async def _ensure_client(self):
        if self.http_client is None:
            # Disable proxy for localhost connections
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=16,
                    max_connections=64,
                    keepalive_expiry=30.0,
                ),
                trust_env=False,  # Don't use environment proxy settings
            )

    async def query_vllm(self, prompt: str) -> str:
        async with self.semaphore:
            await self._ensure_client()
            for attempt in range(len(self.endpoints)):
                try:
                    endpoint = self._get_next_endpoint()
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                    response = await asyncio.wait_for(
                        self.http_client.post(f"{endpoint}/v1/chat/completions", headers=headers, json=payload),
                        timeout=60.0,
                    )
                    if response.status_code != 200:
                        print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                    result = response.json()
                    choices = result.get("choices", [])
                    if not choices:
                        print(f"[ERROR] Empty choices in response: {result}")
                        raise Exception("Empty choices")
                    content = choices[0].get("message", {}).get("content", "")
                    return content
                except asyncio.TimeoutError as e:
                    print(f"[ERROR] Timeout on {endpoint}: {e}")
                    if attempt == len(self.endpoints) - 1:
                        return "Error: Request timed out"
                except Exception as e:
                    print(f"[ERROR] Request failed on {endpoint}: {e}")
                    if attempt == len(self.endpoints) - 1:
                        return f"Error: {str(e)}"
            return "Error: All endpoints failed"

    async def close(self):
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

def _ensure_loop_started():
    global _ASYNC_LOOP, _LOOP_THREAD
    if _ASYNC_LOOP is not None:
        return
    _ASYNC_LOOP = asyncio.new_event_loop()
    def _runner():
        asyncio.set_event_loop(_ASYNC_LOOP)
        _ASYNC_LOOP.run_forever()
    _LOOP_THREAD = threading.Thread(target=_runner, daemon=True)
    _LOOP_THREAD.start()


async def _ensure_client_async():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = MultiGPUVLLMClient(ENDPOINTS, MODEL_NAME, MAX_CONCURRENT)
    return _LLM_CLIENT


async def _process_and_write_report(json_data: Dict[str, Any], output_path: str):
    client = await _ensure_client_async()
    prompt = build_prompt(json_data)
    report = await client.query_vllm(prompt)
    output_data = {"prompt": prompt.strip(), "report": (report or "").strip()}
    with open(output_path, "w") as out_f:
        json.dump(output_data, out_f, indent=4)


def submit_llm_job(json_data: Dict[str, Any], output_json_filename: str):
    try:
        _ensure_loop_started()
        output_path = os.path.join(OUTPUT_LLM_DIR, output_json_filename)
        coro = _process_and_write_report(json_data, output_path)
        future = asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)
        return future
    except Exception:
        return None

def process_llm_report_sync(json_data: Dict[str, Any], output_json_filename: str) -> bool:
    """Synchronous LLM report processing to ensure completion"""
    try:
        output_path = os.path.join(OUTPUT_LLM_DIR, output_json_filename)
        prompt = build_prompt(json_data)
        
        # Use synchronous HTTP client for reliable processing
        import requests
        import time
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                endpoint = ENDPOINTS[0]  # Use first endpoint
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
                
                response = requests.post(
                    f"{endpoint}/v1/chat/completions", 
                    headers=headers, 
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    choices = result.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        output_data = {"prompt": prompt.strip(), "report": (content or "").strip()}
                        
                        with open(output_path, "w") as out_f:
                            json.dump(output_data, out_f, indent=4)
                        
                        print(f"[INFO] Successfully generated LLM report: {output_json_filename}")
                        return True
                    else:
                        print(f"[ERROR] Empty choices in response for {output_json_filename}")
                else:
                    print(f"[ERROR] HTTP {response.status_code} for {output_json_filename}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"[ERROR] Timeout on attempt {attempt + 1} for {output_json_filename}")
            except Exception as e:
                print(f"[ERROR] Request failed on attempt {attempt + 1} for {output_json_filename}: {e}")
            
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # If all attempts failed, create empty report
        output_data = {"prompt": prompt.strip(), "report": "Error: Failed to generate report after multiple attempts"}
        with open(output_path, "w") as out_f:
            json.dump(output_data, out_f, indent=4)
        print(f"[ERROR] Failed to generate LLM report after retries: {output_json_filename}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Critical error in sync LLM processing for {output_json_filename}: {e}")
        return False

async def test_vllm_connection():
    """Test vLLM server connection"""
    print("[INFO] Testing vLLM server connection...")
    try:
        # Simple direct test without the complex client, disable proxy
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            endpoint = ENDPOINTS[0]  # Just test first endpoint
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello, this is a test message."}],
                "temperature": 0.7,
                "top_p": 0.9,
            }
            if DEBUG:
                print(f"[DEBUG] Testing endpoint: {endpoint}")
                print(f"[DEBUG] Model: {MODEL_NAME}")
            
            response = await client.post(f"{endpoint}/v1/chat/completions", headers=headers, json=payload)
            if DEBUG:
                print(f"[DEBUG] Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return False
                
            result = response.json()
            choices = result.get("choices", [])
            if not choices:
                print(f"[ERROR] Empty choices: {result}")
                return False
                
            content = choices[0].get("message", {}).get("content", "")
            print(f"[INFO] vLLM test successful: {content[:100]}...")
            return True
            
    except Exception as e:
        print(f"[ERROR] vLLM test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Test vLLM connection first
    if TEST_VLLM_CONNECTION:
        print("[INFO] Testing vLLM connection...")
        if not asyncio.run(test_vllm_connection()):
            print("[ERROR] vLLM server connection failed. Please check if the server is running.")
            return
    else:
        print("[INFO] Skipping vLLM connection test")
    
    # Show processing mode
    if USE_SINGLE_THREAD:
        print("[INFO] Using SINGLE-THREADED processing mode")
    else:
        print(f"[INFO] Using MULTI-THREADED processing mode ({GENERATION_WORKERS} workers)")
    
    print("[INFO] Reading visits.csv ...")
    visits = pd.read_csv('data/visits.csv')
    print("[INFO] Reading orders.csv ...")
    orders = pd.read_csv('data/orders.csv')
    print("[INFO] Reading meds.csv ...")
    meds = pd.read_csv('data/meds.csv')
    print("[INFO] Reading pmh.csv ...")
    pmh = pd.read_csv('data/pmh.csv')
    print("[INFO] Reading labs.csv ...")
    labs = pd.read_csv('data/labs.csv')
    print("[INFO] Reading rads.csv ...")
    rads = pd.read_csv('data/rads.csv')
    print("[INFO] Reading numerics.csv ...")
    numerics = pd.read_csv('data/numerics.csv')
    print("[INFO] Reading waveform_summary.csv ...")
    waveform_summary = pd.read_csv('data/waveform_summary.csv')

    # Process all MRNs
    all_mrns = visits['MRN'].unique()
    print(f"[INFO] Processing all {len(all_mrns)} MRNs")
    
    with tqdm(total=len(all_mrns), desc="Processing MRNs", unit="MRN") as pbar:
        for mrn in all_mrns:
            print(f"[INFO] Start processing MRN: {mrn}")
            try:
                process_patient(mrn, (visits, orders, meds, pmh, labs, rads, numerics, waveform_summary))
                print(f"[INFO] Finished processing MRN: {mrn}")
            except Exception as e:
                print(f"[ERROR] Failed processing MRN: {mrn}: {e}")
            finally:
                pbar.update(1)

    # Wait for any pending LLM tasks to complete before exit
    if _LLM_FUTURES:
        print(f"[INFO] Waiting for {len(_LLM_FUTURES)} LLM tasks to complete...")
        completed = 0
        failed = 0
        for fut in _LLM_FUTURES:
            try:
                fut.result(timeout=120)
                completed += 1
            except Exception as e:
                failed += 1
                print(f"[ERROR] LLM task failed: {e}")
        print(f"[INFO] LLM tasks completed: {completed}, failed: {failed}")
    else:
        print("[INFO] No LLM tasks to complete")

if __name__ == "__main__":
    main()