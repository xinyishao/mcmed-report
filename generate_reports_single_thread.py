#!/usr/bin/env python3
"""
Single-threaded report generator to fix missing LLM reports
Handles race conditions by processing files sequentially
"""

import os
import json
import numpy as np
import requests
import time
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# Disable proxy for localhost connections
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

# Configuration
OUTPUT_DIR = "outputs"
OUTPUT_JSON_DIR = "outputs_json"
OUTPUT_LLM_DIR = "outputs_Llama"
ENDPOINTS = ["http://localhost:8000"]
MODEL_NAME = "/gemini/platform/public/aigc/Lirui/chengding/Meta-Llama-3.1-8B-Instruct"
SKIP_VLLM_TEST = False  # Set to True to skip vLLM connection test

# Statistics tracking
stats = {
    "total_npz_files": 0,
    "missing_json": 0,
    "empty_json": 0,
    "missing_reports": 0,
    "empty_reports": 0,
    "generated_reports": 0,
    "failed_reports": 0,
    "skipped_reports": 0
}

def build_prompt(patient_json: Dict[str, Any]) -> str:
    """Build prompt for LLM report generation"""
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

def check_json_file(json_path: str) -> Tuple[bool, bool, Dict]:
    """
    Check JSON file status
    Returns: (exists, is_empty, data)
    """
    if not os.path.exists(json_path):
        return False, True, {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        is_empty = len(data) == 0
        return True, is_empty, data
    except Exception as e:
        print(f"[ERROR] Failed to read JSON file {json_path}: {e}")
        return True, True, {}

def check_report_file(report_path: str) -> Tuple[bool, bool, str]:
    """
    Check LLM report file status
    Returns: (exists, is_valid, content)
    """
    if not os.path.exists(report_path):
        return False, False, ""
    
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        report_content = data.get('report', '').strip()
        is_valid = report_content != '' and not report_content.startswith('Error:')
        return True, is_valid, report_content
    except Exception as e:
        print(f"[ERROR] Failed to read report file {report_path}: {e}")
        return True, False, ""

def generate_report_sync(json_data: Dict[str, Any], report_path: str, filename: str) -> bool:
    """Generate LLM report synchronously with retry logic"""
    try:
        prompt = build_prompt(json_data)
        
        for attempt in range(3):  # Retry up to 3 times
            try:
                endpoint = ENDPOINTS[0]
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
                
                print(f"[INFO] Generating report for {filename} (attempt {attempt + 1}/3)")
                response = requests.post(
                    f"{endpoint}/v1/chat/completions", 
                    headers=headers, 
                    json=payload,
                    timeout=60,
                    proxies={'http': None, 'https': None}  # Disable proxy
                )
                
                if response.status_code == 200:
                    result = response.json()
                    choices = result.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        output_data = {"prompt": prompt.strip(), "report": (content or "").strip()}
                        
                        # Write report file
                        with open(report_path, "w") as out_f:
                            json.dump(output_data, out_f, indent=4)
                        
                        # Verify the report was written correctly
                        if verify_report(report_path):
                            print(f"[SUCCESS] Generated report for {filename}")
                            return True
                        else:
                            print(f"[ERROR] Report verification failed for {filename}")
                    else:
                        print(f"[ERROR] Empty choices in response for {filename}")
                else:
                    print(f"[ERROR] HTTP {response.status_code} for {filename}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"[ERROR] Timeout on attempt {attempt + 1} for {filename}")
            except Exception as e:
                print(f"[ERROR] Request failed on attempt {attempt + 1} for {filename}: {e}")
            
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # If all attempts failed, create error report
        output_data = {"prompt": prompt.strip(), "report": "Error: Failed to generate report after multiple attempts"}
        with open(report_path, "w") as out_f:
            json.dump(output_data, out_f, indent=4)
        print(f"[ERROR] Failed to generate report after retries: {filename}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Critical error in report generation for {filename}: {e}")
        return False

def verify_report(report_path: str) -> bool:
    """Double-check if report was generated correctly"""
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        report_content = data.get('report', '').strip()
        return report_content != '' and not report_content.startswith('Error:')
    except Exception:
        return False

def process_npz_file(npz_filename: str) -> None:
    """Process a single NPZ file and ensure it has a proper report"""
    base_filename = os.path.splitext(npz_filename)[0]
    json_filename = f"{base_filename}.json"
    report_filename = f"{base_filename}.json"  # Same name in LLM directory
    
    npz_path = os.path.join(OUTPUT_DIR, npz_filename)
    json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
    report_path = os.path.join(OUTPUT_LLM_DIR, report_filename)
    
    # Check NPZ file exists
    if not os.path.exists(npz_path):
        print(f"[ERROR] NPZ file not found: {npz_path}")
        return
    
    # Check JSON file
    json_exists, json_empty, json_data = check_json_file(json_path)
    
    if not json_exists:
        print(f"[WARNING] Missing JSON file: {json_filename}")
        stats["missing_json"] += 1
        return
    
    if json_empty:
        print(f"[WARNING] Empty JSON file: {json_filename}")
        stats["empty_json"] += 1
        # Still try to generate a report for empty JSON
    
    # Check report file
    report_exists, report_valid, report_content = check_report_file(report_path)
    
    if not report_exists:
        print(f"[INFO] Missing report: {report_filename}")
        stats["missing_reports"] += 1
    elif not report_valid:
        print(f"[INFO] Invalid/empty report: {report_filename}")
        stats["empty_reports"] += 1
    else:
        print(f"[SKIP] Valid report exists: {report_filename}")
        stats["skipped_reports"] += 1
        return
    
    # Generate report
    print(f"[PROCESSING] Generating report for {report_filename}")
    success = generate_report_sync(json_data, report_path, report_filename)
    
    if success:
        stats["generated_reports"] += 1
    else:
        stats["failed_reports"] += 1

def test_vllm_connection() -> bool:
    """Test vLLM server connection"""
    print("[INFO] Testing vLLM server connection...")
    try:
        endpoint = ENDPOINTS[0]
        
        # First check if the server is running on the port
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 8000))
            sock.close()
            if result != 0:
                print(f"[ERROR] Port 8000 is not open on localhost")
                return False
            print("[DEBUG] Port 8000 is open on localhost")
        except Exception as e:
            print(f"[ERROR] Cannot check port 8000: {e}")
            return False
        
        # First try a simple health check (bypass proxy)
        try:
            health_response = requests.get(
                f"{endpoint}/health", 
                timeout=5,
                proxies={'http': None, 'https': None}
            )
            print(f"[DEBUG] Health check response: {health_response.status_code}")
        except:
            print("[DEBUG] Health check endpoint not available, trying chat completions directly")
        
        # Try chat completions with different headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Python-requests/2.28.0"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        print(f"[DEBUG] Testing endpoint: {endpoint}")
        print(f"[DEBUG] Model: {MODEL_NAME}")
        
        response = requests.post(
            f"{endpoint}/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=15,
            proxies={'http': None, 'https': None}  # Disable proxy
        )
        
        print(f"[DEBUG] Response status: {response.status_code}")
        print(f"[DEBUG] Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[SUCCESS] vLLM connection successful! Response: {result}")
            return True
        elif response.status_code == 403:
            print(f"[ERROR] vLLM server returned 403 Forbidden. This might be due to:")
            print(f"[ERROR] - Authentication required")
            print(f"[ERROR] - CORS policy blocking requests")
            print(f"[ERROR] - Server configuration issues")
            print(f"[ERROR] Response text: {response.text}")
            return False
        else:
            print(f"[ERROR] vLLM connection failed: HTTP {response.status_code}")
            print(f"[ERROR] Response text: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Cannot connect to vLLM server: {e}")
        print(f"[ERROR] Make sure the vLLM server is running on {ENDPOINTS[0]}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] vLLM server timeout: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] vLLM connection failed: {e}")
        return False

def main():
    print("=== Single-Threaded Report Generator ===")
    print("This script will process each NPZ file sequentially to avoid race conditions")
    print()
    
    # Test vLLM connection (unless skipped)
    if not SKIP_VLLM_TEST:
        if not test_vllm_connection():
            print("[ERROR] Cannot proceed without vLLM connection")
            print("[INFO] You can set SKIP_VLLM_TEST = True to bypass this check")
            return
    else:
        print("[INFO] Skipping vLLM connection test")
    
    # Get all NPZ files
    if not os.path.exists(OUTPUT_DIR):
        print(f"[ERROR] Output directory not found: {OUTPUT_DIR}")
        return
    
    npz_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')]
    npz_files.sort()  # Process in consistent order
    
    stats["total_npz_files"] = len(npz_files)
    print(f"[INFO] Found {len(npz_files)} NPZ files to process")
    
    if len(npz_files) == 0:
        print("[WARNING] No NPZ files found!")
        return
    
    # Process each NPZ file sequentially
    print("\n[INFO] Starting sequential processing...")
    for i, npz_filename in enumerate(npz_files, 1):
        print(f"\n--- Processing {i}/{len(npz_files)}: {npz_filename} ---")
        process_npz_file(npz_filename)
    
    # Print final statistics
    print("\n=== FINAL STATISTICS ===")
    print(f"Total NPZ files processed: {stats['total_npz_files']}")
    print(f"Missing JSON files: {stats['missing_json']}")
    print(f"Empty JSON files: {stats['empty_json']}")
    print(f"Missing reports: {stats['missing_reports']}")
    print(f"Empty reports: {stats['empty_reports']}")
    print(f"Reports generated: {stats['generated_reports']}")
    print(f"Reports failed: {stats['failed_reports']}")
    print(f"Reports skipped (already valid): {stats['skipped_reports']}")
    
    # Print statistics (no file generation)
    print(f"\n=== Report Generation Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if stats['generated_reports'] > 0:
        print(f"\n[SUCCESS] Generated {stats['generated_reports']} reports successfully!")
    if stats['failed_reports'] > 0:
        print(f"[WARNING] {stats['failed_reports']} reports failed to generate")

if __name__ == "__main__":
    main()
