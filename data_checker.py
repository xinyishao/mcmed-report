#!/usr/bin/env python3
"""
Data Checker for Multimodal Clinical Monitoring
Validates NPZ and JSON file alignment and content integrity
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
OUTPUT_DIR = "outputs"
OUTPUT_JSON_DIR = "outputs_json"
OUTPUT_LLM_DIR = "outputs_Llama"

class DataChecker:
    def __init__(self):
        self.issues_found = []
        self.stats = {
            'total_npz_files': 0,
            'total_json_files': 0,
            'total_llm_files': 0,
            'empty_npz_files': 0,
            'empty_json_files': 0,
            'missing_json_files': 0,
            'missing_npz_files': 0,
            'mismatched_files': 0
        }
    
    def check_directory_exists(self, dir_path: str) -> bool:
        """Check if directory exists and is accessible"""
        if not os.path.exists(dir_path):
            logger.error(f"Directory does not exist: {dir_path}")
            self.issues_found.append(f"Missing directory: {dir_path}")
            return False
        
        if not os.path.isdir(dir_path):
            logger.error(f"Path is not a directory: {dir_path}")
            self.issues_found.append(f"Path is not a directory: {dir_path}")
            return False
        
        return True
    
    def get_file_list(self, dir_path: str, extension: str) -> List[str]:
        """Get list of files with specific extension in directory"""
        if not self.check_directory_exists(dir_path):
            return []
        
        try:
            files = [f for f in os.listdir(dir_path) if f.endswith(extension)]
            logger.info(f"Found {len(files)} {extension} files in {dir_path}")
            return sorted(files)
        except Exception as e:
            logger.error(f"Error reading directory {dir_path}: {e}")
            self.issues_found.append(f"Error reading directory {dir_path}: {e}")
            return []
    
    def check_npz_file_content(self, file_path: str) -> Tuple[bool, Dict]:
        """Check if NPZ file is valid and not empty"""
        try:
            data = np.load(file_path, allow_pickle=True)
            file_info = {
                'keys': list(data.keys()),
                'shapes': {},
                'total_elements': 0,
                'data_types': {}
            }
            
            # Safely check each array
            for key in data.keys():
                try:
                    arr = data[key]
                    file_info['shapes'][key] = arr.shape
                    file_info['data_types'][key] = str(arr.dtype)
                    file_info['total_elements'] += arr.size
                    
                    # Special handling for timestamp arrays
                    if key == 'timestamp' and arr.size > 0:
                        try:
                            # Try to get a sample timestamp to check if it's valid
                            sample_ts = arr[0] if arr.size > 0 else None
                            logger.debug(f"Sample timestamp in {file_path}: {sample_ts} (type: {type(sample_ts)})")
                        except Exception as ts_e:
                            logger.warning(f"Could not read timestamp sample in {file_path}: {ts_e}")
                            
                except Exception as arr_e:
                    logger.warning(f"Could not read array '{key}' in {file_path}: {arr_e}")
                    file_info['shapes'][key] = "ERROR"
                    file_info['data_types'][key] = "ERROR"
            
            # Check if file is empty
            is_empty = file_info['total_elements'] == 0
            
            if is_empty:
                logger.warning(f"Empty NPZ file: {file_path}")
                self.issues_found.append(f"Empty NPZ file: {file_path}")
                self.stats['empty_npz_files'] += 1
            
            # Check for required keys (timestamp and ppg_value based on the script)
            required_keys = ['timestamp', 'ppg_value']
            missing_keys = [key for key in required_keys if key not in data.keys()]
            if missing_keys:
                logger.warning(f"NPZ file missing required keys {missing_keys}: {file_path}")
                self.issues_found.append(f"NPZ file missing required keys {missing_keys}: {file_path}")
            
            # Check if timestamp array has valid data
            if 'timestamp' in data.keys():
                try:
                    ts_arr = data['timestamp']
                    if ts_arr.size > 0:
                        logger.debug(f"Timestamp array in {file_path}: shape={ts_arr.shape}, dtype={ts_arr.dtype}, sample={ts_arr[0]}")
                    else:
                        logger.warning(f"Empty timestamp array in {file_path}")
                        self.issues_found.append(f"Empty timestamp array in NPZ: {file_path}")
                except Exception as ts_e:
                    logger.warning(f"Could not read timestamp array in {file_path}: {ts_e}")
                    self.issues_found.append(f"Timestamp array error in NPZ: {file_path}")
            
            data.close()
            return not is_empty, file_info
            
        except Exception as e:
            logger.error(f"Error reading NPZ file {file_path}: {e}")
            self.issues_found.append(f"Error reading NPZ file {file_path}: {e}")
            return False, {}
    
    def check_json_file_content(self, file_path: str) -> Tuple[bool, Dict]:
        """Check if JSON file is valid and not empty"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if file is empty
            is_empty = len(data) == 0
            
            # Detailed analysis of empty JSON files
            if is_empty:
                logger.warning(f"Empty JSON file: {file_path}")
                self.issues_found.append(f"Empty JSON file: {file_path}")
                self.stats['empty_json_files'] += 1
                
                # Try to extract MRN and segment info from filename for debugging
                filename = os.path.basename(file_path)
                logger.warning(f"  Empty JSON details - Filename: {filename}")
                
                # Check if corresponding NPZ exists and has data
                npz_path = file_path.replace('outputs_json', 'outputs').replace('.json', '.npz')
                if os.path.exists(npz_path):
                    logger.warning(f"  Corresponding NPZ exists: {npz_path}")
                    try:
                        npz_data = np.load(npz_path, allow_pickle=True)
                        if 'timestamp' in npz_data.keys():
                            ts_arr = npz_data['timestamp']
                            logger.warning(f"  NPZ timestamp range: {ts_arr[0]} to {ts_arr[-1]} ({len(ts_arr)} points)")
                        npz_data.close()
                    except Exception as npz_e:
                        logger.warning(f"  Could not read corresponding NPZ: {npz_e}")
                else:
                    logger.warning(f"  No corresponding NPZ file found: {npz_path}")
            
            # Check for expected event types
            expected_types = ['pmh', 'med', 'visit', 'lab', 'numeric']
            found_types = list(data.keys())
            unexpected_types = [t for t in found_types if t not in expected_types]
            if unexpected_types:
                logger.warning(f"JSON file has unexpected event types {unexpected_types}: {file_path}")
                self.issues_found.append(f"JSON file has unexpected event types {unexpected_types}: {file_path}")
            
            # Count events by type
            event_counts = {}
            for event_type in found_types:
                event_counts[event_type] = len(data.get(event_type, []))
            
            return not is_empty, {
                'event_types': found_types,
                'event_counts': event_counts,
                'total_events': sum(event_counts.values())
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            self.issues_found.append(f"Invalid JSON in file {file_path}: {e}")
            return False, {}
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            self.issues_found.append(f"Error reading JSON file {file_path}: {e}")
            return False, {}
    
    def extract_base_filename(self, filename: str) -> str:
        """Extract base filename without extension"""
        return os.path.splitext(filename)[0]
    
    def check_file_alignment(self) -> None:
        """Check alignment between NPZ and JSON files"""
        logger.info("=== Checking File Alignment ===")
        
        # Get file lists
        npz_files = self.get_file_list(OUTPUT_DIR, '.npz')
        json_files = self.get_file_list(OUTPUT_JSON_DIR, '.json')
        llm_files = self.get_file_list(OUTPUT_LLM_DIR, '.json')
        
        self.stats['total_npz_files'] = len(npz_files)
        self.stats['total_json_files'] = len(json_files)
        self.stats['total_llm_files'] = len(llm_files)
        
        # Create sets of base filenames
        npz_bases = {self.extract_base_filename(f) for f in npz_files}
        json_bases = {self.extract_base_filename(f) for f in json_files}
        
        # Find mismatches
        missing_json = npz_bases - json_bases
        missing_npz = json_bases - npz_bases
        
        self.stats['missing_json_files'] = len(missing_json)
        self.stats['missing_npz_files'] = len(missing_npz)
        
        if missing_json:
            logger.warning(f"NPZ files without corresponding JSON: {len(missing_json)}")
            for base in sorted(missing_json):
                logger.warning(f"  Missing JSON for: {base}")
                self.issues_found.append(f"Missing JSON file for NPZ: {base}")
        
        if missing_npz:
            logger.warning(f"JSON files without corresponding NPZ: {len(missing_npz)}")
            for base in sorted(missing_npz):
                logger.warning(f"  Missing NPZ for: {base}")
                self.issues_found.append(f"Missing NPZ file for JSON: {base}")
        
        # Check for exact matches
        matched_files = npz_bases & json_bases
        logger.info(f"Properly matched NPZ-JSON pairs: {len(matched_files)}")
        
        return matched_files
    
    def check_file_contents(self, matched_files: Set[str]) -> None:
        """Check contents of matched files"""
        logger.info("=== Checking File Contents ===")
        
        for base_filename in sorted(matched_files):
            npz_path = os.path.join(OUTPUT_DIR, f"{base_filename}.npz")
            json_path = os.path.join(OUTPUT_JSON_DIR, f"{base_filename}.json")
            
            # Check NPZ content
            npz_valid, npz_info = self.check_npz_file_content(npz_path)
            
            # Check JSON content
            json_valid, json_info = self.check_json_file_content(json_path)
            
            # Log detailed info for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"File pair: {base_filename}")
                logger.debug(f"  NPZ info: {npz_info}")
                logger.debug(f"  JSON info: {json_info}")
    
    def check_llm_outputs(self) -> None:
        """Check LLM output files"""
        logger.info("=== Checking LLM Outputs ===")
        
        llm_files = self.get_file_list(OUTPUT_LLM_DIR, '.json')
        json_files = self.get_file_list(OUTPUT_JSON_DIR, '.json')
        
        # Create sets for comparison
        llm_bases = {self.extract_base_filename(f) for f in llm_files}
        json_bases = {self.extract_base_filename(f) for f in json_files}
        
        # Find missing LLM reports
        missing_llm = json_bases - llm_bases
        if missing_llm:
            logger.warning(f"Missing LLM reports for {len(missing_llm)} JSON files:")
            for base in sorted(missing_llm):
                logger.warning(f"  Missing LLM report for: {base}")
                self.issues_found.append(f"Missing LLM report for JSON: {base}")
        
        # Check existing LLM files
        empty_reports = 0
        for llm_file in llm_files:
            llm_path = os.path.join(OUTPUT_LLM_DIR, llm_file)
            try:
                with open(llm_path, 'r') as f:
                    data = json.load(f)
                
                # Check for required fields
                required_fields = ['prompt', 'report']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    logger.warning(f"LLM file missing fields {missing_fields}: {llm_file}")
                    self.issues_found.append(f"LLM file missing fields {missing_fields}: {llm_file}")
                
                # Check if report is empty
                report_content = data.get('report', '').strip()
                if report_content == '':
                    logger.warning(f"Empty LLM report: {llm_file}")
                    self.issues_found.append(f"Empty LLM report: {llm_file}")
                    empty_reports += 1
                    
                    # Check if corresponding JSON has data
                    json_path = llm_path.replace('outputs_Llama', 'outputs_json')
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                json_data = json.load(f)
                            if len(json_data) > 0:
                                logger.warning(f"  Corresponding JSON has data but LLM report is empty: {llm_file}")
                                logger.warning(f"  JSON event types: {list(json_data.keys())}")
                        except Exception as json_e:
                            logger.warning(f"  Could not read corresponding JSON: {json_e}")
                
            except Exception as e:
                logger.error(f"Error reading LLM file {llm_file}: {e}")
                self.issues_found.append(f"Error reading LLM file {llm_file}: {e}")
        
        logger.info(f"LLM output summary: {len(llm_files)} files, {empty_reports} empty reports, {len(missing_llm)} missing reports")
    
    def analyze_root_causes(self) -> None:
        """Analyze potential root causes of data issues"""
        logger.info("=== Root Cause Analysis ===")
        
        # Check if data directories exist
        data_dirs = ['data/visits.csv', 'data/orders.csv', 'data/meds.csv', 'data/pmh.csv', 
                    'data/labs.csv', 'data/rads.csv', 'data/numerics.csv', 'data/waveform_summary.csv']
        
        missing_data_files = []
        for data_file in data_dirs:
            if not os.path.exists(data_file):
                missing_data_files.append(data_file)
        
        if missing_data_files:
            logger.warning(f"Missing data files: {missing_data_files}")
            self.issues_found.append(f"Missing data files: {missing_data_files}")
        
        # Check waveform data directory
        waveform_dir = "data/waveforms"
        if not os.path.exists(waveform_dir):
            logger.warning(f"Missing waveform directory: {waveform_dir}")
            self.issues_found.append(f"Missing waveform directory: {waveform_dir}")
        else:
            # Count waveform subdirectories
            try:
                subdirs = [d for d in os.listdir(waveform_dir) if os.path.isdir(os.path.join(waveform_dir, d))]
                logger.info(f"Found {len(subdirs)} waveform subdirectories")
            except Exception as e:
                logger.warning(f"Could not read waveform directory: {e}")
        
        # Analyze empty JSON patterns
        empty_json_files = []
        json_files = self.get_file_list(OUTPUT_JSON_DIR, '.json')
        for json_file in json_files:
            json_path = os.path.join(OUTPUT_JSON_DIR, json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if len(data) == 0:
                    empty_json_files.append(json_file)
            except:
                pass
        
        if empty_json_files:
            logger.warning(f"Found {len(empty_json_files)} empty JSON files")
            logger.warning("This suggests issues with:")
            logger.warning("  1. Event timestamp parsing (safe_parse_timestamp function)")
            logger.warning("  2. Event filtering logic in _write_segment_files")
            logger.warning("  3. Missing or invalid event data in source CSV files")
            logger.warning("  4. Time window alignment between waveform segments and events")
            
            # Sample a few empty files for detailed analysis
            sample_empty = empty_json_files[:3]
            for sample_file in sample_empty:
                logger.warning(f"  Sample empty JSON: {sample_file}")
                # Extract MRN from filename for debugging
                if 'p' in sample_file and '-' in sample_file:
                    mrn_part = sample_file.split('-')[0]
                    logger.warning(f"    MRN: {mrn_part}")
    
    def generate_summary_report(self) -> None:
        """Generate summary report of all checks"""
        logger.info("=== SUMMARY REPORT ===")
        
        logger.info(f"Total NPZ files: {self.stats['total_npz_files']}")
        logger.info(f"Total JSON files: {self.stats['total_json_files']}")
        logger.info(f"Total LLM files: {self.stats['total_llm_files']}")
        logger.info(f"Empty NPZ files: {self.stats['empty_npz_files']}")
        logger.info(f"Empty JSON files: {self.stats['empty_json_files']}")
        logger.info(f"Missing JSON files: {self.stats['missing_json_files']}")
        logger.info(f"Missing NPZ files: {self.stats['missing_npz_files']}")
        
        if self.issues_found:
            logger.warning(f"Total issues found: {len(self.issues_found)}")
            logger.warning("Issues summary:")
            for i, issue in enumerate(self.issues_found, 1):
                logger.warning(f"  {i}. {issue}")
        else:
            logger.info("No issues found! All files are properly aligned and contain data.")
    
    def run_full_check(self) -> None:
        """Run complete data validation check"""
        logger.info("Starting data validation check...")
        logger.info(f"Checking directories: {OUTPUT_DIR}, {OUTPUT_JSON_DIR}, {OUTPUT_LLM_DIR}")
        
        # Check file alignment
        matched_files = self.check_file_alignment()
        
        # Check file contents
        if matched_files:
            self.check_file_contents(matched_files)
        
        # Check LLM outputs
        self.check_llm_outputs()
        
        # Analyze root causes
        self.analyze_root_causes()
        
        # Generate summary
        self.generate_summary_report()
        
        logger.info("Data validation check completed.")

def main():
    """Main function to run the data checker"""
    checker = DataChecker()
    checker.run_full_check()
    
    # Exit with error code if issues found
    if checker.issues_found:
        logger.error(f"Data validation failed with {len(checker.issues_found)} issues")
        exit(1)
    else:
        logger.info("Data validation passed successfully")
        exit(0)

if __name__ == "__main__":
    main()
