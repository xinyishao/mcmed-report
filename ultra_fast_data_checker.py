#!/usr/bin/env python3
"""
Ultra Fast Data Checker for Multimodal Clinical Monitoring
Uses file size checking instead of loading files for maximum speed
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Setup minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_fast_data_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory paths
OUTPUT_DIR = "outputs"
OUTPUT_JSON_DIR = "outputs_json"
OUTPUT_LLM_DIR = "outputs_Llama"

class UltraFastDataChecker:
    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers
        self.issues_found = []
        self.stats = {
            'total_npz_files': 0,
            'total_json_files': 0,
            'total_llm_files': 0,
            'empty_npz_files': 0,
            'empty_json_files': 0,
            'missing_json_files': 0,
            'missing_npz_files': 0,
            'missing_llm_files': 0
        }
    
    def get_file_list(self, dir_path: str, extension: str) -> List[str]:
        """Get list of files with specific extension in directory"""
        if not os.path.exists(dir_path):
            logger.error(f"Directory does not exist: {dir_path}")
            return []
        
        try:
            files = [f for f in os.listdir(dir_path) if f.endswith(extension)]
            return sorted(files)
        except Exception as e:
            logger.error(f"Error reading directory {dir_path}: {e}")
            return []
    
    def check_file_empty_by_size(self, file_path: str) -> Tuple[str, bool]:
        """Check if file is empty by file size (much faster than loading content)"""
        try:
            size = os.path.getsize(file_path)
            return os.path.basename(file_path), size == 0
        except Exception as e:
            logger.warning(f"Error checking file size {file_path}: {e}")
            return os.path.basename(file_path), True  # Treat errors as empty
    
    def check_json_empty_quick(self, file_path: str) -> Tuple[str, bool]:
        """Quickly check if JSON file is empty by reading first few characters"""
        try:
            with open(file_path, 'r') as f:
                first_char = f.read(1)
            return os.path.basename(file_path), first_char == '' or first_char == '{' and os.path.getsize(file_path) < 10
        except Exception as e:
            logger.warning(f"Error checking JSON file {file_path}: {e}")
            return os.path.basename(file_path), True
    
    def check_llm_empty_quick(self, file_path: str) -> Tuple[str, bool]:
        """Quickly check if LLM file has empty report"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            # Quick check: if file is very small or doesn't contain "report" field
            if len(content) < 50 or '"report":""' in content or '"report": ""' in content:
                return os.path.basename(file_path), True
            return os.path.basename(file_path), False
        except Exception as e:
            logger.warning(f"Error checking LLM file {file_path}: {e}")
            return os.path.basename(file_path), True
    
    def extract_base_filename(self, filename: str) -> str:
        """Extract base filename without extension"""
        return os.path.splitext(filename)[0]
    
    def check_file_alignment(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """Check alignment between NPZ and JSON files"""
        logger.info("=== Checking File Alignment ===")
        
        # Get file lists
        npz_files = self.get_file_list(OUTPUT_DIR, '.npz')
        json_files = self.get_file_list(OUTPUT_JSON_DIR, '.json')
        llm_files = self.get_file_list(OUTPUT_LLM_DIR, '.json')
        
        self.stats['total_npz_files'] = len(npz_files)
        self.stats['total_json_files'] = len(json_files)
        self.stats['total_llm_files'] = len(llm_files)
        
        logger.info(f"Found {len(npz_files)} NPZ files, {len(json_files)} JSON files, {len(llm_files)} LLM files")
        
        # Create sets of base filenames
        npz_bases = {self.extract_base_filename(f) for f in npz_files}
        json_bases = {self.extract_base_filename(f) for f in json_files}
        llm_bases = {self.extract_base_filename(f) for f in llm_files}
        
        # Find mismatches
        missing_json = npz_bases - json_bases
        missing_npz = json_bases - npz_bases
        missing_llm = json_bases - llm_bases
        
        self.stats['missing_json_files'] = len(missing_json)
        self.stats['missing_npz_files'] = len(missing_npz)
        self.stats['missing_llm_files'] = len(missing_llm)
        
        if missing_json:
            logger.warning(f"NPZ files without corresponding JSON: {len(missing_json)}")
            for base in sorted(list(missing_json)[:10]):  # Show first 10
                self.issues_found.append(f"Missing JSON file for NPZ: {base}")
            if len(missing_json) > 10:
                logger.warning(f"  ... and {len(missing_json) - 10} more")
        
        if missing_npz:
            logger.warning(f"JSON files without corresponding NPZ: {len(missing_npz)}")
            for base in sorted(list(missing_npz)[:10]):  # Show first 10
                self.issues_found.append(f"Missing NPZ file for JSON: {base}")
            if len(missing_npz) > 10:
                logger.warning(f"  ... and {len(missing_npz) - 10} more")
        
        if missing_llm:
            logger.warning(f"JSON files without corresponding LLM report: {len(missing_llm)}")
            for base in sorted(list(missing_llm)[:10]):  # Show first 10
                self.issues_found.append(f"Missing LLM report for JSON: {base}")
            if len(missing_llm) > 10:
                logger.warning(f"  ... and {len(missing_llm) - 10} more")
        
        # Check for exact matches
        matched_files = npz_bases & json_bases
        logger.info(f"Properly matched NPZ-JSON pairs: {len(matched_files)}")
        
        return matched_files, json_bases, llm_bases
    
    def check_empty_files_ultra_fast(self, matched_files: Set[str], json_bases: Set[str], llm_bases: Set[str]) -> None:
        """Check for empty files using ultra-fast methods"""
        logger.info("=== Checking Empty Files (Ultra Fast) ===")
        
        # Prepare file paths for multithreaded processing
        npz_paths = [os.path.join(OUTPUT_DIR, f"{base}.npz") for base in matched_files]
        json_paths = [os.path.join(OUTPUT_JSON_DIR, f"{base}.json") for base in json_bases]
        llm_paths = [os.path.join(OUTPUT_LLM_DIR, f"{base}.json") for base in llm_bases if base in llm_bases]
        
        # Process NPZ files using file size (much faster)
        empty_npz_files = []
        if npz_paths:
            logger.info(f"Checking {len(npz_paths)} NPZ files for empty content (using file size)...")
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {executor.submit(self.check_file_empty_by_size, path): path for path in npz_paths}
                completed = 0
                for future in as_completed(future_to_path):
                    filename, is_empty = future.result()
                    if is_empty:
                        empty_npz_files.append(filename)
                        self.issues_found.append(f"Empty NPZ file: {filename}")
                    completed += 1
                    if completed % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        logger.info(f"Processed {completed}/{len(npz_paths)} NPZ files ({rate:.0f} files/sec)")
            
            elapsed = time.time() - start_time
            logger.info(f"NPZ check completed in {elapsed:.2f}s ({len(npz_paths)/elapsed:.0f} files/sec)")
        
        # Process JSON files
        empty_json_files = []
        if json_paths:
            logger.info(f"Checking {len(json_paths)} JSON files for empty content...")
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {executor.submit(self.check_json_empty_quick, path): path for path in json_paths}
                completed = 0
                for future in as_completed(future_to_path):
                    filename, is_empty = future.result()
                    if is_empty:
                        empty_json_files.append(filename)
                        self.issues_found.append(f"Empty JSON file: {filename}")
                    completed += 1
                    if completed % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        logger.info(f"Processed {completed}/{len(json_paths)} JSON files ({rate:.0f} files/sec)")
            
            elapsed = time.time() - start_time
            logger.info(f"JSON check completed in {elapsed:.2f}s ({len(json_paths)/elapsed:.0f} files/sec)")
        
        # Process LLM files
        empty_llm_files = []
        if llm_paths:
            logger.info(f"Checking {len(llm_paths)} LLM files for empty content...")
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {executor.submit(self.check_llm_empty_quick, path): path for path in llm_paths}
                completed = 0
                for future in as_completed(future_to_path):
                    filename, is_empty = future.result()
                    if is_empty:
                        empty_llm_files.append(filename)
                        self.issues_found.append(f"Empty LLM report: {filename}")
                    completed += 1
                    if completed % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        logger.info(f"Processed {completed}/{len(llm_paths)} LLM files ({rate:.0f} files/sec)")
            
            elapsed = time.time() - start_time
            logger.info(f"LLM check completed in {elapsed:.2f}s ({len(llm_paths)/elapsed:.0f} files/sec)")
        
        # Update stats
        self.stats['empty_npz_files'] = len(empty_npz_files)
        self.stats['empty_json_files'] = len(empty_json_files)
        
        logger.info(f"Empty files found: {len(empty_npz_files)} NPZ, {len(empty_json_files)} JSON, {len(empty_llm_files)} LLM")
    
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
        logger.info(f"Missing LLM files: {self.stats['missing_llm_files']}")
        
        if self.issues_found:
            logger.warning(f"Total issues found: {len(self.issues_found)}")
            logger.warning("Sample issues:")
            for i, issue in enumerate(self.issues_found[:20], 1):  # Show first 20
                logger.warning(f"  {i}. {issue}")
            if len(self.issues_found) > 20:
                logger.warning(f"  ... and {len(self.issues_found) - 20} more issues")
        else:
            logger.info("No issues found! All files are properly aligned and contain data.")
    
    def run_ultra_fast_check(self) -> None:
        """Run ultra fast data validation check"""
        start_time = time.time()
        logger.info("Starting ultra fast data validation check...")
        logger.info(f"Using {self.max_workers} threads for parallel processing")
        logger.info(f"Checking directories: {OUTPUT_DIR}, {OUTPUT_JSON_DIR}, {OUTPUT_LLM_DIR}")
        
        # Check file alignment
        matched_files, json_bases, llm_bases = self.check_file_alignment()
        
        # Check empty files using ultra-fast methods
        self.check_empty_files_ultra_fast(matched_files, json_bases, llm_bases)
        
        # Generate summary
        self.generate_summary_report()
        
        end_time = time.time()
        logger.info(f"Ultra fast data validation check completed in {end_time - start_time:.2f} seconds")

def main():
    """Main function to run the ultra fast data checker"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra Fast Data Checker for Multimodal Clinical Monitoring')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker threads (default: 16)')
    args = parser.parse_args()
    
    checker = UltraFastDataChecker(max_workers=args.workers)
    checker.run_ultra_fast_check()
    
    # Exit with error code if issues found
    if checker.issues_found:
        logger.error(f"Data validation failed with {len(checker.issues_found)} issues")
        exit(1)
    else:
        logger.info("Data validation passed successfully")
        exit(0)

if __name__ == "__main__":
    main()
