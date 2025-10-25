#!/usr/bin/env python3
"""
Main script to orchestrate the tool deduplication pipeline.

This script parses the configuration file and runs the two main steps:
1. compute_embs.py - Compute embeddings for tool JSON files
2. run_deduplication.py - Run deduplication on tool embeddings

Usage:
  python main_dedup.py [--config CONFIG_FILE] [--step STEP_NAME]

Examples:
  python main_dedup.py
  python main_dedup.py --config configs/deduplicate_tools_config.yml
  python main_dedup.py --step compute_embs
  python main_dedup.py --step run_deduplication
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def build_command_args(config_section: Dict[str, Any], script_name: str) -> List[str]:
    args = []
    
    if script_name == "compute_embs":
        if 'inputs' in config_section:
            inputs = config_section['inputs']
            if isinstance(inputs, list):
                args.extend(inputs)
            else:
                args.append(inputs)
        
        for key, value in config_section.items():
            if key == 'inputs':
                continue
            elif isinstance(value, bool) and not value:
                continue
            else:
                if key == "output_dir":
                    arg_key = "-o"
                elif key == "batch_size":
                    arg_key = "--batch-size"
                else:
                    arg_key = f"--{key}"
                args.extend([arg_key, str(value)])
    
    elif script_name == "run_deduplication":
        for key, value in config_section.items():
            if isinstance(value, bool) and not value:
                continue
            elif value is None:
                continue
            else:
                if key == "tool_json_dir":
                    arg_key = "--tool-json-dir"
                elif key == "emb_dir":
                    arg_key = "--emb-dir"
                elif key == "yaml_dir":
                    arg_key = "--yaml-dir"
                elif key == "out_path":
                    arg_key = "--out"
                elif key == "out_dir":
                    arg_key = "--out-dir"
                elif key == "random_seed":
                    arg_key = "--random-seed"
                elif key == "log_file":
                    arg_key = "--log-file"
                elif key == "log_level":
                    arg_key = "--log-level"
                elif key == "exact_ded":
                    arg_key = "--exact-ded"
                elif key == "taus":
                    arg_key = "--taus"
                    if isinstance(value, list):
                        args.extend([arg_key] + [str(v) for v in value])
                    else:
                        args.extend([arg_key, str(value)])
                    continue
                else:
                    arg_key = f"--{key.replace('_', '-')}"
                
                args.extend([arg_key, str(value)])
    
    return args

def run_script(script_path: str, args: List[str], config: Dict[str, Any], script_name: str) -> bool:
    cmd = [sys.executable, script_path] + args
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=0)
        
        print("Starting script execution...")
        
        import select
        import sys as sys_module
        
        while True:
            if process.poll() is not None:
                break
                
            if sys_module.platform != 'win32':
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        print(line.rstrip(), flush=True)
                    else:
                        break
                else:
                    print(".", end="", flush=True)
            else:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip(), flush=True)
                else:
                    break
        
        remaining_output = process.stdout.read()
        if remaining_output:
            print(remaining_output.rstrip(), flush=True)
        
        process.wait()
        
        if process.returncode == 0:
            print("\n" + "-" * 80)
            print(f"Script completed successfully: {script_path}")
            return True
        else:
            print("\n" + "-" * 80)
            print(f"Script failed with exit code {process.returncode}: {script_path}")
            return False
            
    except Exception as e:
        print("\n" + "-" * 80)
        print(f"Error running script {script_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Orchestrate the tool deduplication pipeline")
    parser.add_argument(
        "--config",
        default="configs/deduplicate_tools_config.yml",
        help="Path to configuration file (default: configs/deduplicate_tools_config.yml)"
    )
    parser.add_argument(
        "--step",
        choices=["compute_embs", "run_deduplication"],
        help="Run only a specific step instead of the full pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing them"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    scripts_dir = Path(__file__).parent / "scripts"
    script_paths = {
        "compute_embs": scripts_dir / "compute_embs.py",
        "run_deduplication": scripts_dir / "run_deduplication.py"
    }
    
    for name, path in script_paths.items():
        if not path.exists():
            print(f"Script not found: {path}")
            sys.exit(1)
    
    pipeline_steps = ["compute_embs", "run_deduplication"]
    
    if args.step:
        steps_to_run = [args.step]
    else:
        steps_to_run = pipeline_steps
    
    print(f"Pipeline steps to run: {', '.join(steps_to_run)}")
    print("=" * 80)
    
    success_count = 0
    total_steps = len(steps_to_run)
    
    for step in steps_to_run:
        print(f"\nRunning step: {step}")
        
        if step not in config:
            print(f"Configuration section '{step}' not found in config file")
            continue
        
        script_path = script_paths[step]
        script_args = build_command_args(config[step], step)
        
        if args.dry_run:
            cmd = [sys.executable, str(script_path)] + script_args
            print(f"Would run: {' '.join(cmd)}")
            success_count += 1
        else:
            if run_script(str(script_path), script_args, config, step):
                success_count += 1
            else:
                print(f"Pipeline stopped due to failure in step: {step}")
                break
    
    print("\n" + "=" * 80)
    if args.dry_run:
        print(f"Dry run completed. Would run {success_count}/{total_steps} steps.")
    else:
        if success_count == total_steps:
            print(f"Pipeline completed successfully! ({success_count}/{total_steps} steps)")
        else:
            print(f"Pipeline completed with failures. ({success_count}/{total_steps} steps successful)")
            sys.exit(1)

if __name__ == "__main__":
    main()
