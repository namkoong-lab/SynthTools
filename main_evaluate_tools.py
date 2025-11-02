#!/usr/bin/env python3
"""
Main script to orchestrate the tool evaluation pipeline.

This script parses the configuration file and runs the three main steps:
1. simulate_tool.py - Simulate tool calls with metadata (mode 3)
2. judge_simulator.py - Judge simulator response quality (mode 3)
3. count.py - Count and update tool_list.json with reliability

Usage:
  python main_evaluate_tools.py [--config CONFIG_FILE] [--step STEP_NAME]

Examples:
  python main_evaluate_tools.py
  python main_evaluate_tools.py --config configs/evaluate_tools_config.yml
  python main_evaluate_tools.py --step simulate_tool
  python main_evaluate_tools.py --step judge_simulator
  python main_evaluate_tools.py --step count
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

def resolve_path(path_str: str, base_dir: Path) -> str:
    """Resolve relative paths relative to project root."""
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())

def build_command_args(config_section: Dict[str, Any], script_name: str, proj_root: Path) -> List[str]:
    args = []
    
    if script_name == "simulate_tool":
        args.append("--mode")
        args.append("3")
        
        for key, value in config_section.items():
            if value is None:
                continue
            else:
                if key == "tool_json_dir":
                    arg_key = "--tool_json_dir"
                elif key == "tool_meta_dir":
                    arg_key = "--tool_meta_dir"
                elif key == "output_dir":
                    arg_key = "--output_dir"
                else:
                    arg_key = f"--{key}"
                resolved_path = resolve_path(str(value), proj_root)
                args.extend([arg_key, resolved_path])
    
    elif script_name == "judge_simulator":
        args.append("--mode")
        args.append("3")
        
        for key, value in config_section.items():
            if isinstance(value, bool) and not value:
                continue
            elif value is None:
                continue
            else:
                if key == "log_dir":
                    arg_key = "--log_dir"
                elif key == "model":
                    arg_key = "--model"
                elif key == "verbose":
                    if value:
                        args.append("--verbose")
                    continue
                else:
                    arg_key = f"--{key}"
                
                if key == "log_dir":
                    resolved_path = resolve_path(str(value), proj_root)
                    args.extend([arg_key, resolved_path])
                else:
                    args.extend([arg_key, str(value)])
    
    elif script_name == "count":
        for key, value in config_section.items():
            if value is None:
                continue
            else:
                if key == "directory":
                    arg_key = "--directory"
                elif key == "tool_list":
                    arg_key = "--tool_list"
                elif key == "output":
                    if value is not None and str(value).lower() != "null":
                        arg_key = "--output"
                        resolved_path = resolve_path(str(value), proj_root)
                        args.extend([arg_key, resolved_path])
                    continue
                else:
                    arg_key = f"--{key}"
                
                if key in ["directory", "tool_list"]:
                    resolved_path = resolve_path(str(value), proj_root)
                    args.extend([arg_key, resolved_path])
                else:
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
    parser = argparse.ArgumentParser(description="Orchestrate the tool evaluation pipeline")
    parser.add_argument(
        "--config",
        default="configs/evaluate_tools_config.yml",
        help="Path to configuration file (default: configs/evaluate_tools_config.yml)"
    )
    parser.add_argument(
        "--step",
        choices=["simulate_tool", "judge_simulator", "count"],
        help="Run only a specific step instead of the full pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without executing them"
    )
    
    args = parser.parse_args()
    
    proj_root = Path(__file__).parent.resolve()
    config_path = resolve_path(args.config, proj_root) if not Path(args.config).is_absolute() else args.config
    
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    
    simulator_dir = proj_root / "simulator"
    scripts_dir = proj_root / "scripts"
    
    script_paths = {
        "simulate_tool": simulator_dir / "simulate_tool.py",
        "judge_simulator": simulator_dir / "judge_simulator.py",
        "count": scripts_dir / "count.py"
    }
    
    for name, path in script_paths.items():
        if not path.exists():
            print(f"Script not found: {path}")
            sys.exit(1)
    
    pipeline_steps = ["simulate_tool", "judge_simulator", "count"]
    
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
        script_args = build_command_args(config[step], step, proj_root)
        
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

