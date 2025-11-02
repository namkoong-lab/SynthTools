import json
import os
import argparse
from pathlib import Path

def get_reliable_status_from_folder(tool_folder):
    """
    Check if a tool is reliable based on its evaluation folder.
    
    Args:
        tool_folder (str): Path to the tool's evaluation folder
        
    Returns:
        int or None: 1 if reliable, 0 if not reliable, None if not evaluated
    """
    judge_file_path = os.path.join(tool_folder, "meta_data_tool_call.judge.json")
    
    if not os.path.exists(judge_file_path):
        return None
    
    try:
        with open(judge_file_path, 'r') as f:
            judge_data = json.load(f)
        judgment = judge_data.get("judgment", "")
        if judgment == "correct":
            return 1
        else:
            return 0
    except (json.JSONDecodeError, Exception):
        return None

def update_tool_list_with_reliable(tool_list_path, eval_logs_dir, output_path=None):
    """
    Read tool_list.json, add 'reliable' column based on evaluations from tool_eval_logs folders.
    Tool IDs are extracted from folder names in eval_logs_dir.
    
    Args:
        tool_list_path (str): Path to tool_list.json (JSONL format)
        eval_logs_dir (str): Path to tool_eval_logs directory
        output_path (str, optional): Output path. If None, overwrites input file.
    """
    if not os.path.exists(tool_list_path):
        print(f"Error: Tool list file '{tool_list_path}' does not exist")
        return
    
    if not os.path.exists(eval_logs_dir):
        print(f"Error: Evaluation logs directory '{eval_logs_dir}' does not exist")
        return
    
    # Build a map of tool_id -> reliable status from folder names
    tool_reliable_map = {}
    for item in os.listdir(eval_logs_dir):
        item_path = os.path.join(eval_logs_dir, item)
        if os.path.isdir(item_path):
            # Folder name is the tool ID
            tool_id = item
            reliable_status = get_reliable_status_from_folder(item_path)
            tool_reliable_map[tool_id] = reliable_status
    
    if output_path is None:
        output_path = tool_list_path
    
    updated_tools = []
    
    with open(tool_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tool_data = json.loads(line)
                tool_id = tool_data.get('id')
                if tool_id:
                    # Get reliable status from the map (None if not in map = not evaluated)
                    reliable_status = tool_reliable_map.get(tool_id, None)
                    tool_data['reliable'] = reliable_status
                updated_tools.append(tool_data)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    # Write updated tool list
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for tool_data in updated_tools:
            f.write(json.dumps(tool_data, ensure_ascii=False) + '\n')
    
    print(f"Updated {len(updated_tools)} tools in {output_path}")
    print(f"Found {len(tool_reliable_map)} evaluated tools in {eval_logs_dir}")
    
    # Print summary
    reliable_count = sum(1 for t in updated_tools if t.get('reliable') == 1)
    not_reliable_count = sum(1 for t in updated_tools if t.get('reliable') == 0)
    not_evaluated_count = sum(1 for t in updated_tools if t.get('reliable') is None)
    
    print("\nReliability Summary:")
    print(f"  Reliable (1): {reliable_count}")
    print(f"  Not Reliable (0): {not_reliable_count}")
    print(f"  Not Evaluated (null): {not_evaluated_count}")
    print(f"  Total: {len(updated_tools)}")

def count_folders_and_json_files(directory_path):
    """
    Count the number of folders, non-empty folders, correct judgments, missing judge files, and incorrect judgments.
    
    Args:
        directory_path (str): Path to the directory to analyze
        
    Returns:
        tuple: (folders, non_empty_folders, correct_folders, missing_judge_files, incorrect_judgments)
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist")
        return 0, 0, 0, 0, 0
        
    folders = 0
    non_empty_folders = 0
    correct_folders = 0
    missing_judge_files = 0
    incorrect_judgments = 0
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folders += 1
            
            # Check for both files
            meta_data_file = os.path.join(item_path, "meta_data_tool_call.json")
            judge_file_path = os.path.join(item_path, "meta_data_tool_call.judge.json")
            
            has_meta_data = os.path.exists(meta_data_file)
            has_judge = os.path.exists(judge_file_path)
            
            # Check if folder is non-empty (has at least one file)
            if os.listdir(item_path):
                non_empty_folders += 1
            
            # Case 1: Missing both files -> empty folder (do nothing, already counted)
            if not has_meta_data and not has_judge:
                # Empty folder, skip
                continue
            
            # Case 2: Missing judge file only
            if has_meta_data and not has_judge:
                missing_judge_files += 1
                print(f"Folder: {item}")
                print(f"   Status: Missing judge file (meta_data_tool_call.judge.json not found)")
                print()
            
            # Case 3: Having both files
            elif has_meta_data and has_judge:
                try:
                    with open(judge_file_path, 'r') as f:
                        judge_data = json.load(f)
                        judgment = judge_data.get("judgment", "")
                        if judgment == "correct":
                            correct_folders += 1
                        else:
                            # Both files exist but judgment is not "correct"
                            incorrect_judgments += 1
                            print(f"Folder: {item}")
                            print(f"   Status: Incorrect response")
                            print(f"   Judgment: {judgment}")
                            reasoning = judge_data.get('reasoning', 'No reasoning provided')
                            if reasoning:
                                print(f"   Reasoning: {reasoning}")
                            print()
                except json.JSONDecodeError:
                    print(f"Folder: {item}")
                    print(f"   Status: Error parsing judge file")
                    print(f"   Error: Could not parse JSON in {judge_file_path}")
                    print()
                except Exception as e:
                    print(f"Folder: {item}")
                    print(f"   Status: Error reading judge file")
                    print(f"   Error: {str(e)}")
                    print()
            
    return folders, non_empty_folders, correct_folders, missing_judge_files, incorrect_judgments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count folders and judge results in a directory, optionally update tool_list.json with reliability")
    parser.add_argument("--directory", type=str, required=True, help="Directory path to analyze (e.g., tool_eval_logs)")
    parser.add_argument("--tool_list", type=str, default=None, help="Optional path to tool_list.json to update with 'reliable' column")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for updated tool_list.json (default: overwrites input)")
    args = parser.parse_args()
    
    # If tool_list is provided, update it with reliable status
    if args.tool_list:
        update_tool_list_with_reliable(args.tool_list, args.directory, args.output)
        print()
        print("=" * 60)
        print("DIRECTORY ANALYSIS:")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("INCORRECT OR MISSING JUDGMENTS:")
        print("=" * 60)
        print()
    
    folders, non_empty_folders, correct_folders, missing_judge_files, incorrect_judgments = count_folders_and_json_files(args.directory)
    
    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Number of folders: {folders}")
    print(f"Number of non-empty folders: {non_empty_folders}")
    print(f"Number of folders with correct judgment: {correct_folders}")
    print(f"Number of folders with missing judge file: {missing_judge_files}")
    print(f"Number of folders with incorrect judgment: {incorrect_judgments}")
    print(f"Total issues: {missing_judge_files + incorrect_judgments}")