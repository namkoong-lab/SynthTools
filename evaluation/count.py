import json
import os
def count_folders_and_json_files(directory_path):
    """
    Count the number of folders and .json files in the specified directory.
    Also counts folders with 'correct' judgment in meta_data_tool_call_fixed_judge.json.
    
    Args:
        directory_path (str): Path to the directory to analyze
        
    Returns:
        tuple: (number of folders, number of .json files, number of folders with correct judgment)
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist")
        return 0, 0, 0
        
    folders = 0
    json_files = 0
    correct_folders = 0
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            folders += 1
            # Check for meta_data_tool_call_fixed_judge.json in each folder
            judge_file_path = os.path.join(item_path, "meta_data_tool_call_fixed.judge.json")
            if os.path.exists(judge_file_path):
                try:
                    with open(judge_file_path, 'r') as f:
                        judge_data = json.load(f)
                        if judge_data.get("judgment") == "correct":
                            correct_folders += 1
                        else:
                            # print out the name of the folder and the judgement as well as reasoning
                            print(f"Folder: {item}, Judgment: {judge_data.get('judgment')}, Reasoning: {judge_data.get('reasoning')}")
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON in {judge_file_path}")
                except Exception as e:
                    print(f"Error reading {judge_file_path}: {str(e)}")
        elif item.endswith('.json'):
            json_files += 1
            
    return folders, json_files, correct_folders

if __name__ == "__main__":
    directory = "/Users/ny2336/Desktop/adaptive-tool-use/scaling_tools/tool_content/tool_final/tool_json"
    folders, json_files, correct_folders = count_folders_and_json_files(directory)
    print(f"Number of folders: {folders}")
    print(f"Number of .json files: {json_files}")
    print(f"Number of folders with correct judgment: {correct_folders}")