import json
import sys
import os
from pathlib import Path
from model_inference.multi_turn.execution_role import EXECUTION
import anthropic
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
# import utils.misc_utils as mscu


# load the meta data generating prompt from meta_data_generating.yml
with open(Path(__file__).resolve().parent / "meta_data_generating.yml", "r") as f:
    mega_data_prompt = f.read()

def get_api_keys(key: str):
    """Get API keys from the config file."""
    # Make folder path based on absolute file location
    script_dir = Path(__file__).resolve().parent.parent.parent
    config_path = script_dir / "configs" / "api_keys.json"
    with open(config_path) as f:
        api_keys = json.load(f)

    try:
        return api_keys.get(key)
    except KeyError:
        print(
            f"API key '{key}' not found."
            f" Please add it to the config file at {config_path}"
        )
        return None

def extract_text(response) -> str:
    """Extract text from the Anthropic API response."""
    parts = []
    for block in response.content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        else:
            t = getattr(block, "type", None)
            if t == "text":
                parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def generate_meta_data(class_file_path, mega_data_prompt, tool_config):
    # get the api key from configs/api_keys.json 
    print(f"Generating meta data for {class_file_path}")
    client = anthropic.Anthropic(api_key=get_api_keys("anthropic"))
    print(f"Using API key: {get_api_keys('anthropic')}")
    # turn a python file into a string
    with open(class_file_path, "r") as f:
        python_file = f.read()
    print(f"Using python file: {python_file}")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": mega_data_prompt + "\n\n" + python_file + "\n\n" + tool_config}],
        max_tokens=10000,
    )
    print(f"Response: {response}")
    return extract_text(response)

scaling_tools_directory = Path(__file__).parents[2]
sys.path.append(str(scaling_tools_directory))

multi_turn_test_cases_path = scaling_tools_directory / "evaluation" / "acebench" / "data_en" / "collection" / "data_agent_multi_turn.json"
multi_step_test_cases_path = scaling_tools_directory / "evaluation" / "acebench" / "data_en" / "collection" / "data_agent_multi_step.json"

# Load JSONL file (one JSON object per line)
multi_turn_test_cases = []
with open(multi_turn_test_cases_path, "r") as f:
    for line in f:
        if line.strip():  # Skip empty lines
            multi_turn_test_cases.append(json.loads(line))

multi_step_test_cases = []
with open(multi_step_test_cases_path, "r") as f:
    for line in f:
        if line.strip():  # Skip empty lines
            multi_step_test_cases.append(json.loads(line))

test_cases = multi_turn_test_cases + multi_step_test_cases
for test_case in test_cases:
    for function in test_case["function"]:
        tool_config_path = scaling_tools_directory / "evaluation" / "acebench" / "data_en" / f"{function['name']}_tool_config.json"
        meta_data_tool_config_path = scaling_tools_directory / "evaluation" / "acebench" / "data_en" / f"{function['name']}_mega_data_tool_config.json"
        
        # Check if the tool already exists, add in the involved_classes in the tool_config, and continue
        if os.path.exists(tool_config_path):
            with open(tool_config_path, "r") as f:
                tool_config = json.load(f)
            tool_config["involved_classes"] = test_case["involved_classes"]
            with open(tool_config_path, "w") as f:
                json.dump(tool_config, f, indent=4, sort_keys=False)
            continue
        # Check if the tool already exists, add in the initial_config in the tool_config, and continue
        # if os.path.exists(tool_config_path):
        #     # with open(tool_config_path, "r") as f:
        #     #     tool_config = json.load(f)
        #     # tool_config["initial_config"] = test_case["initial_config"]
        #     # with open(tool_config_path, "w") as f:
        #     #     json.dump(tool_config, f, indent=4, sort_keys=False)
        #     # wrap meta data in mscu.LiteralString
        #     with open(tool_config_path, "r") as f:
        #         tool_config = json.load(f)
        #     tool_config["meta_data"] = mscu.LiteralString(tool_config["meta_data"])
        #     print(f"Literal string: {tool_config['meta_data']}")
        #     with open(tool_config_path, "w") as f:
        #         json.dump(tool_config, f, indent=4, sort_keys=False)
        #     continue
            
        parameters = function["parameters"]
        required_params = parameters["required"]
        parameter_properties = parameters["properties"]
        # In parameter_properties, for each parameter dictionary, add a required key
        for param in parameter_properties:
            parameter_properties[param]["required"] = param in required_params
            
        tool_config = {
            "tool_name": function["name"],
            "tool_description": function["description"],
            "parameters": parameter_properties,
            "error_messages": [],
            "usage": function["description"],
            "involved_classes": test_case["involved_classes"]
        }
        print(f"Tool config: {tool_config}")
        print("*"*100)

        # for class_name in test_case["involved_classes"]:
        #     print(f"Class name: {class_name}")
        #     if class_name == "BaseApi":
        #         class_file_path = scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "base_api.py"
        #         meta_data = generate_meta_data(class_file_path, mega_data_prompt, str(tool_config))
        #         if "meta_data" in tool_config:
        #             tool_config["meta_data"] += "\n" + meta_data
        #         else:
        #             tool_config["meta_data"] = meta_data
        #     elif class_name == "MessageApi":
        #         class_file_path = scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "message.py"
        #         meta_data = generate_meta_data(class_file_path, mega_data_prompt, str(tool_config))
        #         if "meta_data" in tool_config:
        #             tool_config["meta_data"] += "\n" + meta_data
        #         else:
        #             tool_config["meta_data"] = meta_data
        #     elif class_name == "ReminderApi":
        #         class_file_path = scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "reminder.py"
        #         meta_data = generate_meta_data(class_file_path, mega_data_prompt, str(tool_config))
        #         if "meta_data" in tool_config:
        #             tool_config["meta_data"] += "\n" + meta_data
        #         else:
        #             tool_config["meta_data"] = meta_data
        #     elif class_name == "FoodPlatform":
        #         class_file_path = scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "food_services.py"
        #         meta_data = generate_meta_data(class_file_path, mega_data_prompt, str(tool_config))
        #         if "meta_data" in tool_config:
        #             tool_config["meta_data"] += "\n" + meta_data
        #         else:
        #             tool_config["meta_data"] = meta_data
        #     elif class_name == "Travel":
        #         class_file_path = scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "travel.py"
        #         meta_data = generate_meta_data(class_file_path, mega_data_prompt, str(tool_config))
        #         if "meta_data" in tool_config:
        #             tool_config["meta_data"] += "\n" + meta_data
        #         else:
        #             tool_config["meta_data"] = meta_data
        #     else:
        #         raise ValueError(f"Class name {class_name} not found")
        #     print(f"Tool config: {tool_config}")
        #     print(f"Meta data: {tool_config['meta_data']}")
        # print(f"Tool config: {tool_config}")
        # print("*"*100)
        
        # Write to the file
        # with open(tool_config_path, "w") as f:
        #     json.dump(tool_config, f, indent=4, sort_keys=False)
        # print(f"Created tool config for {function['name']}: {json.dumps(tool_config, indent=2)}")


