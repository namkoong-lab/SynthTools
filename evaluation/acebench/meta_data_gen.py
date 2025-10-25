import json
import sys
import os
from pathlib import Path
from model_inference.multi_turn.execution_role import EXECUTION
import anthropic


scaling_tools_directory = Path(__file__).parents[2]
sys.path.append(str(scaling_tools_directory))
import utils.misc_utils as mscu
from simulate_tool import AnthropicToolClient
from pathlib import Path
import json

# load the meta data generating prompt from meta_data_generating.yml
with open(Path(__file__).resolve().parent / "meta_data_generating.yml", "r") as f:
    meta_data_prompt = f.read()

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


def generate_meta_data(class_file_path, mega_data_prompt):
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
        messages=[{"role": "user", "content": meta_data_prompt + "\n\n" + python_file}],
        max_tokens=10000,
    )
    print(f"Response: {response}")
    return extract_text(response)

print(f"Scaling tools directory: {scaling_tools_directory}")

# Load in the tool_config.json files in data_en
tool_config_files = list(Path(scaling_tools_directory / "evaluation" / "acebench" / "data_en").glob("*.json"))

# base_api_file = Path(scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "base_api.py")
# base_api_meta_data = generate_meta_data(base_api_file, meta_data_prompt)

# message_api_file = Path(scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "message.py")
# message_api_meta_data = generate_meta_data(message_api_file, meta_data_prompt)

# reminder_api_file = Path(scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "reminder.py")
# reminder_api_meta_data = generate_meta_data(reminder_api_file, meta_data_prompt)

# food_platform_file = Path(scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "phone_platform" / "food_services.py")
# food_platform_meta_data = generate_meta_data(food_platform_file, meta_data_prompt)

# travel_file = Path(scaling_tools_directory / "evaluation" / "acebench" / "model_inference" / "multi_turn" / "scenariosen" / "travel.py")
# travel_meta_data = generate_meta_data(travel_file, meta_data_prompt)

for tool_config_file in tool_config_files:
    with open(tool_config_file, "r") as f:
        tool_config = json.load(f)
    tool_config["meta_data"] = mscu.LiteralString(tool_config["meta_data"])
    
    # print(f"Tool config: {tool_config}")

    # for involved_class in tool_config["involved_classes"]:
    #     if involved_class == "BaseApi":
    #         tool_config["meta_data"] = base_api_meta_data + "\n" + tool_config["meta_data"]
    #     elif involved_class == "MessageApi":
    #         tool_config["meta_data"] = message_api_meta_data + "\n" + tool_config["meta_data"]
    #     elif involved_class == "ReminderApi":
    #         tool_config["meta_data"] = reminder_api_meta_data + "\n" + tool_config["meta_data"]
    #     elif involved_class == "FoodPlatform":
    #         tool_config["meta_data"] = food_platform_meta_data + "\n" + tool_config["meta_data"]
    #     elif involved_class == "Travel":
    #         tool_config["meta_data"] = travel_meta_data + "\n" + tool_config["meta_data"]

    with open(tool_config_file, "w") as f:
        json.dump(tool_config, f, indent=4, sort_keys=False)

    print(f"Tool config: {tool_config}")
    print("*"*100)