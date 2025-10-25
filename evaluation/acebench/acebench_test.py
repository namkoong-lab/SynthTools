import json
import sys
import os
from pathlib import Path
from model_inference.multi_turn.execution_role import EXECUTION

scaling_tools_directory = Path(__file__).parents[2]
sys.path.append(str(scaling_tools_directory))

from simulate_tool import AnthropicToolClient
from pathlib import Path
import json

print(f"Scaling tools directory: {scaling_tools_directory}")

#Load in the subdirectories in data_en that ends with _logs
log_files_dirs = []
for log_file_dir in Path(scaling_tools_directory / "evaluation" / "acebench" / "data_en").glob("**/*_logs"):
    log_files_dirs.append(log_file_dir)

print(f"Log files dirs: {log_files_dirs}")

for log_file_dir in log_files_dirs:
    # extract the tool name from the log file directory
    execution_comparison = {}
    tool_name = log_file_dir.stem
    # directory_name = tool_name.replace(" ", "_")
    # # make a key with the tool name
    # execution_comparison[directory_name] = {}
    print(f"Tool name: {tool_name}")
    for tool_call in log_file_dir.glob("Tool_call*.json"):
        execution_comparison[tool_call.stem] = {}
        # load the tool call
        with open(tool_call, "r") as f:
            tool_call_data = json.load(f)
        # get the tool call parameters
        tool_call_parameters = tool_call_data["tool_parameters"]
        print(f"Tool call parameters: {tool_call_parameters}")
        # get the tool call response
        tool_call_response = tool_call_data["response"]
        print(f"Tool call response: {tool_call_response}")
        # get the tool name from the chat_configs
        tool_name = tool_call_data["chat_configs"]["tool_configs"]["tool_name"]
        print(f"Tool name: {tool_name}")

        # Construct the tool call message
        # For string values, replace spaces with underscores
        tool_call_parts = []
        for k, v in tool_call_parameters.items():
            # Skip empty values
            if v == "" or v is None:
                continue
            # WTF
            if k == "class":
                k = "cabin"
            # WHAT THE ACTUAL FUCK
            elif k == "pass":
                k = "password"
            if isinstance(v, str):
                # Replace spaces with underscores in string values
                # formatted_value = v.replace(" ", " ")
                v = v.replace("\"", "'")
                formatted_value = '"' + v + '"'
                tool_call_parts.append(f"{k} = {formatted_value}")
            elif isinstance(v, list):
                # Handle list values (like in add_food_delivery_order)
                formatted_list = []
                for item in v:
                    if isinstance(item, dict):
                        # Format each dictionary in the list
                        formatted_dict = {key: '"' + val.replace("\"", "'") + '"' if isinstance(val, str) else val 
                                         for key, val in item.items()}
                        formatted_list.append(formatted_dict)
                    else:
                        formatted_list.append(item)
                tool_call_parts.append(f"{k} = {formatted_list}")
            elif isinstance(v, dict):
                formatted_dict = {key: '"' + val.replace("\"", "'") + '"' if isinstance(val, str) else val 
                                 for key, val in v.items()}
                tool_call_parts.append(f"{k} = [{formatted_dict}]")
            else:
                # For non-string values, use as is
                tool_call_parts.append(f"{k} = {v}")
        
        tool_call_message_formatted = f"[{tool_name}({', '.join(tool_call_parts)})]"
        tool_call_data["formatted_tool_call_message"] = tool_call_message_formatted
        print(f"Formatted tool call message: {tool_call_data['formatted_tool_call_message']}")
        print(f"Failure mode: {tool_call_data['failure_mode']}")
        # all classes
        # involved_classes = ["BaseApi", "MessageApi", "ReminderApi", "FoodPlatform", "Travel"]
        initial_config = tool_call_data["chat_configs"]["tool_configs"]["initial_config"]
        # execution = EXECUTION(agent_model_name="gpt-4o", initial_config=tool_call_data["chat_configs"]["tool_configs"]["initial_config"], involved_classes=involved_classes, test_id=tool_call_data["conversation_id"], language="en")
        # initial_config = {
        # "BaseApi": {
        # "wifi": True,
        # "logged_in": True
        #     }
        # }
        # Replace spaces with underscores in the conversation_id to create a valid test_id
        involved_classes = tool_call_data["chat_configs"]["tool_configs"]["involved_classes"]
        test_id = tool_call_data["conversation_id"].replace(" ", "_")
        execution = EXECUTION(agent_model_name="gpt-4o", initial_config=initial_config, involved_classes=involved_classes, test_id=test_id, language="en")


        current_message, result_instance = execution.respond([{"message": tool_call_data["formatted_tool_call_message"]}])


        execution_comparison[tool_call.stem]["original_tool_call_message"] = tool_call_data["message"]
        execution_comparison[tool_call.stem]["formated_tool_call_messages"] = tool_call_data["formatted_tool_call_message"]
        execution_comparison[tool_call.stem]["tool_call_response"] = tool_call_data["response"]
        execution_comparison[tool_call.stem]["execution_messages"] = current_message["message"]
        execution_comparison[tool_call.stem]["initial_config"] = initial_config
        execution_comparison[tool_call.stem]["failure_mode"] = tool_call_data["failure_mode"]
        execution_comparison[tool_call.stem]["tool_call_parameters"] = tool_call_data["tool_parameters"]
        execution_comparison[tool_call.stem]["tool_parameter_scheme"] = tool_call_data["chat_configs"]["tool_configs"]["parameters"]



        print(f"Current message: {current_message}\n")
        print(f"Result instance: {result_instance}\n")
    
    # Check if file exists and overwrite it
    execution_comparison_file = log_file_dir / "execution_comparison_updated.json"
    with open(execution_comparison_file, "w") as f:
        json.dump(execution_comparison, f, indent=4)




# tool_call_example = [{"message": "[add_food_delivery_order(username = Jack, merchant_name = 'Burger King', items = [{'product': 'Whopper', 'quantity': 1}, {'product': 'Fries', 'quantity': 1}, {'product': 'Coke', 'quantity': 1}])]"}]
# test_cases_path = scaling_tools_directory / "evaluation" / "acebench" / "data_en" / "collection" / "data_agent_multi_turn.json"
# # Load JSONL file (one JSON object per line)
# test_cases = []
# with open(test_cases_path, "r") as f:
#     for line in f:
#         if line.strip():  # Skip empty lines
#             test_cases.append(json.loads(line))

# test_case = test_cases[-1]
# print("initial config: ", test_case["initial_config"])
# print("involved classes: ", test_case["involved_classes"])
# print("id: ", test_case["id"])

# tool_client = AnthropicToolClient(tool_config_file=scaling_tools_directory / "tool_configs" / "example_wifi_tool.json", mega_data_tool_file=scaling_tools_directory / "mega_data_tool_configs" / "example_wifi_tool__output.json", using_states=False)

# initial_config = {
#     "BaseApi": {
#         "wifi": True,
#         "logged_in": True
#     }
# }
# involved_classes = ["BaseApi", "MessageApi", "ReminderApi", "FoodPlatform", "Travel"]
# execution = EXECUTION(agent_model_name="gpt-4o", initial_config=initial_config, involved_classes=involved_classes, test_id="hi", language="en")


# current_message, result_instance = execution.respond(tool_call_example)
# print(f"Current message: {current_message}\n")
# print(f"Result instance: {result_instance}\n")
