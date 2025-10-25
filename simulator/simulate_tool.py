import json
import os
import time
import uuid
from pathlib import Path
import utils.misc_utils as mscu

import yaml
from anthropic import Anthropic
from openai import OpenAI

DEFAULT_CONFIG = {
    "max_tokens": 10000,
    "model": "claude-sonnet-4-20250514",
    "model_provider": "anthropic",
    "tool_collection_version": "4",
    "tool_final": True,
    "temperature": 0.0,
    "base_delay": 2.0,
    "backoff_factor": 2.0,
    "max_retries": 3
}


class ChatLogger:
    def __init__(self, log_config=None, log_config_file=None, **kwargs):
        self.log_config = log_config
        self.log_config_file = (
            log_config_file
            if log_config_file
            else Path(__file__).resolve().parent / "configs" / "log_configs.json"
        )
        self.project_name = kwargs.get("project_name", "ToolSimulator")
        self.chat_configs = kwargs.get("chat_configs", {})

        if self.log_config_file and log_config is None:
            with open(self.log_config_file) as f:
                self.log_config = json.load(f)

        self._update_relative_paths()
        self._validate_log_config()

        self.index_format = "{:08d}_{}.json"

        print(f"Index logging folders: {list(self.index_logging_folders)}")
        print(f"ID logging folders: {list(self.id_logging_folders)}")

    def update_chat_configs(self, new_chat_configs):
        self.chat_configs.update(new_chat_configs)

    def log(self, message, response):
        id = self._make_id()
        log_json = self.compile_logs(message, response)
        log_json["conversation_id"] = id

        for id_folder in self.id_logging_folders:
            self._log_to_id(id_folder, log_json, id)
        for index_folder in self.index_logging_folders:
            self._log_to_index(index_folder, log_json, id)

    def compile_logs(self, message, response):
        if len(self.chat_configs) == 0:
            print(
                f"WARNING: No chat setups provided to ChatLogger. Is there no chat configuration you want to log?"
            )
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")

        log_json = {
            "message": message,
            "response": response,
            "timestamp": timestamp,
            "chat_configs": self.chat_configs,
        }
        return log_json

    def _make_id(self):
        "Make id from uuid and YMD-HMS"
        return str(uuid.uuid4()) + "_" + time.strftime("%Y%m%d-%H%M%S")

    def _log_to_id(self, id_folder, log_json, id):
        log_filename = f"{id}.json"
        log_filepath = id_folder / log_filename

        with open(log_filepath, "w") as f:
            json.dump(log_json, f, indent=2)

    def _log_to_index(self, index_folder, log_json, id):
        latest_index_file = index_folder / "_latest_index.txt"
        if os.path.exists(latest_index_file):
            with open(latest_index_file) as f:
                latest_index = int(f.read().strip())
        else:
            latest_index = -1

        new_index = latest_index + 1
        log_filename = self.index_format.format(new_index, id)
        log_filepath = index_folder / log_filename

        with open(log_filepath, "w") as f:
            json.dump(log_json, f, indent=2)

        with open(index_folder / "_latest_index.txt", "w") as f:
            f.write(str(new_index))

    def _update_relative_paths(self):
        "Change ./ prefix to start from script root"
        script_root = Path(__file__).resolve().parent

        for folder_set in ["index_logging_folders", "id_logging_folders"]:
            new_folder_sets = {}
            for folder, config in self.log_config.get(folder_set, {}).items():
                if folder.startswith("./"):
                    new_folder = script_root / folder[2:]
                    new_folder_sets[new_folder] = config
                else:
                    new_folder_sets[Path(folder)] = config
                self.log_config[folder_set] = new_folder_sets

    def _validate_log_config(self):
        required_fields = ["index_logging_folders", "id_logging_folders"]
        for field in required_fields:
            if field not in self.log_config:
                raise ValueError(f"Missing required log config field: {field}")

        for folder_set in ["index_logging_folders", "id_logging_folders"]:
            idc = []
            for folder, config in self.log_config.get(folder_set, {}).items():
                if config.get("create", True):
                    os.makedirs(folder, exist_ok=True)
                else:
                    if not Path(folder).is_dir():
                        print(
                            f"WARNING. Specified {folder_set} folder does not exist: {folder}"
                        )
                        idc.append(folder)
            self.log_config[folder_set] = {
                folder: config
                for folder, config in self.log_config.get(folder_set, {}).items()
                if folder not in idc
            }

        self.index_logging_folders = self.log_config.get("index_logging_folders", {})
        self.id_logging_folders = self.log_config.get("id_logging_folders", {})


class ModelClient:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", DEFAULT_CONFIG["max_tokens"])
        self.temperature = kwargs.get("temperature", DEFAULT_CONFIG["temperature"])

        self.max_retries = kwargs.get("max_retries", DEFAULT_CONFIG["max_retries"])
        self.base_delay = kwargs.get("base_delay", DEFAULT_CONFIG["base_delay"])
        self.backoff_factor = kwargs.get("backoff_factor", DEFAULT_CONFIG["backoff_factor"])

        self.chat_logger = ChatLogger(
            chat_configs={
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )

    def _message(self, content):
        raise NotImplementedError("Subclasses must implement the message method.")

    def message(self, content, log = True, **kwargs):
        for i in range(self.max_retries):
            try:
                response = self._message(content, **kwargs)
                if log:
                    self.chat_logger.log(message=content, response=response)
                return response
            except Exception as e:
                print(f"Attempt {i + 1} failed: {e}")
                time.sleep(self.base_delay * (self.backoff_factor**i))
        raise RuntimeError("All attempts failed.")


class ToolClient(ModelClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        script_dir = Path(__file__).resolve().parent
        self.tool_config_file = kwargs.get(
            "tool_config_file", f"{script_dir}/tool_configs/example_wifi_tool.json"
        )
        self.metadata_tool_file = kwargs.get("metadata_tool_file", None)
        self.using_states = kwargs.get("using_states", False)
        self.load_tool_config()
        if self.metadata_tool_file:
            self.load_metadata_tool_config()

        if self.using_states:
            if "states" in kwargs:
                print(
                    f"Replacing tool states from {self.tool_config['states']} to {kwargs['states']}"
                )
                self.tool_config["states"] = kwargs["states"]
            else:
                print(f"Using default tool states: {self.tool_config['states']}")

        self.combined_prompt_template_file = kwargs.get(
            "combined_prompt_template_file",
            f"{script_dir}/prompt_templates/tool_simulator_template.yml",
        )
        self.parameter_check_prompt_template_file = kwargs.get(
            "parameter_check_prompt_template_file",
            f"{script_dir}/prompt_templates/parameter_check.yml",
        )
        self.return_message_gen_prompt_template_file = kwargs.get(
            "return_message_gen_prompt_template_file",
            f"{script_dir}/prompt_templates/return_message_gen.yml",
        )
        self.load_combined_prompt_template()
        self.load_parameter_check_prompt_template()
        self.load_return_message_gen_prompt_template()
        self._validate_tool_config()

        prompt_data = self.tool_config.copy()
        prompt_data["tool_details"]= str(self.tool_config)
        if hasattr(self, 'metadata_tool_config'):
            if 'Tool Call' in self.metadata_tool_config:
                prompt_data['tool_call'] = self.metadata_tool_config['Tool Call']
            if 'Return Data' in self.metadata_tool_config:
                prompt_data['return_data'] = self.metadata_tool_config['Return Data']
            
            excluded_keys = {'Tool Call', 'Return Data'}
            context_metadata = {k: v for k, v in self.metadata_tool_config.items() 
                              if k not in excluded_keys}
            prompt_data['metadata'] = context_metadata
            prompt_data['meta_data'] = context_metadata
            
            for key, value in self.metadata_tool_config.items():
                if key not in prompt_data and key not in excluded_keys:
                    prompt_data[key] = value
        
        default_values = {
            'return_data': 'No example provided',
            'tool_call': 'No example provided',
            'metadata': 'No context metadata provided',
            'meta_data': 'No context metadata provided',
            'initial_config': 'No initial config provided'
        }
        for key, default_value in default_values.items():
            if key not in prompt_data:
                prompt_data[key] = mscu.LiteralString(default_value)
        
        self.combined_prompt = self.combined_prompt_template.format(**prompt_data)
        self.parameter_check_prompt = self.parameter_check_prompt_template.format(**prompt_data)
        self.return_message_gen_prompt = self.return_message_gen_prompt_template.format(**prompt_data)

        self.chat_logger.update_chat_configs(
            {
                "tool_configs": self.tool_config,
                "combined_prompt_template": self.combined_prompt_template,
                "prompt": self.combined_prompt,
                "parameter_check_prompt_template": self.parameter_check_prompt_template,
                "parameter_check_prompt": self.parameter_check_prompt,
                "return_message_gen_prompt_template": self.return_message_gen_prompt_template,
                "return_message_gen_prompt": self.return_message_gen_prompt,
            }
        )

    def load_tool_config(self):
        with open(self.tool_config_file) as f:
            self.tool_config = json.load(f)

    def load_metadata_tool_config(self):
        with open(self.metadata_tool_file) as f:
            self.metadata_tool_config = json.load(f)

    def load_combined_prompt_template(self):
        with open(self.combined_prompt_template_file) as f:
            self.combined_prompt_yaml = yaml.safe_load(f)

        self.combined_prompt_schema = self.combined_prompt_yaml.get("schema", None)
        if not self.combined_prompt_schema:
            raise ValueError(f"Missing prompt schema in {self.combined_prompt_template_file}")

        self.combined_prompt_template = self.combined_prompt_yaml.get("template", None)
        if not self.combined_prompt_template:
            raise ValueError(f"Missing prompt template in {self.combined_prompt_template_file}")
    
    def load_parameter_check_prompt_template(self):
        with open(self.parameter_check_prompt_template_file) as f:
            self.parameter_check_prompt_yaml = yaml.safe_load(f)

        self.parameter_check_prompt_schema = self.parameter_check_prompt_yaml.get("schema", None)
        if not self.parameter_check_prompt_schema:
            raise ValueError(f"Missing prompt schema in {self.parameter_check_prompt_template_file}")

        self.parameter_check_prompt_template = self.parameter_check_prompt_yaml.get("template", None)
        if not self.parameter_check_prompt_template:
            raise ValueError(f"Missing prompt template in {self.parameter_check_prompt_template_file}")
    
    def load_return_message_gen_prompt_template(self):

        with open(self.return_message_gen_prompt_template_file) as f:
            self.return_message_gen_prompt_yaml = yaml.safe_load(f)

        self.return_message_gen_prompt_schema = self.return_message_gen_prompt_yaml.get("schema", None)
        if not self.return_message_gen_prompt_schema:
            raise ValueError(f"Missing prompt schema in {self.return_message_gen_prompt_template_file}")

        self.return_message_gen_prompt_template = self.return_message_gen_prompt_yaml.get("template", None)
        if not self.return_message_gen_prompt_template:
            raise ValueError(f"Missing prompt template in {self.return_message_gen_prompt_template_file}")
    
    def _validate_tool_config(self):
        required_fields = self.combined_prompt_schema.get("required", [])

        for field in required_fields:
            if field not in self.tool_config and not (hasattr(self, 'metadata_tool_config') and field in self.metadata_tool_config):
                raise ValueError(f"Missing required field: {field}")


class AnthropicToolClient(ToolClient):
    ANTHROPIC_MODELS = [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
    ]

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)

        self.client = self.get_anthropic_client()
        if model:
            assert model in self.ANTHROPIC_MODELS
            self.model = model
        else:
            self.model = "claude-sonnet-4-20250514"
        self.model = DEFAULT_CONFIG["model"]

        self.chat_logger.update_chat_configs({"model": self.model})

    def get_api_keys(self, key: str):
        """Get API keys from the config file."""
        script_dir = Path(__file__).resolve().parent
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

    def get_anthropic_client(self) -> Anthropic:
        """Get an instance of the Anthropic client."""
        api_key = self.get_api_keys("anthropic")
        return Anthropic(api_key=api_key)

    def extract_text(self, response) -> str:
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

    def _message(self, content, text_only=True, prompt_type="combined"):
        if prompt_type == "combined":
            msg_content = self.combined_prompt + "\n" + content
        elif prompt_type == "parameter_check":
            msg_content = self.parameter_check_prompt + "\n" + content
            print(f"Parameter check prompt:\n{msg_content}")
        elif prompt_type == "return_message_gen":
            msg_content = self.return_message_gen_prompt + "\n" + content
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        if DEFAULT_CONFIG["model_provider"] == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": msg_content}],
            )
            if text_only:
                response = self.extract_text(response)
        elif DEFAULT_CONFIG["model_provider"] == "openai":
            client = OpenAI(api_key=self.get_api_keys("openai"))
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": msg_content}],
                max_completion_tokens=self.max_tokens,
            )
            if text_only:
                response = response.choices[0].message.content

        return response

    def generate_test_tool_calls(self, tool_file, log_folder, model=None, max_tokens=None, model_provider=None, is_simulated=True):
        if model is None:
            model = DEFAULT_CONFIG["model"]
        if max_tokens is None:
            max_tokens = DEFAULT_CONFIG["max_tokens"]
        if model_provider is None:
            model_provider = DEFAULT_CONFIG["model_provider"]
        if is_simulated:
            template_file = Path(__file__).resolve().parent / "prompt_templates" / "generate_testing_functions_tool_simulator" / "generate_testing_tool_calls.yml"
        else:
            template_file = Path(__file__).resolve().parent / "prompt_templates" / "generate_acebench_tool_calls" / "generate_successful_tool_calls.yml"
        with open(template_file) as f:
            template = yaml.safe_load(f)
        template = template["template"]

        with open(tool_file) as f:
            tool_config = json.load(f)

        tool_config_new = {}
        tool_config_new["tool_details"] = str(tool_config)

        message = template.format(**tool_config_new)

        print(f"Message to the llm:\n{message}")

        if model_provider == "openai":
            openai_client = OpenAI(api_key=self.get_api_keys("openai"))
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_completion_tokens=max_tokens,
            )
            tool_calls_str = response.choices[0].message.content
        elif model_provider == "anthropic":
            anthropic_client = Anthropic(api_key=self.get_api_keys("anthropic"))
            response = anthropic_client.messages.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens,
            )
            tool_calls_str = self.extract_text(response)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        print(f"Tool calls str:\n{tool_calls_str}")

        try:
            tool_calls = json.loads(tool_calls_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Attempting to extract JSON from response...")
            import re
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', tool_calls_str, re.DOTALL)
            if json_match:
                try:
                    tool_calls = json.loads(json_match.group(1))
                    print("Successfully extracted JSON from markdown")
                except json.JSONDecodeError:
                    print("Failed to parse extracted JSON, raising original error")
                    raise e
            else:
                print("No JSON found in response, raising original error")
                raise e

        print(f"Tool calls:\n{tool_calls}")
        tool_calls_dir = log_folder / "tool_calls.json"
        print(f"Tool calls dir:\n{tool_calls_dir}")
        with open(tool_calls_dir, "w") as f:
            json.dump(tool_calls, f, indent=4)
        print(f"Tool calls dumped to:\n{tool_calls_dir}")
        return tool_calls_dir

def run_simulated_tools(tool_final = None,tool_collection_version=None):
    if tool_final is None:
        tool_final = DEFAULT_CONFIG["tool_final"]
    if tool_final:
        base_path = Path(__file__).resolve().parent / "tool_content" / "tool_final"
        tool_call_files = base_path / "tool_json"
        metadata_tool_files = base_path / "tool_meta"
    else:
        if tool_collection_version is None:
            tool_collection_version = DEFAULT_CONFIG["tool_collection_version"]
        base_path = Path(__file__).resolve().parent / "tool_content" / "full_tool_specs"
        tool_call_files = base_path / f"tool_collection_json_{tool_collection_version}"
        metadata_tool_files = base_path / f"tool_collection_meta_data_{tool_collection_version}"
    print("Running Simulated Tools mode...")

    if not tool_call_files.exists():
        print(f"Tool collection directory not found: {tool_call_files}")
        return
    if not metadata_tool_files.exists():
        print(f"Meta data directory not found: {metadata_tool_files}")
        return

    process_tool_files(tool_call_files, metadata_tool_files, is_simulated=True)

def run_acebench_tools():
    print("Running Acebench Tools mode...")
    
    tool_call_files = Path(__file__).resolve().parent / "evaluation" / "acebench" / "data_en" 
    metadata_tool_files = Path(__file__).resolve().parent / "evaluation" / "acebench" / "data_en"
    
    process_tool_files(tool_call_files, metadata_tool_files, is_simulated=False)

def process_tool_files(tool_call_files, metadata_tool_files, is_simulated=True):
    separator = "*" * 42
    number_problems = 0
    
    for tool_file in tool_call_files.glob("*.json"):
        try:
            tool_file_name = tool_file.stem

            print(f"\n{separator * 2}\n")
            print(f"\nTool file name: {tool_file_name}")
            
            if is_simulated:
                metadata_tool_file = metadata_tool_files / f"{tool_file_name}__output.json"
            else:
                metadata_tool_file = tool_file

            log_folder = tool_file.parent / f"{tool_file.stem}_logs"
            log_folder.mkdir(parents=True, exist_ok=True)
            print(f"Made log folder: {log_folder}")

            existing_tool_calls = list(log_folder.glob("Tool_call_*.json"))
            if len(existing_tool_calls) > 5:
                print(f"Tool calls already generated for {tool_file_name}")
                continue
                
            tool_client = AnthropicToolClient(tool_config_file=tool_file, metadata_tool_file=metadata_tool_file, using_states=False)

            print(f"Initialized tool client")
            
            tool_calls_dir = tool_client.generate_test_tool_calls(tool_file, log_folder, model_provider=DEFAULT_CONFIG["model_provider"], is_simulated=is_simulated)
            tool_calls_dir = log_folder / "tool_calls.json"
            with open(tool_calls_dir, "r") as f:
                tool_calls = json.load(f)

            for tool_call in tool_calls:
                print(f"Tool call: {tool_calls[tool_call]}")
                response = tool_client.message(content=tool_calls[tool_call]["Tool call message"], prompt_type="parameter_check", log=False)
                response = mscu.LiteralString(response)

                if "Status: PASS" in response:
                    response_passed = tool_client.message(content=tool_calls[tool_call]["Tool call message"], log=False, prompt_type="return_message_gen")
                    response_passed = mscu.LiteralString(response_passed)
                    full_response = response + "\nTool call response:" + response_passed
                    log_json = tool_client.chat_logger.compile_logs(message=tool_calls[tool_call]["Tool call message"], response=full_response)
                    log_json["conversation_id"] = tool_call
                    log_json["failure_mode"] = tool_calls[tool_call]["Failure mode"]
                    log_json["tool_parameters"] = tool_calls[tool_call]["Tool parameters"]
                    safe_filename = tool_call.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    with open(log_folder / f"{safe_filename}.json", "w") as f:
                        json.dump(log_json, f, indent=4)
                else: 
                    log_json = tool_client.chat_logger.compile_logs(message=tool_calls[tool_call]["Tool call message"], response=response)
                    log_json["conversation_id"] = tool_call
                    log_json["failure_mode"] = tool_calls[tool_call]["Failure mode"]
                    log_json["tool_parameters"] = tool_calls[tool_call]["Tool parameters"]
                    safe_filename = tool_call.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    with open(log_folder / f"{safe_filename}.json", "w") as f:
                        json.dump(log_json, f, indent=4)

        except (ValueError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error: {e} in {tool_file_name}")
            number_problems += 1
            continue
        except Exception as e:
            print(f"Unexpected error: {e} in {tool_file_name}")
            number_problems += 1
            continue
        
        print(f"\n{separator * 2}\n")
    print(f"Number of problems: {number_problems}")


def main():    
    mode = input("Choose mode (1 for Simulated tools, 2 for Acebench tools): ")
    
    if mode == "1":
        run_simulated_tools()
    elif mode == "2":
        run_acebench_tools()
    else:
        print("Invalid mode selected. Please choose 1 or 2.")

if __name__ == "__main__":
    main()
