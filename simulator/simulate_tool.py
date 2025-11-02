import argparse
import json
import os
import time
import uuid
from pathlib import Path
import utils.misc_utils as mscu

import yaml
from anthropic import Anthropic
from openai import OpenAI

# Configuration constants
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
        self.log_config_file = None  # Disabled logging
        if log_config_file:
            self.log_config_file = log_config_file
        self.project_name = kwargs.get("project_name", "ToolSimulator")
        self.chat_configs = kwargs.get("chat_configs", {})

        if self.log_config_file and log_config is None:
            with open(self.log_config_file) as f:
                self.log_config = json.load(f)

        if self.log_config:
            self._update_relative_paths()
            self._validate_log_config()

        self.index_format = "{:08d}_{}.json"

        if self.log_config:
            print(f"Index logging folders: {list(self.index_logging_folders)}")
            print(f"ID logging folders: {list(self.id_logging_folders)}")
        else:
            self.index_logging_folders = {}
            self.id_logging_folders = {}

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
        script_root = Path(__file__).resolve().parent.parent  # SynthTools root directory

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

        # Delete invalid index logging folders
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
        proj_root = script_dir.parent  # SynthTools root directory
        self.is_task_simulation = kwargs.get("is_task_simulation", False)
        
        self.tool_config_file = kwargs.get(
            "tool_config_file", f"{proj_root}/tool_configs/example_wifi_tool.json"
        )
        # Support both naming conventions
        self.mega_data_tool_file = kwargs.get("mega_data_tool_file", None) or kwargs.get("metadata_tool_file", None)
        self.using_states = kwargs.get("using_states", False)
        self.load_tool_config()
        if self.mega_data_tool_file:
            self.load_mega_data_tool_config()

        if self.using_states:
            if "states" in kwargs:
                print(
                    f"Replacing tool states from {self.tool_config['states']} to {kwargs['states']}"
                )
                self.tool_config["states"] = kwargs["states"]
            else:
                print(f"Using default tool states: {self.tool_config['states']}")

        # Different template paths for task simulation vs tool evaluation
        if self.is_task_simulation:
            self.combined_prompt_template_file = kwargs.get(
                "combined_prompt_template_file",
                proj_root / "prompt_templates" / "tool_simulator" / "tool_simulator_template_metadata.yml",
            )
            self.parameter_check_prompt_template_file = kwargs.get(
                "parameter_check_prompt_template_file",
                proj_root / "prompt_templates" / "generate_tools" / "parameter_check.yml",
            )
            self.return_message_gen_prompt_template_file = kwargs.get(
                "return_message_gen_prompt_template_file",
                proj_root / "prompt_templates" / "judge_simulator" / "return_message_gen.yml",
            )
        else:
            self.combined_prompt_template_file = kwargs.get(
                "combined_prompt_template_file",
                proj_root / "prompt_templates" / "tool_simulator" / "tool_simulator_template.yml",
            )
            self.parameter_check_prompt_template_file = kwargs.get(
                "parameter_check_prompt_template_file",
                proj_root / "prompt_templates" / "generate_tools" / "parameter_check.yml",
            )
            self.return_message_gen_prompt_template_file = kwargs.get(
                "return_message_gen_prompt_template_file",
                proj_root / "prompt_templates" / "judge_simulator" / "return_message_gen.yml",
            )
        self.load_combined_prompt_template()
        self.load_parameter_check_prompt_template()
        self.load_return_message_gen_prompt_template()
        self._validate_tool_config()

        prompt_data = self.tool_config.copy()
        prompt_data["tool_details"]= str(self.tool_config)
        
        if hasattr(self, 'mega_data_tool_config') and self.mega_data_tool_config:
            if 'Tool Call' in self.mega_data_tool_config:
                prompt_data['tool_call'] = self.mega_data_tool_config['Tool Call']
            if 'Return Data' in self.mega_data_tool_config:
                prompt_data['return_data'] = self.mega_data_tool_config['Return Data']
            
            excluded_keys = {'Tool Call', 'Return Data'}
            context_metadata = {k: v for k, v in self.mega_data_tool_config.items() 
                              if k not in excluded_keys}
            prompt_data['metadata'] = context_metadata
            prompt_data['meta_data'] = context_metadata
            
            for key, value in self.mega_data_tool_config.items():
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

    def load_mega_data_tool_config(self):
        with open(self.mega_data_tool_file) as f:
            self.mega_data_tool_config = json.load(f)

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
            if field not in self.tool_config and not (hasattr(self, 'mega_data_tool_config') and field in self.mega_data_tool_config):
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
        proj_root = script_dir.parent  # SynthTools root directory
        config_path = proj_root / "configs" / "api_keys.json"
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
        elif prompt_type == "return_message_gen":
            msg_content = self.return_message_gen_prompt + "\n" + content
            print(f"Return message gen prompt:\n{msg_content}")
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
        script_dir = Path(__file__).resolve().parent
        proj_root = script_dir.parent  # SynthTools root directory
        if is_simulated:
            template_file = proj_root / "prompt_templates" / "generate_testing_functions_tool_simulator" / "generate_testing_tool_calls.yml"
        else:
            template_file = proj_root / "prompt_templates" / "generate_acebench_tool_calls" / "generate_successful_tool_calls.yml"
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
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent  # SynthTools root directory
    if tool_final is None:
        tool_final = DEFAULT_CONFIG["tool_final"]
    if tool_final:
        base_path = proj_root / "tool_content" / "tool_final"
        tool_call_files = base_path / "tool_json"
        mega_tool_files = base_path / "tool_meta"
    else:
        if tool_collection_version is None:
            tool_collection_version = DEFAULT_CONFIG["tool_collection_version"]
        base_path = proj_root / "tool_content" / "full_tool_specs"
        tool_call_files = base_path / f"tool_collection_json_{tool_collection_version}"
        mega_tool_files = base_path / f"tool_collection_meta_data_{tool_collection_version}"
    print("Running Simulated Tools mode...")

    if not tool_call_files.exists():
        print(f"Tool collection directory not found: {tool_call_files}")
        return
    if not mega_tool_files.exists():
        print(f"Meta data directory not found: {mega_tool_files}")
        return

    process_tool_files(tool_call_files, mega_tool_files, is_simulated=True)

def run_acebench_tools():
    print("Running Acebench Tools mode...")
    
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent  # SynthTools root directory
    tool_call_files = proj_root / "evaluation" / "acebench" / "data_en" 
    mega_tool_files = proj_root / "evaluation" / "acebench" / "data_en"
    
    process_tool_files(tool_call_files, mega_tool_files, is_simulated=False)

def process_tool_files(tool_call_files, mega_tool_files, is_simulated=True):
    separator = "*" * 42
    number_problems = 0
    count = 0
    for tool_file in tool_call_files.glob("*.json"):
        try:
            tool_file_name = tool_file.stem

            print(f"\n{separator * 2}\n")
            print(f"\nTool file name: {tool_file_name}")
            
            if is_simulated:
                mega_tool_file = mega_tool_files / f"{tool_file_name}__output.json"
            else:
                mega_tool_file = tool_file

            log_folder = tool_file.parent.parent / "tool_eval_logs" / tool_file.stem
            log_folder.mkdir(parents=True, exist_ok=True)
            print(f"Made log folder: {log_folder}")

            existing_tool_calls = list(log_folder.glob("Tool_call_*.json"))
            if len(existing_tool_calls) > 5:
                print(f"Tool calls already generated for {tool_file_name}")
                count += 1
                continue
                
            tool_client = AnthropicToolClient(tool_config_file=tool_file, mega_data_tool_file=mega_tool_file, using_states=False)

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
    print(f"Number of tool files processed: {count}")
    print(f"Number of problems: {number_problems}")



def process_tool_files_with_meta_data(tool_call_files, mega_tool_files, output_dir=None, is_simulated=True):
    separator = "*" * 42
    number_problems = 0
    count = 0
    for tool_file in tool_call_files.glob("*.json"):
        try:
            tool_file_name = tool_file.stem

            print(f"\n{separator * 2}\n")
            print(f"\nTool file name: {tool_file_name}")
            
            mega_tool_file = mega_tool_files / f"{tool_file_name}__output.json"
            # Use provided output_dir or default to tool_content/tool_eval_logs
            if output_dir:
                log_folder = Path(output_dir) / tool_file.stem
            else:
                script_dir = Path(__file__).resolve().parent
                proj_root = script_dir.parent  # SynthTools root directory
                log_folder = proj_root / "tool_content" / "tool_eval_logs" / tool_file.stem
            log_folder.mkdir(parents=True, exist_ok=True)
            print(f"Made log folder: {log_folder}")

            tool_client = AnthropicToolClient(tool_config_file=tool_file, mega_data_tool_file=mega_tool_file, using_states=False)
            print(f"Initialized tool client")
            
            meta_data_tool_calls_dir = log_folder / "meta_data_tool_call.json"


            import re
            tool_name_match = re.search(r"_tool_spec_(\d+)__", tool_file_name)
            if tool_name_match:
                tool_name = tool_file_name.split(tool_name_match.group(0))[1]
            else:
                raise ValueError(f"Tool name not found in {tool_file_name}")
            print(f"Tool name: {tool_name}")

            with open(mega_tool_file, "r") as f:
                meta_data= json.load(f)
                meta_data_tool_call = meta_data.get("Tool Call", {})
                meta_data_tool_call_message = f"{tool_name}[{', '.join([f'{param} = {value}' if not isinstance(value, str) else f'{param} = {repr(value)}'for param, value in meta_data_tool_call.items()])}]"
                with open(meta_data_tool_calls_dir, "w") as f:
                    meta_data_tool_call_json = {
                        "Tool call message": meta_data_tool_call_message,
                        "Tool parameters": meta_data_tool_call,
                        "Ground truth return data": meta_data.get("Return Data", {})
                    }
                    json.dump(meta_data_tool_call_json, f, indent=4)

            response = tool_client.message(content=meta_data_tool_call_message, prompt_type="parameter_check", log=False)
            response = mscu.LiteralString(response)

            if "Status: PASS" in response:
                response_passed = tool_client.message(content=meta_data_tool_call_message, log=False, prompt_type="return_message_gen")
                response_passed = mscu.LiteralString(response_passed)
                full_response = response + "\nTool call response:" + response_passed
                log_json = tool_client.chat_logger.compile_logs(message=meta_data_tool_call_message, response=full_response)
                with open(meta_data_tool_calls_dir, "r") as f:
                    existing_data = json.load(f)
                existing_data.update(log_json)
                with open(meta_data_tool_calls_dir, "w") as f:
                    json.dump(existing_data, f, indent=4)
            else:
                log_json = tool_client.chat_logger.compile_logs(message=meta_data_tool_call_message, response=response)
                with open(meta_data_tool_calls_dir, "a") as f:
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
    print(f"Number of tool files processed: {count}")
    print(f"Number of problems: {number_problems}")

def run_simulated_tools_with_meta_data(tool_json_dir=None, tool_meta_dir=None, output_dir=None):
    print("Running Simulated Tools with Meta Data mode...")
    
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent  # SynthTools root directory
    
    # Use provided paths or default to tool_content paths
    if tool_json_dir is None:
        tool_call_files = proj_root / "tool_content" / "tool_json"
    else:
        tool_call_files = Path(tool_json_dir)
    
    if tool_meta_dir is None:
        mega_tool_files = proj_root / "tool_content" / "tool_meta"
    else:
        mega_tool_files = Path(tool_meta_dir)
    
    process_tool_files_with_meta_data(tool_call_files, mega_tool_files, output_dir=output_dir, is_simulated=True)

def run_task_simulation(tool_final=None, tool_collection_version=None):
    """Run task simulation mode - logs in same directory as tool files"""
    print("Running Task Simulation mode...")
    
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent  # SynthTools root directory
    if tool_final is None:
        tool_final = DEFAULT_CONFIG["tool_final"]
    if tool_final:
        base_path = proj_root / "tool_content" / "tool_final"
        tool_call_files = base_path / "tool_json"
        metadata_tool_files = base_path / "tool_meta"
    else:
        if tool_collection_version is None:
            tool_collection_version = DEFAULT_CONFIG["tool_collection_version"]
        base_path = proj_root / "tool_content" / "full_tool_specs"
        tool_call_files = base_path / f"tool_collection_json_{tool_collection_version}"
        metadata_tool_files = base_path / f"tool_collection_meta_data_{tool_collection_version}"
    
    if not tool_call_files.exists():
        print(f"Tool collection directory not found: {tool_call_files}")
        return
    if not metadata_tool_files.exists():
        print(f"Meta data directory not found: {metadata_tool_files}")
        return

    process_tool_files_for_tasks(tool_call_files, metadata_tool_files, is_simulated=True)

def process_tool_files_for_tasks(tool_call_files, metadata_tool_files, is_simulated=True):
    """Process tool files for task simulation - logs in same directory as tool files"""
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
                
            tool_client = AnthropicToolClient(
                tool_config_file=tool_file, 
                metadata_tool_file=metadata_tool_file, 
                using_states=False,
                is_task_simulation=True
            )

            print(f"Initialized tool client")
            
            tool_calls_dir = tool_client.generate_test_tool_calls(
                tool_file, log_folder, 
                model_provider=DEFAULT_CONFIG["model_provider"], 
                is_simulated=is_simulated
            )
            tool_calls_dir = log_folder / "tool_calls.json"
            with open(tool_calls_dir, "r") as f:
                tool_calls = json.load(f)

            for tool_call in tool_calls:
                print(f"Tool call: {tool_calls[tool_call]}")
                response = tool_client.message(
                    content=tool_calls[tool_call]["Tool call message"], 
                    prompt_type="parameter_check", 
                    log=False
                )
                response = mscu.LiteralString(response)

                if "Status: PASS" in response:
                    response_passed = tool_client.message(
                        content=tool_calls[tool_call]["Tool call message"], 
                        log=False, 
                        prompt_type="return_message_gen"
                    )
                    response_passed = mscu.LiteralString(response_passed)
                    full_response = response + "\nTool call response:" + response_passed
                    log_json = tool_client.chat_logger.compile_logs(
                        message=tool_calls[tool_call]["Tool call message"], 
                        response=full_response
                    )
                    log_json["conversation_id"] = tool_call
                    log_json["failure_mode"] = tool_calls[tool_call]["Failure mode"]
                    log_json["tool_parameters"] = tool_calls[tool_call]["Tool parameters"]
                    safe_filename = tool_call.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    with open(log_folder / f"{safe_filename}.json", "w") as f:
                        json.dump(log_json, f, indent=4)
                else: 
                    log_json = tool_client.chat_logger.compile_logs(
                        message=tool_calls[tool_call]["Tool call message"], 
                        response=response
                    )
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
    parser = argparse.ArgumentParser(description="Simulate tool calls and responses")
    parser.add_argument("--mode", type=str, choices=["1", "2", "3", "4"], 
                        help="Simulation mode: 1=Simulated tools, 2=Acebench tools, 3=Simulated tools with meta data, 4=Task simulation")
    parser.add_argument("--tool_json_dir", type=str, default=None,
                        help="Directory containing tool JSON files (for mode 3)")
    parser.add_argument("--tool_meta_dir", type=str, default=None,
                        help="Directory containing tool metadata files (for mode 3)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for output logs (for mode 3)")
    args = parser.parse_args()
    
    if args.mode:
        mode = args.mode
    else:
        mode = input("Choose mode (1 for Simulated tools, 2 for Acebench tools, 3 for Simulated tools with meta data, 4 for Task simulation): ")
    
    if mode == "1":
        run_simulated_tools()
    elif mode == "2":
        run_acebench_tools()
    elif mode == "3":
        run_simulated_tools_with_meta_data(
            tool_json_dir=args.tool_json_dir,
            tool_meta_dir=args.tool_meta_dir,
            output_dir=args.output_dir
        )
    elif mode == "4":
        run_task_simulation()
    else:
        print("Invalid mode selected. Please choose 1, 2, 3, or 4.") 

if __name__ == "__main__":
    main()
