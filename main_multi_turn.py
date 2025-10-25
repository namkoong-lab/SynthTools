#!/usr/bin/env python3
"""
Automated Multi-Turn Agent

Replaces the human REPL in `multi_turn_human.py` with an LLM "agent" that:
- Reads the task + tools from curated task YAML(s)
- Uses full conversational history as role-structured messages
- Emits either a single tool call or a final <ATTEMPT> block
- Stops when solution is achieved (if ground truth present) or when max turns reached
- Processes multiple tasks sequentially when multiple task files are provided

Key features:
- Loads curated task YAML(s) containing `meta_data` and `tools_data`
- Caches tool simulator clients per tool to avoid re-initialization cost
- Uses role-based conversation history (user/assistant messages)
- Logs prompts and responses to a YAML log file if `--log_folder` is provided
- Optionally verifies `<ATTEMPT>...</ATTEMPT>` blocks against ground truth
- Processes tasks sequentially with clear separation between tasks
- Continues processing remaining tasks even if one task fails

Usage:
  # Using configuration file (recommended):
  python main_multi_turn.py --config configs/multi_turn_config.yml

  # Using command line arguments:
  python main_multi_turn.py \
    tool_content2/task_meta/output/task1.yml \
    tool_content2/task_meta/output/task2.yml \
    --metadata_folder tool_content2/tool_meta \
    --log_folder tool_content2/task_simulation \
    --max_interactions 20

  # Single task with command line:
  python main_multi_turn.py \
    tool_content2/task_meta/output/task1.yml \
    --metadata_folder tool_content2/tool_meta \
    --log_folder tool_content2/task_simulation
"""

import sys
from pathlib import Path
import json
import argparse
import yaml
import re
from io import StringIO
from typing import List
import tempfile

from simulator.simulate_tool import AnthropicToolClient
from utils.misc_utils import LiteralString, setup_yaml

setup_yaml()

from scripts import multi_turn_human
from scripts.multi_turn_human import (
    filter_metadata,
    build_multi_turn_user_prompt,
    extract_attempt_blocks,
    extract_all_function_calls,
    parse_tool_call_input,
    extract_response_to_agent,
    append_to_log as _base_append_to_log,
    extract_solution_calls,
    normalized_string,
    normalize_tool_call_string,
    dump_yaml_readable,
    normalize_name,
)

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.scalarstring import LiteralScalarString
    _RUAMEL_AVAILABLE = True
except Exception:
    _RUAMEL_AVAILABLE = False

def _require_ruamel():
    if not _RUAMEL_AVAILABLE:
        raise RuntimeError(
            "ruamel.yaml is required to emit literal block scalars. Install with: pip install ruamel.yaml"
        )

def _get_yaml_writer():
    _require_ruamel()
    y = YAML()
    y.indent(mapping=2, sequence=2, offset=2)
    y.width = 1000000
    y.preserve_quotes = False
    return y

def _as_block(s: str):
    """Wrap text so ruamel emits it with '|' style. Do not touch chomping here."""
    _require_ruamel()
    return LiteralScalarString("" if s is None else str(s))

_BIG_TEXT_KEYS = {
    "multi_turn_prompt",
    "agent_system_prompt",
    "agent_messages_init",
    "agent_text",
    "prompt_simulator",
    "response_simulator",
    "response_to_agent",
    "attempt",
    "verification",
    "agent_error",
    "agent_parse_error",
    "guidance_to_agent",
    "tool_error",
}

def _coerce_blocks_for_all_interactions(data: dict) -> dict:
    """
    Convert large or multiline string fields under any interaction_* to literal blocks.
    """
    for k, v in list(data.items()):
        if not isinstance(k, str) or not k.startswith("interaction_"):
            continue
        inter = v
        if not isinstance(inter, dict):
            continue
        for subk, subv in list(inter.items()):
            if isinstance(subv, str):
                if subk in _BIG_TEXT_KEYS or "\n" in subv or len(subv) > 80:
                    inter[subk] = _as_block(subv)
    return data

def _add_chomp_indicators(text: str) -> str:
    """
    Change every 'key: |' header line to 'key: |-' so blocks do not add a trailing newline.
    Applied to the full document; safe for our logs.
    """
    return re.sub(r'^(\s*[^:\n]+:\s*)\|\s*$', r'\1|-', text, flags=re.MULTILINE)

def _yaml_dump_to_file(data: dict, out_path: Path):
    """Dump with ruamel, then post process to add '|-' on block headers."""
    _require_ruamel()
    yaml_writer = _get_yaml_writer()
    sio = StringIO()
    yaml_writer.dump(data, sio)
    raw = sio.getvalue()
    final = _add_chomp_indicators(raw)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(final)

def append_to_log(log_file: Path, interaction_num: int, key: str, value: str):
    """Ensure ALL interactions have big fields dumped as literal blocks with '|-'."""
    lf = _base_append_to_log(log_file, interaction_num, key, value)
    try:
        with open(lf, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        data = _coerce_blocks_for_all_interactions(data)
        _yaml_dump_to_file(data, lf)
    except Exception:
        pass
    return lf

def _load_anthropic_api_key() -> str:
    """Load Anthropic API key using the same approach as other scripts (configs/api_keys.json)."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "configs" / "api_keys.json"
    try:
        with open(config_path) as f:
            api_keys = json.load(f)
        return api_keys.get("anthropic")
    except Exception as e:
        raise RuntimeError(f"Failed to load API key from {config_path}: {e}")

class AgentClaudeClient:
    """Anthropic client wrapper aligned with simulate_tool configuration style."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 2000, temperature: float = 0.0):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed. pip install anthropic") from e
        api_key = _load_anthropic_api_key()
        if not api_key:
            raise RuntimeError("Anthropic API key not found in configs/api_keys.json under key 'anthropic'.")
        self._anthropic = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete_messages(self, system_prompt: str, messages: list[dict]) -> str:
        if not messages or not (messages[0].get("content") or "").strip():
            raise ValueError("First user message content is empty. Aborting request.")
        resp = self._anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=messages,
        )
        parts: List[str] = []
        for block in resp.content or []:
            t = getattr(block, "type", None)
            if t == "text":
                parts.append(getattr(block, "text", ""))
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()

def find_tool_metadata_file(metadata_folder: Path, tool_name: str, spec_id: str = None):
    """
    Find the metadata file for a given tool name in the metadata folder.
    
    The naming pattern is: {domain}_tool_spec_{id}__{Tool_Name}__output.json
    For example: ecommerce_and_retail_tool_spec_43__Refund_Calculator__output.json
    
    Args:
        metadata_folder: Path to folder containing tool metadata files
        tool_name: The tool name as it appears in the task YAML (e.g., "Refund Calculator")
        spec_id: Optional spec ID to filter results (e.g., "43" from "tool_spec_43")
    
    Returns:
        Path to the metadata file, or None if not found
    """
    tool_name_for_file = tool_name.replace(" ", "_")
    
    pattern = f"*__{tool_name_for_file}__output.json"
    matching_files = list(metadata_folder.glob(pattern))
    
    if spec_id and matching_files:
        spec_pattern = f"_spec_{spec_id}__"
        filtered_files = [f for f in matching_files if spec_pattern in f.name]
        if filtered_files:
            matching_files = filtered_files
        elif matching_files:
            print(f"⚠️ No metadata file found with spec_{spec_id} for tool {tool_name}, using any match")
    
    if not matching_files:
        print(f"⚠️ No metadata file found for tool: {tool_name} (searched for: {pattern})")
        return None
    
    if len(matching_files) > 1:
        print(f"⚠️ Multiple metadata files found for tool {tool_name}, using first: {matching_files[0].name}")
    
    return matching_files[0]


def get_or_create_client(selected_tool_name: str, client_cache: dict, tools_registry: dict, filtered_metadata_file: Path, metadata_folder: Path = None, spec_id: str = None):
    """Return (client, tool_config) for the given tool, creating and caching if needed."""
    if selected_tool_name in client_cache:
        client, _tool_tempfile = client_cache[selected_tool_name]
        tool_config = tools_registry[normalize_name(selected_tool_name)]['tool_json']
        return client, tool_config

    tool_config = tools_registry[normalize_name(selected_tool_name)]['tool_json']
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(tool_config, tmp, indent=2)
    tmp.close()
    tool_tempfile = Path(tmp.name)

    tool_metadata_file = filtered_metadata_file
    
    if metadata_folder:
        tool_specific_metadata = find_tool_metadata_file(metadata_folder, selected_tool_name, spec_id)
        if tool_specific_metadata:
            with open(filtered_metadata_file, 'r') as f:
                base_metadata = json.load(f)
            with open(tool_specific_metadata, 'r') as f:
                tool_metadata = json.load(f)
            
            merged = base_metadata.copy()
            merged.update(tool_metadata)
            
            tmp_meta = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(merged, tmp_meta, indent=2)
            tmp_meta.close()
            tool_metadata_file = Path(tmp_meta.name)
            print(f"Using tool-specific metadata for: {selected_tool_name}")
        else:
            print(f"⚠️ No tool-specific metadata found for {selected_tool_name}, using base metadata only")

    script_dir = Path(__file__).resolve().parent
    metadata_template_path = script_dir / "prompt_templates" / "tool_simulator" / "tool_simulator_template_metadata.yml"

    client = AnthropicToolClient(
        tool_config_file=tool_tempfile,
        metadata_tool_file=tool_metadata_file,
        using_states=False,
        combined_prompt_template_file=str(metadata_template_path)
    )
    client_cache[selected_tool_name] = (client, tool_tempfile)
    return client, tool_config


def load_agent_system_prompt(template_path: str = None) -> str:
    """Load the agent system prompt from the specified template file."""
    script_dir = Path(__file__).resolve().parent
    
    if template_path:
        # Use provided template path
        if not Path(template_path).is_absolute():
            template_file = script_dir / template_path
        else:
            template_file = Path(template_path)
    else:
        # Use default template
        template_file = script_dir / "prompt_templates" / "multi_turn_task_simulation" / "multi_turn_template.yml"
    
    with open(template_file, 'r', encoding='utf-8') as f:
        template_yaml = yaml.safe_load(f)
    return template_yaml['template']

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Automated multi turn agent LLM drives tool calls until solution or cap")
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument("task_files", nargs="*", type=str, help="Path(s) to task YAML file(s) containing meta_data and tools_data (if not using --config)")
    parser.add_argument("--metadata_folder", type=str, help="Path to folder containing tool-specific metadata files (e.g., tool_meta/)")
    parser.add_argument("--log_folder", type=str, default=None, help="Optional folder to save simulation logs")
    parser.add_argument("--order_matters", action="store_true", default=False, help="Whether solution comparison enforces order default false")
    parser.add_argument("--max_interactions", type=int, default=20, help="Maximum agent tool interactions before stopping")
    parser.add_argument("--agent_model", type=str, default="claude-sonnet-4-20250514", help="Agent model name")
    parser.add_argument("--agent_max_tokens", type=int, default=2000, help="Max tokens for agent reply")
    parser.add_argument("--agent_temperature", type=float, default=0.0, help="Agent sampling temperature")
    parser.add_argument("--prompt_template", type=str, help="Path to prompt template file")

    args = parser.parse_args()

    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
        
        # Override command line args with config values
        task_files = [Path(f) for f in config.get('task_files', [])]
        log_folder = Path(config.get('log_folder')) if config.get('log_folder') else None
        metadata_folder = Path(config.get('metadata_folder')) if config.get('metadata_folder') else None
        order_matters = config.get('order_matters', False)
        max_interactions = config.get('max_interactions', 20)
        agent_model = config.get('agent_model', 'claude-sonnet-4-20250514')
        agent_max_tokens = config.get('agent_max_tokens', 2000)
        agent_temperature = config.get('agent_temperature', 0.0)
        prompt_template = config.get('prompt_template')
    else:
        # Use command line arguments
        if not args.task_files:
            print("Error: Must provide either --config or task_files as positional arguments")
            sys.exit(1)
        task_files = [Path(f) for f in args.task_files]
        log_folder = Path(args.log_folder) if args.log_folder else None
        metadata_folder = Path(args.metadata_folder) if args.metadata_folder else None
        order_matters = args.order_matters
        max_interactions = args.max_interactions
        agent_model = args.agent_model
        agent_max_tokens = args.agent_max_tokens
        agent_temperature = args.agent_temperature
        prompt_template = args.prompt_template

    for task_file in task_files:
        if not task_file.exists():
            print(f"Task file not found: {task_file}")
            sys.exit(1)

    if metadata_folder and not metadata_folder.exists():
        print(f"Metadata folder not found: {metadata_folder}")
        sys.exit(1)

    for task_idx, task_file in enumerate(task_files):
        print(f"\n{'='*80}")
        print(f"PROCESSING TASK {task_idx + 1}/{len(task_files)}: {task_file.name}")
        print(f"{'='*80}")
        
        try:
            with open(task_file, "r", encoding='utf-8') as f:
                task_yaml = yaml.safe_load(f) or {}

            solution_reference = None
            if "solution" in task_yaml:
                solution_reference = task_yaml["solution"]
                del task_yaml["solution"]

            filtered_metadata, filtered_metadata_file = filter_metadata(task_file)
            
            tool_call_history = []
            
            spec_id = None
            if "tool_spec_" in task_file.stem:
                match = re.search(r'tool_spec_(\d+)', task_file.stem)
                if match:
                    spec_id = match.group(1)
                    print(f"Detected tool spec ID: {spec_id}")
            
            if metadata_folder:
                print(f"Metadata folder provided: {metadata_folder}")
                print("   Tool-specific metadata will be loaded dynamically for each tool call")
            else:
                print("⚠️ No metadata folder provided, using filtered metadata from task file only")

            tools_registry = {}
            for entry in (task_yaml.get("tools_data") or []):
                tool_json = entry.get("tool_json") or {}
                tool_name = tool_json.get("tool_name") or entry.get("tool_name")
                if not tool_name or not tool_json:
                    continue
                normalized = normalize_name(tool_name)
                tools_registry[normalized] = {
                    "tool_name": tool_name,
                    "tool_json": tool_json,
                }

            client_cache = {}

            def get_log_file_path(log_folder: Path, task_file: Path) -> Path:
                log_folder.mkdir(parents=True, exist_ok=True)
                return log_folder / f"{task_file.stem}.log.yml"

            log_file = None
            _pending_log_data = None
            if log_folder:
                log_file = get_log_file_path(log_folder, task_file)
                if log_file.exists():
                    log_file.unlink()
                print(f"Logging to: {log_file}")

            print(f"Loaded task: {task_file.name}")
            print("Tools available: " + ", ".join([v["tool_name"] for v in tools_registry.values()]))
            
            if solution_reference is None:
                print("⚠️ No ground truth solution available, solution verification disabled")
            else:
                print("Ground truth solution available, verification enabled")

            try:
                task_and_tools_header = build_multi_turn_user_prompt(task_yaml, prompt_template)
                if not task_and_tools_header:
                    raise ValueError("Empty multi turn user prompt was produced.")
                if log_file:
                    _log_data = {}
                    ikey = "interaction_0"
                    _log_data[ikey] = {}
                    _log_data[ikey]["multi_turn_prompt"] = _as_block(task_and_tools_header)
                    _pending_log_data = _log_data
                else:
                    _pending_log_data = None
            except Exception as e:
                print(f"⚠️ Could not build multi turn prompt: {e}")
                task_and_tools_header = ""

            agent_system_prompt = load_agent_system_prompt(prompt_template)
            if log_file:
                try:
                    if _pending_log_data is not None:
                        _pending_log_data["interaction_0"]["agent_system_prompt"] = _as_block(agent_system_prompt)
                        _pending_log_data = _coerce_blocks_for_all_interactions(_pending_log_data)
                        _yaml_dump_to_file(_pending_log_data, log_file)
                    else:
                        data = {"interaction_0": {"agent_system_prompt": _as_block(agent_system_prompt)}}
                        data = _coerce_blocks_for_all_interactions(data)
                        _yaml_dump_to_file(data, log_file)
                except Exception as e:
                    print(f"⚠️ Failed writing prompts to log: {e}")

            agent = AgentClaudeClient(
                model=agent_model,
                max_tokens=agent_max_tokens,
                temperature=agent_temperature,
            )

            messages: list[dict] = []
            if task_and_tools_header:
                messages.append({"role": "user", "content": task_and_tools_header})
            else:
                print("Aborting because the first user message would be empty.")
                continue

            if log_file:
                append_to_log(log_file, 0, "agent_messages_init", json.dumps(messages, indent=2))
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        _all = yaml.safe_load(f) or {}
                    _all = _coerce_blocks_for_all_interactions(_all)
                    _yaml_dump_to_file(_all, log_file)
                except Exception:
                    pass

            interaction_count = 0
            task_completed = False
            while interaction_count < max_interactions and not task_completed:
                interaction_count += 1

                agent_text = agent.complete_messages(agent_system_prompt, messages)
                if log_file:
                    append_to_log(log_file, interaction_count, "agent_text", agent_text)

                messages.append({"role": "assistant", "content": agent_text})

                attempt_blocks = extract_attempt_blocks(agent_text)
                if attempt_blocks:
                    attempt_text = "\n\n".join(attempt_blocks)
                    if log_file:
                        append_to_log(log_file, interaction_count, "attempt", attempt_text)

                    if solution_reference is not None:
                        candidate_calls = extract_all_function_calls(attempt_text)
                        stripped_attempt = re.sub(r"\s+", "", attempt_text)
                        only_calls_stripped = re.sub(r"\s+", "", "".join(candidate_calls))
                        has_extraneous = stripped_attempt != only_calls_stripped

                        gt_calls_raw = extract_solution_calls(solution_reference)
                        gt_norm = [normalized_string(normalize_tool_call_string(s)) for s in gt_calls_raw]
                        cand_norm = [normalized_string(normalize_tool_call_string(s)) for s in candidate_calls]

                        is_correct = False
                        if not has_extraneous:
                            if order_matters:
                                is_correct = (gt_norm == cand_norm)
                            else:
                                is_correct = (sorted(gt_norm) == sorted(cand_norm))

                        if is_correct:
                            msg = "Solution is correct."
                            print(msg)
                            if log_file:
                                append_to_log(log_file, interaction_count, "verification", msg)
                            task_completed = True
                            break
                        else:
                            msg = (
                                "Solution is wrong. Remember that order matters and that nothing else should be within the attempt "
                                "besides tool calls with the right parameters, parameter order, and avoiding optional parameters unless necessary."
                            )
                            print(msg)
                            messages.append({"role": "user", "content": msg})
                            if log_file:
                                append_to_log(log_file, interaction_count, "verification", msg)
                            continue
                    else:
                        msg = "Attempt received but no ground truth available to verify. Continuing..."
                        print(msg)
                        messages.append({"role": "user", "content": msg})
                        if log_file:
                            append_to_log(log_file, interaction_count, "verification", msg)
                        continue

                parsed = parse_tool_call_input(agent_text, tools_registry, n=1)
                if not parsed:
                    print("No valid tool call or attempt detected from agent. Continuing...")
                    guidance_msg = (
                        "No tool call or solution block detected. "
                        "Provide a valid function call (e.g., ToolName(...)) or wrap a sequence in <ATTEMPT> ... </ATTEMPT>."
                    )
                    messages.append({"role": "user", "content": guidance_msg})
                    if log_file:
                        append_to_log(log_file, interaction_count, "agent_parse_error", agent_text)
                        append_to_log(log_file, interaction_count, "guidance_to_agent", guidance_msg)
                    continue

                selected_tool_name, tool_call_or_response, is_error = parsed
                if is_error:
                    normalized_available = [v["tool_name"].replace(" ", "") for v in tools_registry.values()]
                    requested_normalized = (selected_tool_name or "").replace(" ", "")
                    available_tools_str = ', '.join([f'"{name}"' for name in normalized_available])
                    normalized_error = (
                        "Status Code: 500\n"
                        "Response: {\n"
                        "  \"error\": \"Tool not available\",\n"
                        f"  \"message\": \"The requested tool '{requested_normalized}' is not available in the current environment.\",\n"
                        "  \"error_code\": \"TOOL_NOT_FOUND\",\n"
                        f"  \"requested_tool\": \"{requested_normalized}\",\n"
                        f"  \"available_tools\": [{available_tools_str}]\n"
                        "}\n"
                    )
                    messages.append({"role": "user", "content": normalized_error})
                    if log_file:
                        append_to_log(log_file, interaction_count, "agent_error", normalized_error)
                    continue

                tool_call = tool_call_or_response

                try:
                    client, tool_config = get_or_create_client(
                        selected_tool_name=selected_tool_name,
                        client_cache=client_cache,
                        tools_registry=tools_registry,
                        filtered_metadata_file=filtered_metadata_file,
                        metadata_folder=metadata_folder,
                        spec_id=spec_id,
                    )

                    history_context = "\n".join(tool_call_history) if tool_call_history else ""
                    full_content = f"{history_context}\n{tool_call}" if history_context else tool_call
                    
                    if log_file:
                        prompt_used = client.combined_prompt + "\n" + full_content
                        append_to_log(log_file, interaction_count, "prompt_simulator", prompt_used)
                    
                    response_full = client.message(content=full_content, prompt_type="combined", log=False)

                    response_to_agent = extract_response_to_agent(response_full)
                    tool_call_history.append(f"Tool call: {tool_call}\nResponse: {response_to_agent}")
                    print(response_to_agent)
                    if log_file:
                        append_to_log(log_file, interaction_count, "response_simulator", response_full)
                        append_to_log(log_file, interaction_count, "response_to_agent", response_to_agent)

                    messages.append({"role": "user", "content": response_to_agent})

                except Exception as e:
                    err_msg = f"Error executing tool: {e}"
                    print(err_msg)
                    messages.append({"role": "user", "content": err_msg})
                    if log_file:
                        append_to_log(log_file, interaction_count, "tool_error", str(e))

            if not task_completed:
                print(f"Reached max interactions ({max_interactions}) for task {task_file.name}. Moving to next task.")
            
            print(f"Completed task {task_idx + 1}/{len(task_files)}: {task_file.name}")
            
        except Exception as e:
            print(f"Error processing task {task_file.name}: {e}")
            print(f"Continuing with next task...")
            continue

    print(f"\n{'='*80}")
    print(f"FINISHED PROCESSING ALL {len(task_files)} TASKS")
    print(f"{'='*80}")
    return

if __name__ == "__main__":
    main()