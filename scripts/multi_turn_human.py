#!/usr/bin/env python3
"""
Interactive Tool Simulator

Description:
  Runs an interactive REPL that takes function-call style inputs corresponding to
  a tool defined in a task YAML. It builds prompts using the tool spec and
  associated metadata, calls the model, and prints only the status code and JSON
  response for the agent. It also supports verifying solution attempts against
  ground truth solutions in the task YAML.

Key features:
  - Loads a curated task YAML containing `meta_data` and `tools_data`
  - Caches a client per tool to avoid re-initialization cost
  - Sends each interaction statelessly (no conversation history)
  - Logs prompts and responses to a YAML log file if `--log_folder` is provided
  - Optionally verifies `<ATTEMPT>...</ATTEMPT>` blocks against the ground truth

Usage (Anthropic):
  python adaptive-tool-use/scaling_tools/scripts/tool_simulator_terminal.py \
    adaptive-tool-use/scaling_tools/tool_content/tool_final/task_curated/<task>.yml \
    --log_folder adaptive-tool-use/scaling_tools/tool_content/tool_final/human_trajectories \
    --order_matters

Notes:
  - Input format expects a single function-call style tool invocation, e.g.
      ToolName(param1="value", param2=123)
  - To submit a solution attempt for auto-verification, wrap calls in:
      <ATTEMPT>
      ToolA(...)
      ToolB(...)
      </ATTEMPT>
"""

import sys
from pathlib import Path
import json
import argparse
import yaml
import re
from typing import List, Optional
from datetime import datetime

# Add the parent directory to the path to import AnthropicToolClient
sys.path.append(str(Path(__file__).parent.parent))

from simulator.simulate_tool import AnthropicToolClient
from utils.misc_utils import LiteralString
import yaml as yaml_module


class _LiteralDumper(yaml.SafeDumper):
    pass


def _literal_string_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data), style='|')


_LiteralDumper.add_representer(LiteralString, _literal_string_representer)


def extract_response_to_agent(response_simulator: str) -> str:
    """Extract only status code and JSON response from the full simulator response."""
    lines = response_simulator.split('\n')
    
    # Find status code line
    status_code = None
    for line in lines:
        if line.strip().startswith('Status Code:'):
            status_code = line.strip()
            break
    
    # Extract JSON response (everything between "Response:" and "Explanation:")
    response_start_idx = None
    response_end_idx = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('Response:'):
            response_start_idx = i
        elif line.strip().startswith('Explanation:'):
            response_end_idx = i
            break
    
    if response_start_idx is None:
        return response_simulator  # Return as-is if format is unexpected
    
    # Get response lines
    if response_end_idx is None:
        response_lines = lines[response_start_idx:]
    else:
        response_lines = lines[response_start_idx:response_end_idx]
    
    # Join status code and response
    result_parts = []
    if status_code:
        result_parts.append(status_code)
    result_parts.extend(response_lines)
    
    return '\n'.join(result_parts).strip()


def normalize_name(name: str) -> str:
    """Normalize a name by removing underscores/special chars and converting to lowercase."""
    import re
    # Remove all non-alphanumeric characters and convert to lowercase
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()


def extract_function_calls_from_text(text: str, n: int = 1, tool_name: str = None) -> List[str]:
    """
    Extract N function calls from text input.
    Format: FunctionName(param1=value1, param2=value2, ...)
    Handles function names with underscores like Agent_Matcher
    """
    import re
    
    function_calls = []
    i = 0
    text_len = len(text)
    
    while i < text_len and len(function_calls) < n:
        # Find potential function name start (letter or underscore)
        while i < text_len and not (text[i].isalpha() or text[i] == '_'):
            i += 1
        
        if i >= text_len:
            break
        
        # Extract function name (alphanumeric + underscores)
        name_start = i
        while i < text_len and (text[i].isalnum() or text[i] == '_'):
            i += 1
        
        function_name = text[name_start:i]
        
        # Skip whitespace
        while i < text_len and text[i].isspace():
            i += 1
        
        # Check for opening parenthesis
        if i >= text_len or text[i] != '(':
            continue
        
        # Extract the parameters (everything between parentheses)
        paren_start = i
        i += 1
        depth = 1
        in_string = False
        string_char = None
        escape = False
        
        while i < text_len and depth > 0:
            ch = text[i]
            
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif in_string:
                if ch == string_char:
                    in_string = False
            elif ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            
            i += 1
        
        if depth == 0:
            # Successfully found matching closing paren
            function_call = text[name_start:i]
            
            # If tool_name is provided, replace the extracted function name with the actual tool name
            if tool_name:
                # Replace the function name part with the actual tool name
                params_part = text[paren_start:i]  # Get the parameters part "(param1=value1, ...)"
                function_call = tool_name + params_part
            
            function_calls.append(function_call)
    
    return function_calls


def parse_tool_call_input(user_input: str, tools_registry: dict, n: int = 1) -> Optional[tuple[str, str, bool]]:
    """Parse user input, determine the target tool from tools_registry, and extract N tool calls.
    Returns a tuple (tool_name_or_error, function_call_string, is_error)."""
    import re

    # Find first potential function name
    match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', user_input)
    if not match:
        return None

    extracted_name = match.group(1)
    normalized_extracted = normalize_name(extracted_name)

    # Resolve against registry
    if normalized_extracted in tools_registry:
        canonical_tool_name = tools_registry[normalized_extracted]['tool_name']
        
        # Extract function call(s) and force the canonical tool name into the call
        function_calls = extract_function_calls_from_text(user_input, n=n, tool_name=canonical_tool_name)

        if not function_calls:
            return None

        # Return first (single tool flow)
        if n == 1:
            return canonical_tool_name, function_calls[0], False

        return canonical_tool_name, "\n\n".join(function_calls), False
    else:
        # Tool not found - return error response
        available_tools = ", ".join([v['tool_name'] for v in tools_registry.values()])
        error_response = f"""Status Code: 500
Response: {{
  "error": "Tool not available",
  "message": "The requested tool '{extracted_name}' is not available in the current environment.",
  "error_code": "TOOL_NOT_FOUND",
  "requested_tool": "{extracted_name}",
  "available_tools": [{', '.join([f'"{v["tool_name"]}"' for v in tools_registry.values()])}]
}}
"""
        return extracted_name, error_response, True

def filter_metadata(metadata_file: Path) -> tuple[dict, Path]:
    """Filter metadata to exclude meta-level fields, keep only context data.
    Returns both the filtered metadata dict and a temporary file path."""
    import tempfile
    
    # Read the metadata file
    with open(metadata_file, 'r') as f:
        if metadata_file.suffix in ['.yml', '.yaml']:
            raw_data = yaml.safe_load(f)
        else:
            raw_data = json.load(f)
    
    # Extract only meta_data if it exists, otherwise filter the top level
    if 'meta_data' in raw_data:
        # We have a structured file with meta_data section
        meta_data = raw_data['meta_data']
        
        # Filter out meta-level fields from meta_data
        excluded_fields = {
            'field_name', 'subfield', 'task', 
            'tool_budget', 'tool_sequences'
        }
        
        filtered_data = {
            key: value for key, value in meta_data.items() 
            if key not in excluded_fields
        }
    else:
        # Legacy format - filter at top level
        excluded_fields = {
            'field_name', 'subfield', 'task', 
            'tool_budget', 'tool_sequences'
        }
        
        filtered_data = {
            key: value for key, value in raw_data.items() 
            if key not in excluded_fields
        }
    
    # Create temporary filtered file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(filtered_data, temp_file, indent=2)
    temp_file.close()
    
    return filtered_data, Path(temp_file.name)


def build_metadata_prompt(tool_config: dict, metadata: dict) -> str:
    """Build a prompt using the metadata-aware template."""
    # Load the metadata template
    script_dir = Path(__file__).resolve().parent.parent
    template_path = script_dir / "prompt_templates" / "tool_simulator_template_metadata.yml"
    
    with open(template_path, 'r') as f:
        template_yaml = yaml_module.safe_load(f)
    
    # Format the template with tool config and metadata
    prompt = template_yaml['template'].format(
        tool_name=tool_config.get('tool_name', 'Unknown'),
        tool_description=tool_config.get('tool_description', ''),
        parameters=tool_config.get('parameters', {}),
        error_messages=tool_config.get('error_messages', []),
        usage=tool_config.get('usage', ''),
        metadata=json.dumps(metadata, indent=2)
    )
    
    return prompt


def get_or_create_client(selected_tool_name: str, client_cache: dict, tools_registry: dict, filtered_metadata_file: Path):
    """Return (client, tool_config) for the given tool, creating and caching if needed."""
    if selected_tool_name in client_cache:
        client, _tool_tempfile = client_cache[selected_tool_name]
        tool_config = tools_registry[normalize_name(selected_tool_name)]['tool_json']
        return client, tool_config

    # Create a temp file for the tool JSON config (client expects a file)
    import tempfile
    tool_config = tools_registry[normalize_name(selected_tool_name)]['tool_json']
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(tool_config, tmp, indent=2)
    tmp.close()
    tool_tempfile = Path(tmp.name)

    client = AnthropicToolClient(
        tool_config_file=tool_tempfile,
        metadata_tool_file=filtered_metadata_file,
        using_states=False
    )
    client_cache[selected_tool_name] = (client, tool_tempfile)
    return client, tool_config


def build_prompt_for_tool(tool_config: dict, metadata: dict, tool_call: str) -> str:
    """Compose the final prompt text for the model for a single tool call."""
    return build_metadata_prompt(tool_config, metadata) + "\n" + tool_call


def _indent_block(text: str, spaces: int = 2) -> str:
    """Indent every line in a multi-line string by a fixed number of spaces."""
    if not text:
        return ""
    prefix = " " * spaces
    return "\n".join(prefix + line if line else prefix for line in text.splitlines())


def build_multi_turn_user_prompt(task_yaml: dict, template_path: str = None) -> str:
    """Build the initial multi-turn prompt from task_data.task and tools_data using the template."""
    # Resolve template path
    script_dir = Path(__file__).resolve().parent.parent
    if template_path:
        if not Path(template_path).is_absolute():
            template_file = script_dir / template_path
        else:
            template_file = Path(template_path)
    else:
        template_file = script_dir / "prompt_templates" / "multi_turn_task_simulation" / "multi_turn_template.yml"

    with open(template_file, 'r') as f:
        template_yaml = yaml_module.safe_load(f)

    template_text: str = template_yaml["template"]

    # Extract task and tools data
    task_text = (task_yaml.get("task_data") or {}).get("task", "")
    tools_data = task_yaml.get("tools_data") or []

    # Pretty print tools_data as YAML and indent to align with template
    tools_yaml = yaml.dump(
        tools_data,
        Dumper=_LiteralDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=4096,
    )

    # Ensure both task and tools blocks are indented to match the template placeholders
    indented_task = _indent_block(task_text, spaces=2)
    indented_tools = _indent_block(tools_yaml.rstrip(), spaces=2)

    # Simple placeholder substitution
    prompt_text = template_text.replace("{{task}}", indented_task).replace("{{tools_data}}", indented_tools)

    return prompt_text


def call_llm_with_prompt(client, prompt_used: str, tool_call: str) -> str:
    """Send the prompt to the model using the best-available client path and return full response text."""
    if hasattr(client, 'client') and hasattr(client.client, 'messages'):
        response_obj = client.client.messages.create(
            model=client.model,
            max_tokens=client.max_tokens,
            messages=[{"role": "user", "content": prompt_used}],
        )
        return client.extract_text(response_obj)
    # Fallback to regular client call
    return client.message(
        content=tool_call,
        prompt_type="combined",
        log=False
    )


def get_log_file_path(log_folder: Path, task_file: Path) -> Path:
    """Get the log file path for a given task file."""
    log_folder.mkdir(parents=True, exist_ok=True)
    task_stem = task_file.stem
    return log_folder / f"{task_stem}.log.yml"


def wrap_strings_in_literal(data):
    """Recursively wrap all string values in LiteralString for readable YAML output."""
    if isinstance(data, dict):
        return {k: wrap_strings_in_literal(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [wrap_strings_in_literal(item) for item in data]
    elif isinstance(data, str):
        return LiteralString(data)
    else:
        return data


def dump_yaml_readable(data: dict) -> str:
    """Dump YAML data in readable format with block scalars."""
    # Wrap all strings in LiteralString
    wrapped_data = wrap_strings_in_literal(data)
    
    # Dump to string using custom dumper
    text = yaml.dump(
        wrapped_data,
        Dumper=_LiteralDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=4096,
    )
    
    # Convert '|' to '|-' for all string keys to match repo style
    # Match any key followed by: |
    pattern = r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*):\s*\|$'
    text = re.sub(
        pattern,
        r'\1\2: |-',
        text,
        flags=re.MULTILINE,
    )
    
    return text


def append_to_log(log_file: Path, interaction_num: int, key: str, value: str):
    """Append a key-value pair to the log file in real-time for a specific interaction."""
    # Read existing data if file exists
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = yaml.safe_load(f) or {}
            except:
                log_data = {}
    else:
        log_data = {}
    
    # Create interaction key if it doesn't exist
    interaction_key = f"interaction_{interaction_num}"
    if interaction_key not in log_data:
        log_data[interaction_key] = {}
    
    # Update with new key-value
    log_data[interaction_key][key] = value
    
    # Dump using readable format
    text = dump_yaml_readable(log_data)
    
    # Write back to file
    with open(log_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    
    return log_file


def extract_attempt_blocks(text: str) -> list[str]:
    """Extract attempt blocks delimited by <ATTEMPT> ... <ATTEMPT> or <ATTEMPT_START> ... <ATTEMPT_END>."""
    blocks: list[str] = []
    # Pattern 1: <ATTEMPT> ... <ATTEMPT>
    pattern1 = re.compile(r"<ATTEMPT>([\s\S]*?)<ATTEMPT>")
    # Pattern 2: <ATTEMPT_START> ... <ATTEMPT_END>
    pattern2 = re.compile(r"<ATTEMPT_START>([\s\S]*?)<ATTEMPT_END>")
    for m in pattern1.finditer(text):
        blocks.append(m.group(1).strip())
    for m in pattern2.finditer(text):
        blocks.append(m.group(1).strip())
    return blocks


def remove_attempt_blocks(text: str) -> str:
    """Remove attempt blocks from text, returning the cleaned string."""
    text = re.sub(r"<ATTEMPT>[\s\S]*?<ATTEMPT>", " ", text)
    text = re.sub(r"<ATTEMPT_START>[\s\S]*?<ATTEMPT_END>", " ", text)
    return text


def extract_all_function_calls(text: str) -> List[str]:
    """Extract all function-call-like strings from text using the same scanner logic."""
    calls: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        # find start of name
        while i < n and not (text[i].isalpha() or text[i] == '_'):
            i += 1
        if i >= n:
            break
        name_start = i
        while i < n and (text[i].isalnum() or text[i] == '_'):
            i += 1
        # skip ws
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != '(':
            continue
        # scan params with nesting
        j = i + 1
        depth = 1
        in_string = False
        string_char = None
        escape = False
        while j < n and depth > 0:
            ch = text[j]
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif in_string:
                if ch == string_char:
                    in_string = False
            elif ch in ('"', "'"):
                in_string = True
                string_char = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            j += 1
        if depth == 0:
            calls.append(text[name_start:j])
            i = j
        else:
            break
    return calls


def extract_solution_calls(solution_section) -> List[str]:
    """Extract solution tool calls from task YAML section (list of strings or dicts)."""
    if not solution_section:
        return []
    calls: List[str] = []
    for item in solution_section:
        if isinstance(item, str):
            calls.append(item)
        elif isinstance(item, dict) and len(item) == 1:
            (_, val) = next(iter(item.items()))
            if isinstance(val, str):
                calls.append(val)
    return calls


def normalize_tool_call_string(call_str: str) -> dict:
    """Normalize a function-call style string into {tool_name, parameters} without coercion."""
    call = call_str.strip()
    m = re.match(r"\s*([^((\s]+|[^()]+?)\s*\((.*)\)\s*$", call, flags=re.DOTALL)
    if not m:
        last_paren = call.rfind('(')
        if last_paren == -1 or not call.endswith(')'):
            return {"tool_name": call, "parameters": {}}
        tool_name = call[:last_paren].strip()
        params_body = call[last_paren + 1:-1]
    else:
        tool_name = m.group(1).strip()
        params_body = m.group(2)

    parts: List[str] = []
    buf = []
    depth_square = depth_brace = depth_paren = 0
    in_string = False
    escape = False
    for ch in params_body:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            buf.append(ch)
            continue
        if ch == '[':
            depth_square += 1
        elif ch == ']':
            depth_square = max(0, depth_square - 1)
        elif ch == '{':
            depth_brace += 1
        elif ch == '}':
            depth_brace = max(0, depth_brace - 1)
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        if ch == ',' and depth_square == depth_brace == depth_paren == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())

    params: dict = {}
    for part in parts:
        if not part or '=' not in part:
            continue
        k, v = part.split('=', 1)
        params[k.strip()] = v.strip()
    return {"tool_name": tool_name, "parameters": params}


def normalized_string(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, separators=(',', ':'))
    s = re.sub(r"\s+", "", s).lower()
    return s

def main():
    parser = argparse.ArgumentParser(description='Interactive tool simulator (multi-tool from task YAML)')
    parser.add_argument('task_file', type=str, help='Path to task YAML file containing meta_data and tools_data')
    parser.add_argument('--log_folder', type=str, default=None, help='Optional folder to save simulation logs')
    parser.add_argument('--order_matters', action='store_true', default=True, help='Whether solution comparison enforces order (default: true)')
    
    args = parser.parse_args()
    
    task_file = Path(args.task_file)
    log_folder = Path(args.log_folder) if args.log_folder else None
    
    # Validate files exist
    if not task_file.exists():
        print(f"❌ Task file not found: {task_file}")
        sys.exit(1)
        
    # Load full task yaml once
    with open(task_file, 'r') as f:
        task_yaml = yaml.safe_load(f)

    # Drop solution immediately to avoid contamination
    solution_reference = None
    if 'solution' in task_yaml:
        solution_reference = task_yaml['solution']
        del task_yaml['solution']

    # Extract and filter metadata from task file
    filtered_metadata, filtered_metadata_file = filter_metadata(task_file)

    # Build tools registry from tools_data
    tools_registry = {}
    for entry in (task_yaml.get('tools_data') or []):
        tool_json = entry.get('tool_json') or {}
        tool_name = tool_json.get('tool_name') or entry.get('tool_name')
        if not tool_name or not tool_json:
            continue
        normalized = normalize_name(tool_name)
        tools_registry[normalized] = {
            'tool_name': tool_name,
            'tool_json': tool_json,
        }

    # Cache: tool_name -> (client, tool_tempfile_path)
    client_cache = {}

    print(f"✅ Loaded task: {task_file.name}")
    print(f"✅ Tools available: " + ", ".join([v['tool_name'] for v in tools_registry.values()]) )
    
    # Set up log file if logging is enabled
    log_file = None
    if log_folder:
        log_file = get_log_file_path(log_folder, task_file)
        print(f"📁 Logging to: {log_file}")
        # Clear the log file for new session
        if log_file.exists():
            log_file.unlink()
    
    print("\n" + "="*50)
    
    # Build and display the initial multi-turn user prompt (task + tools)
    try:
        multi_turn_prompt = build_multi_turn_user_prompt(task_yaml)
        print("\n🧭 Task Prompt (read carefully):\n")
        print(multi_turn_prompt)
        print("\n" + "="*50)
        if log_file:
            append_to_log(log_file, 0, 'multi_turn_prompt', multi_turn_prompt)
            print("💾 Logged: interaction_0.multi_turn_prompt")
    except Exception as e:
        print(f"⚠️ Could not build multi-turn prompt: {e}")

    interaction_count = 0
    
    while True:
        # Get user input
        user_input = input("\nEnter tool call (or 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break
        
        # Increment interaction count
        interaction_count += 1
        
        # Extract any attempt blocks from input and remove them from the text used for parsing
        attempt_blocks = extract_attempt_blocks(user_input)
        cleaned_input = remove_attempt_blocks(user_input)

        # Log user input immediately
        if log_file:
            append_to_log(log_file, interaction_count, 'agent_input', user_input)
            if attempt_blocks:
                append_to_log(log_file, interaction_count, 'attempt', "\n\n".join(attempt_blocks))
                print(f"💾 Logged: interaction_{interaction_count}.attempt")
            print(f"💾 Logged: interaction_{interaction_count}.agent_input")
        
        # If an attempt/solution block is present, validate it against ground truth and exit
        if attempt_blocks and solution_reference is not None:
            # Join all attempt blocks into one candidate text
            attempt_text = "\n\n".join(attempt_blocks)
            # Extract calls from attempt text
            candidate_calls = extract_all_function_calls(attempt_text)
            # If anything other than function calls is present (non-space chars outside), mark invalid
            stripped_attempt = re.sub(r"\s+", "", attempt_text)
            only_calls_stripped = re.sub(r"\s+", "", "".join(candidate_calls))
            has_extraneous = stripped_attempt != only_calls_stripped

            # Prepare normalized lists
            gt_calls_raw = extract_solution_calls(solution_reference)
            gt_norm = [normalized_string(normalize_tool_call_string(s)) for s in gt_calls_raw]
            cand_norm = [normalized_string(normalize_tool_call_string(s)) for s in candidate_calls]

            is_correct = False
            if not has_extraneous:
                if args.order_matters:
                    is_correct = (gt_norm == cand_norm)
                else:
                    is_correct = (sorted(gt_norm) == sorted(cand_norm))

            if is_correct:
                msg = "Solution is correct."
                print(f"\n📤 Response:")
                print(msg)
                print("\n" + "-"*50)
                if log_file:
                    append_to_log(log_file, interaction_count, 'prompt_simulator', 'Solution attempt received; verification performed')
                    append_to_log(log_file, interaction_count, 'response_simulator', msg)
                    append_to_log(log_file, interaction_count, 'response_to_agent', msg)
                # Exit after success
                return
            else:
                msg = (
                    "Solution is wrong. Remember that order matters and that nothing else should be within the attempt "
                    "besides tool calls with the right parameters, parameter order, and avoiding optional parameters unless necessary."
                )
                print(f"\n📤 Response:")
                print(msg)
                print("\n" + "-"*50)
                if log_file:
                    append_to_log(log_file, interaction_count, 'prompt_simulator', 'Solution attempt received; verification performed')
                    append_to_log(log_file, interaction_count, 'response_simulator', msg)
                    append_to_log(log_file, interaction_count, 'response_to_agent', msg)
                # Continue loop for more input
                continue

        # Parse and extract function call from input; determine tool
        parsed = parse_tool_call_input(cleaned_input, tools_registry, n=1)
        if not parsed:
            # If we received an attempt/solution block, acknowledge it and continue without error
            if attempt_blocks:
                ack = "Solution received."
                print(f"\n📤 Response:")
                print(ack)
                print("\n" + "-"*50)
                if log_file:
                    append_to_log(log_file, interaction_count, 'prompt_simulator', 'Solution received; no execution performed')
                    append_to_log(log_file, interaction_count, 'response_simulator', ack)
                    append_to_log(log_file, interaction_count, 'response_to_agent', ack)
                continue
            
            # No tool call and no attempt – return plain guidance (no JSON)
            guidance = (
                "No tool call or solution block detected. "
                "Provide a valid function call (e.g., ToolName(...)) or wrap a sequence in <ATTEMPT> ... <ATTEMPT>."
            )
            print(f"\n📤 Response:")
            print(guidance)
            print("\n" + "-"*50)
            if log_file:
                append_to_log(log_file, interaction_count, 'prompt_simulator', 'No prompt generated: missing tool call/attempt')
                append_to_log(log_file, interaction_count, 'response_simulator', guidance)
                append_to_log(log_file, interaction_count, 'response_to_agent', guidance)
            continue
        selected_tool_name, tool_call_or_response, is_error = parsed

        if is_error:
            # Tool not found - use the error response directly
            print(f"\n❌ Tool '{selected_tool_name}' not available")
            
            # The error response is already formatted, use it as-is
            response = tool_call_or_response
            response_to_agent = extract_response_to_agent(response)
            
            print(f"\n📤 Response:")
            print(response_to_agent)
            print("\n" + "-"*50)
            
            # Log the error response
            if log_file:
                # Log a placeholder prompt (since we didn't call LLM)
                append_to_log(log_file, interaction_count, 'prompt_simulator', f"Tool not available: {selected_tool_name}")
                print(f"💾 Logged: interaction_{interaction_count}.prompt_simulator")
                
                append_to_log(log_file, interaction_count, 'response_simulator', response)
                print(f"💾 Logged: interaction_{interaction_count}.response_simulator")
                
                append_to_log(log_file, interaction_count, 'response_to_agent', response_to_agent)
                print(f"💾 Logged: interaction_{interaction_count}.response_to_agent")
            
            continue

        tool_call = tool_call_or_response

        print(f"\n🔍 Extracted function call (passed to LLM):")
        print(tool_call)
        print()
            
        try:
            client, tool_config = get_or_create_client(
                selected_tool_name=selected_tool_name,
                client_cache=client_cache,
                tools_registry=tools_registry,
                filtered_metadata_file=filtered_metadata_file,
            )

            prompt_used = build_prompt_for_tool(tool_config, filtered_metadata, tool_call)
            
            # Log prompt immediately
            if log_file:
                append_to_log(log_file, interaction_count, 'prompt_simulator', prompt_used)
                print(f"💾 Logged: interaction_{interaction_count}.prompt_simulator")
            
            response = call_llm_with_prompt(client, prompt_used=prompt_used, tool_call=tool_call)
            
            # Extract response for agent (without explanation)
            response_to_agent = extract_response_to_agent(response)
            
            print(f"\n📤 Response:")
            print(response_to_agent)
            print("\n" + "-"*50)
            
            # Log full response immediately
            if log_file:
                append_to_log(log_file, interaction_count, 'response_simulator', response)
                print(f"💾 Logged: interaction_{interaction_count}.response_simulator")
                
                # Log agent response (without explanation)
                append_to_log(log_file, interaction_count, 'response_to_agent', response_to_agent)
                print(f"💾 Logged: interaction_{interaction_count}.response_to_agent")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()