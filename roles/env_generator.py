"""
EnvironmentGenerator role: loads prompt templates, runs the LLM, returns structured output.

Expected template files dict keys:
- "subfield":  path to template with {field_name}
- "task":      path to template with {field_name}, {subfield_name}
- "tool":      path to template with {field_name}, {subfield_name}, {task_name}
- "sequences": path to template with {Data}, {seqs_per_spec}, {seq_length}
- "metadata":  path to template with {Data} (per-tool "fake world state" for the simulator)
"""

import json
from typing import Any, Callable, Dict, Iterable, List

import yaml

from . import Role
from utils import parse_list, extract_json_objects

class EnvironmentGenerator(Role):
    def __init__(self, template_files: Dict[str, str], runner: Callable[[str], str]):
        """
        template_files: mapping of template keys to YAML files containing a 'template' entry.
        runner: persistent LLM callable taking a prompt string and returning a response string.
        """
        prompts = self._load_prompts(template_files)
        super().__init__(prompts)
        self.runner = runner

    def generate_subfields(self, field_name: str) -> Dict[str, Any]:
        prompt = self.get_prompt("subfield", field_name=field_name)
        response = self.runner(prompt)
        usage = self._get_usage()
        parsed = self._unique(parse_list(response))
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def generate_tasks(self, field_name: str, subfield_name: str) -> Dict[str, Any]:
        prompt = self.get_prompt("task", field_name=field_name, subfield_name=subfield_name)
        response = self.runner(prompt)
        usage = self._get_usage()
        parsed = self._unique(parse_list(response))
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def generate_tools(self, field_name: str, subfield_name: str, task_name: str) -> Dict[str, Any]:
        prompt = self.get_prompt("tool", field_name=field_name, subfield_name=subfield_name, task_name=task_name)
        response = self.runner(prompt)
        usage = self._get_usage()
        parsed = extract_json_objects(response)
        tools_only = [obj for obj in parsed if isinstance(obj, dict) and 'tool_name' in obj]
        seen_names = set()
        unique_tools = []
        for tool in tools_only:
            name = tool.get('tool_name')
            if name and name not in seen_names:
                seen_names.add(name)
                unique_tools.append(tool)
        return {"prompt": prompt, "response": response, "parsed": unique_tools, "usage": usage}

    def generate_sequences(self, tool_data: Dict[str, Any], seqs_per_spec: int, seq_length: int) -> Dict[str, Any]:
        prompt = self.get_prompt(
            "sequences",
            Data=json.dumps(tool_data, indent=2),
            seqs_per_spec=seqs_per_spec,
            seq_length=seq_length,
        )
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else {}
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def generate_metadata(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a per-tool 'fake world state' JSON that the simulator can ground its responses on."""
        prompt = self.get_prompt("metadata", Data=json.dumps(tool_data, indent=2))
        response = self.runner(prompt)
        usage = self._get_usage()
        objs = extract_json_objects(response)
        parsed = objs[0] if objs else {}
        return {"prompt": prompt, "response": response, "parsed": parsed, "usage": usage}

    def run(self, action: str, **kwargs):
        """Dispatch to a specific action."""
        actions = {
            "subfields": self.generate_subfields,
            "tasks": self.generate_tasks,
            "tools": self.generate_tools,
            "sequences": self.generate_sequences,
            "metadata": self.generate_metadata,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    @staticmethod
    def _unique(items: Iterable[str]) -> List[str]:
        seen = set()
        unique_items = []
        for item in items or []:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        return unique_items

    @staticmethod
    def _load_prompts(template_files: Dict[str, str]) -> Dict[str, str]:
        prompts: Dict[str, str] = {}
        for key, path in template_files.items():
            with open(path, "r") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "template" in loaded:
                prompts[key] = loaded["template"]
            else:
                prompts[key] = loaded
        return prompts