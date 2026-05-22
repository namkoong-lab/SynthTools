"""
ToolSimulator agent: loads simulator prompts from hard-coded template paths, runs the LLM,
and performs parameter validation before simulation.

Expected hard-coded template files:
- simulate: prompt_templates/tool_simulator/tool_simulator_simulate_template.yml
- parameter_check: prompt_templates/tool_simulator/tool_simulator_parameter_check_template.yml
"""

import ast
import json
from typing import Callable, Dict, Any, Optional
from pathlib import Path

from . import Role
from utils import extract_json_objects

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt_templates" / "tool_simulator"
SIMULATOR_TEMPLATE_FILE = PROMPT_DIR / "tool_simulator_simulate_template.yml"
PARAMETER_CHECK_TEMPLATE_FILE = PROMPT_DIR / "tool_simulator_parameter_check_template.yml"


def _resolve_ast_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _parse_call(call_str: str) -> Dict[str, Any]:
    """Parse a tool call into {parse_status, parsed_call?|error_message?, raw_call?}."""
    try:
        tree = ast.parse(call_str or "", mode="eval")
    except SyntaxError as e:
        return {"parse_status": "syntax_error",
                "error_message": e.msg or "syntax error",
                "raw_call": call_str}
    body = tree.body
    if not isinstance(body, ast.Call) or not isinstance(body.func, ast.Name):
        return {"parse_status": "syntax_error",
                "error_message": "expected a single ToolName(...) call",
                "raw_call": call_str}
    arguments: Dict[str, Any] = {}
    for kw in body.keywords:
        if kw.arg is None:
            continue
        arguments[kw.arg] = _resolve_ast_value(kw.value)
    return {"parse_status": "ok",
            "parsed_call": {"name": body.func.id, "arguments": arguments}}


class ToolSimulator(Role):
    def __init__(self, runner: Callable[[str], str]):
        prompts = self._load_prompts()
        super().__init__(prompts)
        self.runner = runner

    def simulate(
        self,
        tool_data: Dict[str, Any],
        tool_call_message: str,
        metadata: Any = None,
    ) -> Dict[str, Any]:
        """
        Compute ast_result, ALWAYS pass it to the parameter-check LLM, and
        if it returns PASS, pass the same ast_result to the simulator LLM.
        """
        ast_result = _parse_call(tool_call_message)

        check_prompt, check_resp = self._parameter_check(tool_data, tool_call_message, ast_result)
        check_usage = self._get_usage()
        objs = extract_json_objects(check_resp)
        check_json = objs[0] if objs else None

        passed = False
        if isinstance(check_json, dict):
            passed = (check_json.get("status") == "PASS" or check_json.get("status_code") == 200)
        if not passed:
            passed = ("Status: PASS" in check_resp or "Status Code: 200" in check_resp)

        simulation_json = None
        simulation_prompt = None
        simulation_resp = None
        sim_usage = None
        if passed:
            simulation_prompt, simulation_resp = self._simulate_raw(
                tool_data, tool_call_message=tool_call_message,
                ast_result=ast_result, metadata=metadata,
            )
            sim_usage = self._get_usage()
            sobjs = extract_json_objects(simulation_resp)
            simulation_json = sobjs[0] if sobjs else None

        return {
            "prompt": check_prompt,
            "response": check_resp,
            "parsed": {
                "parameter_check": check_json,
                "passed": passed,
                "simulation": simulation_json,
                "simulation_prompt": simulation_prompt,
                "simulation_response": simulation_resp,
                "ast_result": ast_result,
            },
            "usage": {
                "check": check_usage,
                "simulation": sim_usage,
            },
        }

    def parameter_check(self, tool_data: Dict[str, Any], tool_call_message: str) -> Dict[str, Any]:
        """Run only the parameter-check LLM step. Returns {prompt, response, parsed, usage}."""
        ast_result = _parse_call(tool_call_message)
        prompt, raw = self._parameter_check(tool_data, tool_call_message, ast_result)
        usage = self._get_usage()
        objs = extract_json_objects(raw)
        parsed = objs[0] if objs else None
        passed = False
        if isinstance(parsed, dict):
            passed = (parsed.get("status") == "PASS" or parsed.get("status_code") == 200)
        if not passed:
            passed = ("Status: PASS" in raw or "Status Code: 200" in raw)
        return {"prompt": prompt, "response": raw, "parsed": parsed, "passed": passed,
                "usage": usage, "ast_result": ast_result}

    def simulate_raw(
        self,
        tool_data: Dict[str, Any],
        tool_call_message: str,
        metadata: Any = None,
    ) -> Dict[str, Any]:
        """Run only the simulation LLM step (assumes param check already passed)."""
        ast_result = _parse_call(tool_call_message)
        prompt, raw = self._simulate_raw(tool_data, tool_call_message=tool_call_message,
                                          ast_result=ast_result, metadata=metadata)
        usage = self._get_usage()
        objs = extract_json_objects(raw)
        parsed = objs[0] if objs else None
        return {"prompt": prompt, "response": raw, "parsed": parsed, "usage": usage,
                "ast_result": ast_result}

    def run(self, action: str, **kwargs):
        actions = {
            "simulate": self.simulate,
            "parameter_check": self.parameter_check,
            "simulate_raw": self.simulate_raw,
        }
        if action not in actions:
            raise ValueError(f"Unsupported action '{action}'. Valid: {list(actions)}")
        return actions[action](**kwargs)

    @classmethod
    def _load_prompts(cls) -> Dict[str, str]:
        return {
            "simulate": cls._load_single_template(SIMULATOR_TEMPLATE_FILE),
            "parameter_check": cls._load_single_template(PARAMETER_CHECK_TEMPLATE_FILE),
        }

    def _simulate_raw(
        self,
        tool_data: Dict[str, Any],
        tool_call_message: str,
        ast_result: Dict[str, Any],
        metadata: Any = None,
    ) -> tuple[str, str]:
        if not tool_data.get("output_details"):
            raise ValueError("tool_data missing required 'output_details' for simulation prompt")
        prompt = self.get_prompt(
            "simulate",
            tool_name=tool_data.get("tool_name", ""),
            tool_description=self._fmt(tool_data.get("tool_description", "")),
            parameters=self._fmt(tool_data.get("parameters", {})),
            error_messages=self._fmt(tool_data.get("error_messages", [])),
            usage=self._fmt(tool_data.get("usage", "")),
            initial_config=self._fmt(tool_data.get("initial_config", {})),
            tool_call=self._fmt(tool_data.get("tool_call", {}) or tool_call_message),
            ast_result=json.dumps(ast_result, indent=2, ensure_ascii=False, default=str),
            output_details=self._fmt(tool_data.get("output_details", {})),
            metadata=self._fmt(metadata or {}),
        )
        return prompt, self.runner(prompt)

    def _parameter_check(
        self,
        tool_data: Dict[str, Any],
        tool_call_message: str,
        ast_result: Dict[str, Any],
    ) -> tuple[str, str]:
        prompt = self.get_prompt(
            "parameter_check",
            tool_name=tool_data.get("tool_name", ""),
            tool_description=self._fmt(tool_data.get("tool_description", "")),
            parameters=self._fmt(tool_data.get("parameters", {})),
            error_messages=self._fmt(tool_data.get("error_messages", [])),
            usage=self._fmt(tool_data.get("usage", "")),
            ast_result=json.dumps(ast_result, indent=2, ensure_ascii=False, default=str),
        )
        return prompt, self.runner(prompt)
