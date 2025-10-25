"""
Generate and evaluate judge test scenarios for tool configurations.

Pipeline:
- Load tool configuration files (simulated or acebench tools)
- Generate stress test scenarios using LLMs
- Judge tool responses against expected outcomes
- Save test results and judgments to JSON files

Key options:
- tool_collection_version: Version of tool collection to use (default: "4")
- tool_final: Use final tool collection vs simulated tools (default: True)
- model: LLM model for generation (default: "claude-sonnet-4-20250514")
- model_provider: Provider for LLM calls (default: "anthropic")
- max_tokens: Maximum tokens for LLM responses (default: 10000)

Usage (Anthropic):
  python scripts/judge_test.py \
    --tool_collection_version "4" \
    --tool_final True \
    --model "claude-sonnet-4-20250514" \
    --model_provider "anthropic" \
    --max_tokens 10000

Usage (OpenAI):
  python scripts/judge_test.py \
    --tool_collection_version "4" \
    --tool_final True \
    --model "gpt-4o" \
    --model_provider "openai" \
    --max_tokens 10000
"""

from email import message_from_file
from pathlib import Path
import yaml
import json
import time
from simulator.judge_simulator import JudgeSimulatorClient
from openai import OpenAI
from anthropic import Anthropic

DEFAULT_JUDGE_TEST_CONFIG = {
    "max_tokens": 10000,
    "model": "claude-sonnet-4-20250514",
    "model_provider": "anthropic",
    "tool_final": True,
    "tool_collection_version": "4",
    "temperature": 0.02
}


def get_api_keys(key: str):
    """Get API keys from the config file."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "configs" / "api_keys.json"
    with open(config_path) as f:
        api_keys = json.load(f)
    return api_keys.get(key)


def main(tool_collection_version=None, tool_final=None, model=None, model_provider=None, max_tokens=None):
    if tool_collection_version is None:
        tool_collection_version = DEFAULT_JUDGE_TEST_CONFIG["tool_collection_version"]
    if tool_final is None:
        tool_final = DEFAULT_JUDGE_TEST_CONFIG["tool_final"]
    if model is None:
        model = DEFAULT_JUDGE_TEST_CONFIG["model"]
    if model_provider is None:
        model_provider = DEFAULT_JUDGE_TEST_CONFIG["model_provider"]
    if max_tokens is None:
        max_tokens = DEFAULT_JUDGE_TEST_CONFIG["max_tokens"]

    template_file = Path(__file__).resolve().parent / "prompt_templates" / "generate_testing_functions_judge" / "generate_judge_testing_scenarios.yml"
    with open(template_file) as f:
        template = yaml.safe_load(f)
    template = template["template"]

    base_path = Path(__file__).resolve().parent
    if tool_final:
        simulated_tool_config_files = base_path / "tool_content" / "tool_final" / "tool_json"
    else:
        simulated_tool_config_files = base_path / "tool_content" / "full_tool_specs" / f"tool_collection_json_{tool_collection_version}"
    acebench_tool_config_files = base_path / "evaluation" / "acebench" / "data_en"

    mode = input("Choose mode (1 for Simulated tools, 2 for Acebench tools): ")
    if mode == "1":
        tool_config_files = simulated_tool_config_files
        if not tool_config_files.exists():
            print(f"Tool collection directory not found: {tool_config_files}")
            return
    elif mode == "2":
        tool_config_files = acebench_tool_config_files
        if not tool_config_files.exists():
            print(f"Acebench directory not found: {tool_config_files}")
            return
    else:
        print("Invalid mode selected. Please choose 1 or 2.")
        return

    for tool_config_file in tool_config_files.glob("*.json"):
        try:
            with open(tool_config_file) as f:
                tool_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading tool config file {tool_config_file}: {e}")
            continue

        log_folder = Path(str(tool_config_file).replace('.json', '_logs'))
        log_folder.mkdir(parents=True, exist_ok=True)
        stress_tests_judge_file = log_folder / "stress_tests_judge.json"
        
        if stress_tests_judge_file.exists():
            print(f"Stress tests file already exists at {stress_tests_judge_file}, skipping generation")
            with open(stress_tests_judge_file) as f:
                log_call = json.load(f)
        else:
            print(f"Generating for the tool config:\n{tool_config}")
            tool_config_new = {}
            tool_config_new["tool_details"] = str(tool_config)
            message = template.format(**tool_config_new)
            
            if model_provider == "openai":
                openai_client = OpenAI(api_key=get_api_keys("openai"))
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    max_completion_tokens=max_tokens,
                )
                log_call_str = response.choices[0].message.content
            elif model_provider == "anthropic":
                anthropic_client = Anthropic(api_key=get_api_keys("anthropic"))
                response = anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": message}]
                )
                log_call_str = response.content[0].text
            else:
                print("Invalid model provider selected. Defaulting to OpenAI.")
                openai_client = OpenAI(api_key=get_api_keys("openai"))
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    max_tokens=max_tokens,
                )
                log_call_str = response.choices[0].message.content
            print(f"Raw log call response:\n{log_call_str}")


            print("Trying to json load the log call string")
            try:
                log_call = json.loads(log_call_str)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Attempting to extract JSON from response...")
                import re
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', log_call_str, re.DOTALL)
                if json_match:
                    try:
                        log_call = json.loads(json_match.group(1))
                        print("Successfully extracted JSON from markdown")
                    except json.JSONDecodeError:
                        print(f"Failed to parse extracted JSON, skipping {tool_config_file.name}")
                        continue
                else:
                    print(f"No JSON found in response, skipping {tool_config_file.name}")
                    continue
        
            with open(stress_tests_judge_file, "w") as f:
                json.dump(log_call, f, indent=4) 
            print(f"Log call dumped to:\n{stress_tests_judge_file}")
            
        for test in log_call:
            try:
                print(test)
                print("-" * 50)

                test_file = log_folder / f"{test.replace(' ', '_')}.json"
                judge_file = log_folder / f"{test.replace(' ', '_')}_judge.json"

                if judge_file.exists():
                    print(f"Judge file already exists at {judge_file}, skipping judgment")
                    continue

                chat_log = {
                    "tool_details": tool_config_new["tool_details"],
                    "message": log_call[test]["Tool call message"],
                    "failure_mode": log_call[test]["Failure/Success mode"],
                    "response": log_call[test]["Response"],
                }

                with open(test_file, "w") as f:
                    json.dump(chat_log, f, indent=4)

                judge_client = JudgeSimulatorClient(
                    prompt_template_file=Path(__file__).resolve().parent / "prompt_templates" / "judge_template.yml",
                    chat_log_file=test_file
                )
                judgment, confidence, reasoning = judge_client.judge_tool_response(
                    chat_log_file=test_file,
                    judge_test=True
                )

                with open(judge_file, "w") as f:
                    json.dump({"judgment": judgment.value, "confidence": confidence, "reasoning": reasoning}, f, indent=4)

                print(f"Judgment: {judgment.value}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Reasoning: {reasoning}")

            except Exception as e:
                print(f"Error processing test {test}: {e}")
                continue

    
    print(template)

if __name__ == "__main__":
    main()