#!/usr/bin/env python3
"""
Judge Simulator for Tool Response Quality Evaluation

This script evaluates the quality of backend tool simulator responses by using a frontier LLM
to judge whether simulated tool responses are realistic and appropriate. It can detect cases where:
- A tool call should succeed but the simulator returned an error
- A tool call should fail but the simulator returned success  
- The response content doesn't match expected behavior given the tool parameters and states

"""

import argparse
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import glob
import yaml
from anthropic import Anthropic
import re

# Configuration constants
DEFAULT_JUDGE_CONFIG = {
    "max_tokens": 2048,  # Higher for detailed reasoning
    "temperature": 0.02,
    "max_retries": 3,
    "base_delay": 2.0,
    "backoff_factor": 2.0,
    "model": "claude-sonnet-4-20250514",
    "tool_final": True,
    "tool_collection_version": "4",
    "log_file_pattern": "Tool_call_*.json"
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JudgmentResult(Enum):
    """Possible judgment outcomes"""
    CORRECT = "correct"
    SHOULD_SUCCESS_GOT_ERROR = "should_success_got_error" 
    SHOULD_ERROR_GOT_SUCCESS = "should_error_got_success"
    INAPPROPRIATE_RESPONSE = "inappropriate_response"
    UNREALISTIC_BEHAVIOR = "unrealistic_behavior"
    UNKNOWN = "unknown"


@dataclass
class ToolCallEvaluation:
    """Evaluation result for a single tool call"""
    log_file: str
    user_message: str
    tool_call : str
    response: str
    judgment: JudgmentResult
    confidence_score: float
    reasoning: str
    timestamp: str
    tool_config: Dict[str, Any]


@dataclass
class SimulatorEvaluation:
    """Evaluation result for the simulator"""
    total_interactions: int
    evaluations: List[ToolCallEvaluation]
    overall_score: float
    summary: str
    evaluation_timestamp: str


class JudgeSimulatorClient:
    """Client for judging simulator quality - borrows patterns from simulate_tool.py"""
    
    def __init__(self, **kwargs):
        # Set up basic config using DEFAULT_JUDGE_CONFIG
        self.max_tokens = kwargs.get("max_tokens", DEFAULT_JUDGE_CONFIG["max_tokens"])
        self.temperature = kwargs.get("temperature", DEFAULT_JUDGE_CONFIG["temperature"])
        self.max_retries = kwargs.get("max_retries", DEFAULT_JUDGE_CONFIG["max_retries"])
        self.base_delay = kwargs.get("base_delay", DEFAULT_JUDGE_CONFIG["base_delay"])
        self.backoff_factor = kwargs.get("backoff_factor", DEFAULT_JUDGE_CONFIG["backoff_factor"])
        self.chat_log = kwargs.get("chat_log", {})

        # Set up template (borrowed from ToolClient)
        script_dir = Path(__file__).resolve().parent
        self.prompt_template_file = kwargs.get(
            "prompt_template_file",
            f"{script_dir}/prompt_templates/judge_template.yml"
        )
        self.load_prompt_template()

        # Set up Anthropic client (borrowed from AnthropicToolClient)
        self.model = kwargs.get("model", DEFAULT_JUDGE_CONFIG["model"])
        self.client = self.get_anthropic_client()

    def load_chat_log(self, chat_log_file: Path):
        with open(chat_log_file) as f:
            self.chat_log = json.load(f)
    
    def load_prompt_template(self):
        """Load yml template file (borrowed from ToolClient)"""
        with open(self.prompt_template_file) as f:
            self.prompt_yaml = yaml.safe_load(f)
        
        self.prompt_schema = self.prompt_yaml.get("schema", None)
        if not self.prompt_schema:
            raise ValueError(f"Missing prompt schema in {self.prompt_template_file}")
        
        self.prompt_template = self.prompt_yaml.get("template", None)
        if not self.prompt_template:
            raise ValueError(f"Missing prompt template in {self.prompt_template_file}")
    
    def get_api_keys(self, key: str):
        """Get API keys from config file (borrowed from AnthropicToolClient)"""
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / "configs" / "api_keys.json"
        with open(config_path) as f:
            api_keys = json.load(f)
        return api_keys.get(key)
    
    def get_anthropic_client(self):
        """Get Anthropic client (borrowed from AnthropicToolClient)"""
        api_key = self.get_api_keys("anthropic")
        return Anthropic(api_key=api_key)
    
    def extract_text(self, response):
        """Extract text from Anthropic response (borrowed from AnthropicToolClient)"""
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
    
    def message(self, content):
        """Send message with retry logic (borrowed from ModelClient pattern)"""
        for i in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": content}],
                )
                return self.extract_text(response)
            except Exception as e:
                print(f"Attempt {i + 1} failed: {e}")
                import time
                time.sleep(self.base_delay * (self.backoff_factor**i))
        raise RuntimeError("All attempts failed.")
    
    def judge_tool_response(
        self,
        chat_log_file: Path,
        judge_test = False
    ) -> Tuple[JudgmentResult, float, str]:
        """
        Judge whether a backend simulator response is correct
        
        We're evaluating: Did the backend simulator return the correct response
        given the tool call parameters, tool capabilities, and current state?
        
        Returns:
            Tuple of (judgment, confidence_score, reasoning)
        """
        self.load_chat_log(chat_log_file)
        chat_log = self.chat_log
        if judge_test:
            tool_details = chat_log.get('tool_details', {})
        else:  
            tool_config = chat_log.get('chat_configs', {}).get('tool_configs', {})
            tool_details = str(tool_config)
        message = chat_log.get('message', '')
        response = chat_log.get('response', '')
        meta_data = "Example Tool Call: " + str(chat_log.get('Tool parameters', {})) + "\n\nExample Return Data: " + str(chat_log.get('Ground truth return data', {}))
        
        # Prepare data for template
        evaluation_data = {
            'tool_details': tool_details,
            'message': message,
            'response': response,
            'meta_data': meta_data
        }
        
        # Format the prompt using the template
        formatted_prompt = self.prompt_template.format(**evaluation_data)

        print(f"Formatted prompt:\n{formatted_prompt}")
        
        try:
            # Send the formatted prompt directly
            response = self.message(formatted_prompt)
            return self._parse_judgment_response(response)
        except Exception as e:
            logger.error(f"Error getting judgment: {e}")
            return JudgmentResult.UNKNOWN, 0.0, f"Error occurred: {str(e)}"
    
    def _parse_judgment_response(self, response: str) -> Tuple[JudgmentResult, float, str]:
        """Parse the LLM judgment response into structured data"""
        try:
            # Try to extract JSON from the response
            import re

            # First try to extract from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    # Fall through to other methods
                    pass
                else:
                    return self._extract_judgment_data(result)

            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
            else:
                # Fallback parsing - try the whole response
                result = json.loads(response)

            return self._extract_judgment_data(result)

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response content: {response}")
            return JudgmentResult.UNKNOWN, 0.0, f"Failed to parse JSON: {str(e)}"
        except Exception as e:
            logger.error(f"Failed to parse judgment response: {e}")
            logger.error(f"Response content: {response}")
            return JudgmentResult.UNKNOWN, 0.0, f"Failed to parse response: {str(e)}"

    def _extract_judgment_data(self, result: dict) -> Tuple[JudgmentResult, float, str]:
        """Extract judgment data from parsed JSON"""
        judgment_str = result.get("judgment", "unknown").lower()
        confidence = float(result.get("confidence", 0.0))
        reasoning = result.get("reasoning", "No reasoning provided")

        # Map judgment string to enum
        judgment_map = {
            "correct": JudgmentResult.CORRECT,
            "should_success_got_error": JudgmentResult.SHOULD_SUCCESS_GOT_ERROR,
            "should_error_got_success": JudgmentResult.SHOULD_ERROR_GOT_SUCCESS,
            "inappropriate_response": JudgmentResult.INAPPROPRIATE_RESPONSE,
            "unrealistic_behavior": JudgmentResult.UNREALISTIC_BEHAVIOR,
            "unknown": JudgmentResult.UNKNOWN
        }

        judgment = judgment_map.get(judgment_str, JudgmentResult.UNKNOWN)
        return judgment, confidence, reasoning

def judge_single_file(chat_log_file):
    """Judge a single chat log file and save the results."""
    judge = JudgeSimulatorClient()
    judgment, confidence, reasoning = judge.judge_tool_response(chat_log_file=chat_log_file)
    print(f"Judgment for {chat_log_file.name}: {judgment}")
    print(f"Confidence: {confidence}")
    print(f"Reasoning: {reasoning}")
    
    # Log into a file with the same name as the chat log file, but with the extension .judge.json
    judge_log_file = chat_log_file.with_suffix(".judge.json")
    with open(judge_log_file, "w") as f:
        json.dump({"judgment": judgment.value, "confidence": confidence, "reasoning": reasoning}, f, indent=2)
    
    return judgment, confidence, reasoning

def process_log_directory(log_dir, file_pattern=None, meta_data=False):
    """Process all JSON files in a specific log directory."""
    if file_pattern is None:
        file_pattern = DEFAULT_JUDGE_CONFIG["log_file_pattern"]

    log_dir = Path(log_dir)
    if not log_dir.exists():
        logger.error(f"Log directory does not exist: {log_dir}")
        return

    results = {}
    for chat_log_file in log_dir.glob(file_pattern):
        if meta_data:
            # # Process the ones that has the pattern meta_data_tool_call.json
            if not chat_log_file.name == "meta_data_tool_call_fixed.json":
                continue
        else:
            # # Process the ones that has the pattern Tool_call_(number).json
            if not re.match(r"Tool_call_\d+\.json", chat_log_file.name):
                continue
        try:
            print(f"Reached here, processing {chat_log_file}")
            judgment, confidence, reasoning = judge_single_file(chat_log_file)
            results[chat_log_file.name] = {
                "judgment": judgment.value,
                "confidence": confidence,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error(f"Error processing {chat_log_file}: {e}")
            results[chat_log_file.name] = {
                "judgment": "error",
                "confidence": 0.0,
                "reasoning": f"Error processing file: {str(e)}"
            }

    return results

def process_parent_directory(parent_dir, meta_data=False):
    """Process all subdirectories containing that ends with _logs files."""
    parent_dir = Path(parent_dir)
    if not parent_dir.exists():
        logger.error(f"Parent directory does not exist: {parent_dir}")
        return
    
    all_results = {}
    for subdir in parent_dir.iterdir():
        if subdir.is_dir() and subdir.name.endswith("_logs"):
            logger.info(f"Processing directory: {subdir}")
            if meta_data:
                file_pattern = "meta_data_tool_call_fixed.json"
            else:
                file_pattern = "Tool_call_*.json"
            results = process_log_directory(subdir, file_pattern=file_pattern, meta_data=meta_data)
            if results:
                all_results[subdir.name] = results
    
    return all_results

def main():
    # parser = argparse.ArgumentParser(description="Judge tool simulator response quality")
    # parser.add_argument("--log_dir", type=Path, default="./logs", help="Directory containing interaction logs")
    # parser.add_argument("--output_file", type=Path, default="judge_results.json", help="Output evaluation JSON file")
    # parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to use for judging")
    # parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    # parser.add_argument("--tool_config", type=Path, default="./tool_config.json", help="Tool config file")
    # parser.add_argument("--chat_log", type=Path, default="./chat_log.json", help="Chat log file")
    # args = parser.parse_args()
    
    # if args.verbose:
    #     logging.getLogger().setLevel(logging.DEBUG)
    # Input whether we want acebench or simulated tools
    mode = input("Choose mode (1 for Simulated tools, 2 for Acebench tools, 3 for Simulated tools with meta data): ")

    base_path = Path(__file__).resolve().parent
    meta_data = False
    if mode == "1":
        tool_final = DEFAULT_JUDGE_CONFIG["tool_final"]
        if tool_final:
            chat_log_dir = base_path / "tool_content" / "tool_final" / "tool_json"
        else:
            tool_collection_version = DEFAULT_JUDGE_CONFIG["tool_collection_version"]
            chat_log_dir = base_path / "tool_content" / "full_tool_specs" / f"tool_collection_json_{tool_collection_version}"
    elif mode == "2":
        chat_log_dir = base_path / "evaluation" / "acebench" / "data_en"
    elif mode == "3":
        chat_log_dir = base_path / "tool_content" / "tool_final" / "tool_json"
        meta_data = True
    else:
        print("Invalid mode selected. Please choose 1, 2, or 3.")
        return

    if not chat_log_dir.exists():
        logger.error(f"Directory does not exist: {chat_log_dir}")
        return

    process_parent_directory(chat_log_dir, meta_data=meta_data)
    
    # To process a parent directory with multiple log directories:
    # parent_dir = Path(__file__).resolve().parent / "evaluation" / "acebench" / "data_en"
    # all_results = process_parent_directory(parent_dir)
    # with open("all_judgments.json", "w") as f:
    #     json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()