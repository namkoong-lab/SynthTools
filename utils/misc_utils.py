import json
import os
import re

import yaml


def make_directory(path):
    """Create directory if it doesn't exist."""
    if "." in os.path.basename(path):
        dir_path = os.path.dirname(path)
    else:
        dir_path = path

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class LiteralString(str):
    pass


def literal_presenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def setup_yaml():
    """Setup YAML with custom representers"""
    yaml.add_representer(LiteralString, literal_presenter)


def save_yaml(data, filename):
    """Save data to YAML file with custom formatting"""
    setup_yaml()
    with open(filename, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def load_yaml(filename):
    """Load YAML file"""
    with open(filename, "r") as file:
        return yaml.safe_load(file)


def parse_list(text):
    """Parse a list from text, handling various formats."""

    list_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if list_match:
        list_str = list_match.group(1)
        try:
            return json.loads(list_str)
        except json.JSONDecodeError:
            pass

    try:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, list):
            return parsed
    except yaml.YAMLError:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse list from text. {text}")
