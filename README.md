# Scaling Tools and Tasks and Show Values in Training on Them

## Schema
```
SynthTools/
|-- configs/                      # API keys and simulation parameters
|-- evaluation/                   # Evaluation scripts
|-- prompt_templates/             # Prompt templates used for tool generation and simulation
|-- scripts/                      # Scripts for running the whole pipeline
|-- simulator/                    # Core simulation for tools and judge
|-- tool_content/                 # Main curated dataset
|-- utils/                        # Shared utility functions
|
|-- .gitignore                    # Ignore patterns
|-- README.md                     # Project documentation
|-- main_create_tools.py          # Main script to generate synthetic tools
|-- main_dedup.py                 # Main script for deduplicating
|-- pyproject.toml                # Project metadata and build configuration
|-- requirements.txt              # Python dependencies
|-- setup.sh                      # Environment setup script
```          

## Setup

Run `bash setup.sh` to setup venv with uv.

Run `source .venv/bin/activate` to activate the environment.

## Configs

Add your api keys to `/configs/api_keys.json`

## Start

Run `python simulate_tool.py`

