# Scaling Tools and Tasks and Show Values in Training on Them

## Shema
```
SynthTools/
|-- configs/ # API keys and simulation parameters
|-- evaluation/ # Evaluation scripts
|-- prompt_templates/ # Prompt templates used for tool generation and simulation
|-- scripts/ # Scripts for running the whole pipeline
|-- simulator/ # Core simulation for tools and judge
|-- tool_content/ # Main curated dataset
|-- utils/
|
|-- .gitignore
|-- README.md
|-- main_create_tools.py # Main script to generate synthetic tools
|-- main_dedup.py # Main script to generate for deduplicating
|-- pyproject.toml # Project metadata and build configuration
|-- requirements.txt
|-- setup.sh
```
       

## Setup

Run `bash setup.sh` to setup venv with uv.

Run `source .venv/bin/activate` to activate the environment.

Add your api keys to `/configs/api_keys.json`

## Tool Dataset


## Tool Generation Pipeline
### 1. Generate Tools 
Run `python main_create_tools.py --config configs/generate_tool_config.yml`
### 2. Run tool deduplication 
Run `python main_dedup.py --config configs/deduplicate_tools_config.yml`

