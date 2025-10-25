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
```
tool_content/
|-- task_curated/        # Manually curated tasks 
|-- task_meta/           # Metadata for task simulation
|-- task_simulation/     # Simulation outputs from tool–agent interactions
|-- task_yml/            # YAML definitions of tasks
|-- tool_emb/            # Tool embeddings or vector representations
|-- tool_json/           # Tool specifications in JSON format 
|-- tool_meta/           # Metadata for generated tools 
|-- tool_yaml/           # Tool definitions or configurations in YAML format
|-- fields.txt           # Lists or descriptions of key data fields used across the dataset
```

## Tool Generation Pipeline
### 1. Generate Tools 
Run `python main_create_tools.py --config configs/generate_tool_config.yml`
### 2. Run tool deduplication 
Run `python main_dedup.py --config configs/deduplicate_tools_config.yml`

More to come!

