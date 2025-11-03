# SynthTools: A Framework for Scaling Synthetic Tools for Agent Development

SynthTools is a scalable and efficient framework for generating synthetic tools and tasks for agent development. It was introduced in the paper "SynthTools: A Framework for Scaling Synthetic Tools for Agent Development."

The framework addresses the limitations of using real world APIs, which are constrained by access keys, rate limits, and instability, by creating synthetic tool ecosystems that are controllable, reproducible, and scalable. SynthTools consists of six main components:

**1. Tool Generation**: Automatically creates a diverse set of synthetic tools with varied interfaces and functionalities across many domains. This component leverages the generative capacity of large language models to produce rich, realistic, and nontrivial tool behaviors, achieving more than double the diversity and scale of existing benchmarks.

**2. Tool Simulation**: Emulates realistic tool behaviors and interactions, ensuring generated tools respond consistently and flexibly to varied input and output structures. This supports broad experimental setups and allows researchers to generate diverse task environments.

**3. Tool Audit**: Ensures consistency, accuracy, and reliability of generated tools by auditing their responses. The LLM-based judge module achieved 99 percent accuracy and zero false positives when diagnosing simulator performance, confirming the robustness of the audit process.

**4. Task Generation**: Constructs complex, multi-turn tasks from the generated tools. Tasks are designed through LLM prompting and require structured decision making and the coordinated use of multiple composable tools. Metadata generated in parallel ensures consistency between task requirements and tool outputs.

**5. Task Simulation**: Uses the generated metadata to produce grounded and coherent tool responses during task execution, allowing for reproducible simulation of agent performance in synthetic environments.

**6. Curated Dataset**: SynthTools readty-to-use datasete contains approximately 6000 reliable synthetic tools spanning diverse domains. These tools form the basis for constructing realistic and challenging tasks. Human inspection confirms that the tasks are reasonable and solvable, while state of the art models still struggle to complete them successfully. This demonstrates that the synthetic tasks are valuable for advancing agent reasoning and tool use capabilities.

## Schema

The repo is organized as follows:

```
SynthTools/
|-- configs/                # API keys and simulation parameters
|-- evaluation/             # Evaluation scripts
|-- prompt_templates/       # Prompt templates used for tool generation and simulation
|-- scripts/                # Scripts for running the whole pipeline
|-- simulator/              # Core simulation for tools and judge
|-- tool_content/           # Main curated dataset
|-- utils/
|
|-- .gitignore
|-- README.md
|-- main_create_tools.py    # Main script to generate synthetic tools
|-- main_dedup.py           # Main script for deduplicating tools
|-- pyproject.toml          # Project metadata and build configuration
|-- requirements.txt
|-- setup.sh
|-- main_create_tasks.py    # Main script to generate synthetic tasks
|-- main_dedup_tools.py     # Main script for deduplicating tools
|-- main_multi_turn.py      # Main script for multi turn simulations
|-- main_evaluate_tools.py  # Main script for tool evaluation
```

## Setup

Run `bash setup.sh` to setup venv with uv.

Run `source .venv/bin/activate` to activate the environment.

Add your api keys to `/configs/api_keys.json`

## Tool Generation Pipeline

### 1. Generate Tools 
Run `python main_create_tools.py --config configs/generate_tool_config.yml`

### 2. Tool deduplication 

Run `python main_evaluate_tools.py --config configs/evaluate_tools_config.yml`

### 3. Tool evaluation

Run `python main_create_tasks.py --config configs/generate_tasks.yaml`

### 4. Task evaluation

Run `python main_create_tasks.py --config configs/generate_tasks.yaml`

### 5. Task simulation

Run `python main_multi_turn.py --config configs/multi_turn_config.yml`

## Tool Dataset
SynthTools curated dataset approximately contains 6000 reliable synthetic tools spanning diverse domains.

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
|-- tool_eval_logs/      # Tool evaluation calls 
|-- fields.txt           # Lists or descriptions of key data fields used across the dataset
|-- tool_list.json       # Jsonl list with all tool database information 
```
 
