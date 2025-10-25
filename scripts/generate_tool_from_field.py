"""
Generate Tool specification from Field description using LLMs.

Pipeline:
- Field -> Subfield
- Subfield -> Task
- Task -> Tool Specification
- Tool Specification -> Metadata for tool return values

Key options:
- --api-keys: Path to API keys JSON. The file is copied to
  scaling_tools/configs/api_keys.json so existing clients keep working.
  Default: configs/api_keys.json
- --verbose: If set, prints detailed progress. If not set, only the tqdm
  progress bar is shown.

Usage (Anthropic - Single Field):
  python scripts/generate_tool_from_field.py \
    --field_name "Software Engineering" \
    --model_provider anthropic \
    --model claude-3-7-sonnet-20250219 \
    --max_tokens 1500 \
    --temperature 0.1 \
    --max_subfields_per_field 1 \
    --max_tasks_per_subfield 1 \
    --api-keys configs/api_keys.json \
    --output_dir tool_content/full_tool_specs

Usage (OpenAI - Multiple Fields):
  python scripts/generate_tool_from_field.py \
    -f "Software Engineering" "Sports Betting" "Data Science" \
    --model_provider openai \
    --model gpt-4o \
    --max_tokens 1500 \
    --temperature 0.1 \
    --max_subfields_per_field 2 \
    --max_tasks_per_subfield 3 \
    --api-keys configs/api_keys.json \
    --output_dir tool_content/full_tool_specs
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
import utils.misc_utils as mscu
import yaml
from tqdm import tqdm

from utils.client_utils import AnthropicClient, OpenAIClient
from utils.misc_utils import LiteralString, make_directory, save_yaml


PROJ_ROOT = Path(__file__).parent.parent


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate tool parameters.")

    parser.add_argument(
        "-f",
        "--field_name",
        type=str,
        nargs="+",
        help="Name(s) of the field(s) - can specify multiple fields like 'Software Engineering' 'Sports Betting'",
        required=True,
    )
    parser.add_argument(
        "--subfield_gen_template_file",
        type=str,
        help="Path to the subfield generation template file",
        default="prompt_templates/generate_subfields/generate_subfield_template.yml",
    )
    parser.add_argument(
        "--task_gen_template_file",
        type=str,
        help="Path to the task generation template file",
        default="prompt_templates/generate_tasks/generate_tasks_template.yml",
    )
    parser.add_argument(
        "--tool_gen_template_file",
        type=str,
        help="Path to the tool generation template file",
        default="prompt_templates/generate_tools/generate_tools_template.yml",
    )
    parser.add_argument(
        "--metadata_gen_template_file",
        type=str,
        help="Path to the metadata generation template file",
        default="prompt_templates/generate_metadata/generate_metadata_template.yml",
    )

    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514", help="Model name"
    )
    parser.add_argument(
        "--model_provider", type=str, default="anthropic", help="Model provider: anthropic or openai"
    )
    parser.add_argument("--max_tokens", type=int, default=12000, help="Maximum tokens")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature (0.0 for deterministic output)",
    )

    parser.add_argument(
        "--max_subfields_per_field",
        type=int,
        default=1,
        help="Maximum subfields to process per field",
    )
    parser.add_argument(
        "--max_tasks_per_subfield",
        type=int,
        default=1,
        help="Maximum tasks to process per subfield",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the output files",
        default="tool_content/full_tool_specs/",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="Name of the output file",
        default=None,
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        default="configs/api_keys.json",
        help="Path to API keys JSON (will be copied to scaling_tools/configs/api_keys.json for clients)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (prints detailed progress)",
    )

    parser.add_argument(
        "--block_yaml",
        action="store_true",
        default=True,
        help=(
            "If True (default), write block scalars with '|' via default dumper; if False, "
            "write chomped block scalars '|-' so no separate conversion is needed."
        ),
    )

    return parser.parse_args()


class _LiteralDumper(yaml.SafeDumper):
    pass


def _literal_string_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data), style='|')


_LiteralDumper.add_representer(LiteralString, _literal_string_representer)


def load_api_keys_from_path_copy_if_needed(api_keys_path_arg: Path) -> None:
    expected_path = (PROJ_ROOT / "configs" / "api_keys.json").resolve()
    if not api_keys_path_arg:
        return
    try:
        src = Path(api_keys_path_arg)
        if not src.is_absolute():
            src = (PROJ_ROOT / src).resolve()
        if src != expected_path and src.exists():
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(src), str(expected_path))
    except Exception as e:
        print(f"Warning: Failed to sync API keys from {api_keys_path_arg} to {expected_path}: {e}")


def main():
    args = parse_arguments()

    def vprint(*a, **k):
        if args.verbose:
            print(*a, **k)

    vprint(f"Script arguments: {json.dumps(vars(args), indent=2)}")

    def _write_yaml(data, path):
        mscu.setup_yaml()
        text = yaml.dump(
            data,
            Dumper=_LiteralDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=4096,
        )
        pattern = rf"^(?P<indent>\s*)(?P<key>{key_alt}):\s*\|\n"
        text = re.sub(
            pattern,
            lambda m: f"{m.group('indent')}{m.group('key')}: |-\n",
            text,
            flags=re.MULTILINE,
        )
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    load_api_keys_from_path_copy_if_needed(Path(args.api_keys))

    provider = args.model_provider.lower()
    if provider == "anthropic":
        client = AnthropicClient(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    elif provider == "openai":
        if not args.model or "claude" in args.model.lower():
            args.model = "gpt-4o"
        client = OpenAIClient(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        raise ValueError(f"Unsupported model provider: {args.model_provider}")

    with open(PROJ_ROOT / args.subfield_gen_template_file, "r") as f:
        subfield_template_yaml = yaml.safe_load(f)
    with open(PROJ_ROOT / args.task_gen_template_file, "r") as f:
        task_template_yaml = yaml.safe_load(f)
    with open(PROJ_ROOT / args.tool_gen_template_file, "r") as f:
        tool_template_yaml = yaml.safe_load(f)
    with open(PROJ_ROOT / args.metadata_gen_template_file, "r") as f:
        metadata_template_yaml = yaml.safe_load(f)

    vprint("Loaded all templates.")

    for field_name in args.field_name:
        vprint(f"\n{'='*60}")
        vprint(f"Processing field: {field_name}")
        vprint(f"{'='*60}")

        subfield_prompt = subfield_template_yaml["template"].format(
            field_name=field_name
        )
        vprint(f"Subfield Generation Prompt:\n{subfield_prompt}")

        subfields = client.message(subfield_prompt, text_parser=mscu.parse_list)
        vprint(f"Subfield Generation Response:\n{subfields}")

        seen_subfields = set()
        unique_subfields = []
        for s in subfields:
            if s not in seen_subfields:
                seen_subfields.add(s)
                unique_subfields.append(s)

        if not subfields or type(subfields) != list or len(subfields) == 0:
            vprint(f"Warning: No subfields generated for field '{field_name}'. Skipping.")
            continue

        for subfield in unique_subfields[:args.max_subfields_per_field]:
            task_prompt = task_template_yaml["template"].format(
                field_name=field_name, subfield_name=subfield
            )
            vprint(f"Task Generation Prompt for Subfield '{subfield}':\n{task_prompt}")

            tasks = client.message(task_prompt, text_parser=mscu.parse_list)
            if not tasks or type(tasks) != list or len(tasks) == 0:
                vprint(f"Warning: No tasks generated for subfield '{subfield}' in field '{field_name}'. Skipping.")
                continue

            seen_tasks = set()
            unique_tasks = []
            for t in tasks:
                if t not in seen_tasks:
                    seen_tasks.add(t)
                    unique_tasks.append(t)

            vprint(f"Extracted Tasks for Subfield '{subfield}':\n{unique_tasks}")

            for task in unique_tasks[:args.max_tasks_per_subfield]:
                tool_prompt = tool_template_yaml["template"].format(
                    field_name=field_name, subfield_name=subfield, task_name=task
                )
                vprint(f"Tool Generation Prompt for Task '{task}':\n{tool_prompt}")

                tool_description = client.message(tool_prompt)
                if not tool_description:
                    vprint(f"Warning: No tool description generated for task '{task}' in subfield '{subfield}' of field '{field_name}'. Skipping.")
                    continue

                vprint(f"Extracted Tool Description for Task '{task}':\n{tool_description}")

                output_data = {
                    "field_name": field_name,
                    "subfield": subfield,
                    "task": task,
                    "tool_description": mscu.LiteralString(tool_description),
                    
                }
                vprint(f"Output Data (field='{field_name}', subfield='{subfield}', task='{task}'):\n{json.dumps(output_data, indent=2)}")

                output_dir = PROJ_ROOT / args.output_dir
                make_directory(output_dir)
                if args.output_file_name is None:
                    max_index = 0
                    if os.path.isdir(output_dir):
                        for existing_name in os.listdir(output_dir):
                            if existing_name.endswith((".yml", ".yaml")):
                                match = re.search(r"_tool_spec_(\d+)\.ya?ml$", existing_name)
                                if match:
                                    number = int(match.group(1))
                                    if number > max_index:
                                        max_index = number
                    next_index = max_index + 1
                    field_slug = field_name.replace(' ', '_').lower()
                    output_file = f"{field_slug}_tool_spec_{next_index}.yml"

                    _write_yaml(output_data, output_dir / output_file)
                    vprint(
                        f"Generated tool specification saved to {output_dir / output_file} "
                        f"(field='{field_name}', subfield='{subfield}', task='{task}')"
                    )
                else:
                    field_slug = field_name.replace(' ', '_').lower()
                    base_name = Path(args.output_file_name).stem
                    extension = Path(args.output_file_name).suffix
                    output_file = output_dir / f"{field_slug}_{base_name}{extension}"
                    _write_yaml(output_data, output_file)

                    vprint(f"Generated tool specification saved to {output_file}")

    vprint(f"\n{'='*60}")
    vprint(f"Completed processing {len(args.field_name)} field(s): {', '.join(args.field_name)}")
    vprint(f"{'='*60}")


if __name__ == "__main__":
    main()
