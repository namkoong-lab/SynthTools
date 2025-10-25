from model_inference.common_inference import CommonInference
from model_inference.apimodel_inference import APIModelInference


inference_map_groups = {
    APIModelInference: [
        "o1-preview",
        "deepseek-chat",
        "gpt-4o",
        "qwen-max",
        "qwen-plus",
        "gpt-4o-2024-05-13",
        "gpt-4-turbo",
        "gpt-4o-2024-08-06",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-1106-preview-FC",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview-FC",
        "gpt-4-0125-preview",
        "gpt-4-0613-FC",
        "gpt-4-0613",
        "gpt-3.5-turbo-0125-FC",
        "gpt-3.5-turbo-0125",
        "qwen2.5-7b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-72b-instruct",
        "doubao-pro-32k",
        "deepseek-v3",
        "deepseek-r1",
        "o1-mini",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ],
    CommonInference: [
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "qwen2.5-7b-instruct-local",
        "Meta-Llama-3.1-8B-Instruct-local",
        "watt-tool-8B-local",
        "Hammer2.1-7b-local",
        "ToolACE-8B-local",
        "functionary-small-v3.1-local",
        "xLAM-7b-r-local",
        "Llama-3.1-8B-Instruct-local",
        "Qwen2.5-7B-Instruct-local",
        "MiniCPM3-4B-local",
        "Phi-3-mini-128k-instruct-local",
        "Hammer2.1-3b-local",
        "Qwen2.5-3B-Instruct-local",
        "Llama-3.2-3B-Instruct-local",
        "moonshotai/Kimi-K2-Instruct",
    ],
}


inference_map = {model: handler for handler, models in inference_map_groups.items() for model in models}
