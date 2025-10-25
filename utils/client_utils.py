import json
import time
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI


class ModelClient:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.temperature = kwargs.get("temperature", 0.02)

        self.max_retries = kwargs.get("max_retries", 3)
        self.base_delay = kwargs.get("base_delay", 2.0)
        self.backoff_factor = kwargs.get("backoff_factor", 2.0)
        self.model_name = None

    def _message(self, content):
        raise NotImplementedError("Subclasses must implement the message method.")

    def message(self, content, **kwargs):
        for i in range(self.max_retries):
            try:
                response = self._message(content, **kwargs)
                return response
            except Exception as e:
                print(f"Attempt {i + 1} failed: {e}")
                time.sleep(self.base_delay * (self.backoff_factor**i))
        raise RuntimeError("All attempts failed.")

    def get_api_keys(self, key: str):
        """Get API keys from the config file."""
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / "../configs" / "api_keys.json"
        with open(config_path) as f:
            api_keys = json.load(f)

        try:
            return api_keys.get(key)
        except KeyError:
            print(
                f"API key '{key}' not found."
                f" Please add it to the config file at {config_path}"
            )
            return None


class AnthropicClient(ModelClient):
    ANTHROPIC_MODELS = [
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
    ]

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.client = self.get_anthropic_client()
        if model:
            assert model in self.ANTHROPIC_MODELS
            self.model = model
        else:
            self.model = "claude-sonnet-4-20250514"

        self.prompt = kwargs.get("prompt", None)

    def get_api_keys(self, key: str):
        """Get API keys from the config file."""
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / "../configs" / "api_keys.json"
        with open(config_path) as f:
            api_keys = json.load(f)

        try:
            return api_keys.get(key)
        except KeyError:
            print(
                f"API key '{key}' not found."
                f" Please add it to the config file at {config_path}"
            )
            return None

    def get_anthropic_client(self) -> Anthropic:
        """Get an instance of the Anthropic client."""
        api_key = self.get_api_keys("anthropic")
        return Anthropic(api_key=api_key)

    def get_model_name(self):
        return self.model

    def _message(self, content, text_only=True, text_parser=None):
        msg_content = (self.prompt + "\n" + content) if self.prompt else content
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": msg_content}],
        )
        if text_only:
            response = self.extract_text(response)

            if text_parser:
                response = text_parser(response)

        return response

    def extract_text(self, response) -> str:
        """Extract text from the Anthropic API response."""
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


class OpenAIClient(ModelClient):
    OPENAI_MODELS = [
        "gpt-5-2025-08-07",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
    ]

    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.client = self.get_openai_client()
        if model:
            assert model in self.OPENAI_MODELS
            self.model = model
        else:
            self.model = self.OPENAI_MODELS[0]
            print(f"No model specified. Using default model: {self.model}")

        self.prompt = kwargs.get("prompt", None)

    def get_openai_client(self):
        """Get an instance of the OpenAI client."""
        api_key = self.get_api_keys("openai")
        return OpenAI(
            api_key=api_key,
        )

    def get_model_name(self):
        return self.model

    def _message(self, content, text_only=True, text_parser=None):
        msg_content = (self.prompt + "\n" + content) if self.prompt else content

        openai_args = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": msg_content}],
        }

        if self.model in ["gpt-5-2025-08-07"]:
            openai_args["max_completion_tokens"] = self.max_tokens
            openai_args["temperature"] = 1
            if self.temperature != 1:
                print(
                    f"Warning: For model {self.model}, temperature is set to 1 by default. Overriding user-specified temperature {self.temperature} to 1."
                )
                self.temperature = 1
        else:
            openai_args["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**openai_args)
        if text_only:
            response = response.choices[0].message.content

            if text_parser:
                response = text_parser(response)

        return response


class OpenAIEmbeddingClient:
    """OpenAI embeddings client"""

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, model=None, **kwargs):
        self.max_retries = kwargs.get("max_retries", 3)
        self.base_delay = kwargs.get("base_delay", 1.5)
        self.backoff_factor = kwargs.get("backoff_factor", 2.0)
        self.model = model or self.DEFAULT_MODEL
        self.client = self._get_openai_client()

    def _get_openai_client(self) -> OpenAI:
        api_key = self._get_api_keys("openai")
        if not api_key:
            raise ValueError("OpenAI API key missing. Please set it in configs/api_keys.json")
        return OpenAI(api_key=api_key)

    def _get_api_keys(self, key: str):
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir / "../configs" / "api_keys.json"
        with open(config_path) as f:
            api_keys = json.load(f)
        return api_keys.get(key)

    def get_model_name(self) -> str:
        return self.model

    def embed(self, text: str):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if text.strip() == "":
            raise ValueError("text must be non-empty")

        sanitized_text = text.replace("\n", " ")

        last_err = None
        for i in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=sanitized_text)
                return resp.data[0].embedding
            except Exception as e:
                last_err = e
                if i < self.max_retries - 1:
                    time.sleep(self.base_delay * (self.backoff_factor ** i))
                else:
                    break
        assert last_err is not None
        raise last_err

    def embed_many(self, texts):
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be a list of strings")
        if len(texts) == 0:
            return []

        sanitized = [t.replace("\n", " ") for t in texts]

        last_err = None
        for i in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=sanitized)
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                if i < self.max_retries - 1:
                    time.sleep(self.base_delay * (self.backoff_factor ** i))
                else:
                    break
        assert last_err is not None
        raise last_err

if __name__ == "__main__":
    client = OpenAIClient()
    msg = client.message("Hello, how are you?")
    print(msg)
