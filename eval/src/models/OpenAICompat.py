"""OpenAI-compatible API wrapper.

Both Gemini and DeepSeek expose OpenAI-compatible endpoints, so a single
openai package handles them.

Config params:
  base_url  : API endpoint (None uses the OpenAI default)
  api_key   : api_key_info.api_keys[api_key_use]
  gpus      : set to [] (API models don't need a GPU)
"""
from openai import OpenAI
from .Model import Model


class OpenAICompat(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos  = int(config["api_key_info"]["api_key_use"])
        assert 0 <= api_pos < len(api_keys), "api_key_use out of range"

        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        base_url = config["params"].get("base_url", None)

        self.client = OpenAI(
            api_key=api_keys[api_pos],
            base_url=base_url,
        )

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": msg},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[OpenAICompat] API error: {e}")
            return ""
