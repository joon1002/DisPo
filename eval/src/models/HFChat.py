from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .Model import Model


class HFChat(Model):
    """HuggingFace chat template 기반 범용 생성기 (Qwen, Mistral 등)."""

    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def query(self, msg):
        try:
            messages = [{"role": "user", "content": msg}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_output_tokens,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception:
            return ""
