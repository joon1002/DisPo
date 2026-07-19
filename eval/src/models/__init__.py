import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance.
    Provider별 import는 실제로 쓰는 provider만 로드하도록 지연 처리
    (예: openai 패키지가 없는 환경에서도 vicuna는 문제없이 쓸 수 있어야 함).
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'vicuna':
        from .Vicuna import Vicuna
        model = Vicuna(config)
    elif provider == 'gpt':
        from .GPT import GPT
        model = GPT(config)
    elif provider == 'llama':
        from .Llama import Llama
        model = Llama(config)
    elif provider == 'hfchat':
        from .HFChat import HFChat
        model = HFChat(config)
    elif provider == 'openai_compat':
        from .OpenAICompat import OpenAICompat
        model = OpenAICompat(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model