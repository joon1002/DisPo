import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance.
    Imports per provider are deferred so only the provider actually in use gets loaded
    (e.g. vicuna must still work fine even in an environment without the openai package).
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