import requests
from llm.base_llm import BaseLLM


class Mistral(BaseLLM):
    def __init__(self, api_key):
        super().__init__()
        self.api_url = "https://api.mistral.ai/v1/"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def get_models(self):
        response = requests.get(self.api_url + "models", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def generate_text(self, model_name="open-mistral-7b", max_tokens=1024, temperature=0.7, top_p=1.0) -> str:
        # Make sure model_name is not None
        if model_name is None:
            model_name = "open-mistral-7b"
            
        # Ensure we have messages
        messages = self.get_messages()
        if not messages:
            # Add a default user message if none exists
            messages = [{"role": "user", "content": "Hello"}]
            
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
            "safe_prompt": False,
            "random_seed": 1337,
            "response_format": {"type": "json_object"}
        }
        
        print(f"DEBUG: Sending request with model={model_name}")
        print(f"DEBUG: Messages count: {len(messages)}")
        if len(messages) > 0:
            print(f"DEBUG: First message role: {messages[0].get('role', 'unknown')}")
        
        response = requests.post(self.api_url + "chat/completions", headers=self.headers, json=data)

        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            return generated_text
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    @classmethod
    def prompt_template(cls, role: str, message: str) -> dict[str, str]:
        # Mistral doesn't use [INST] tags in their API format
        return {
            "role": role,
            "content": message
        }
