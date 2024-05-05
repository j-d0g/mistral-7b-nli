from openai import OpenAI
from llm.base_llm import BaseLLM


class ChatGPT(BaseLLM):
    def __init__(self, api_key):
        super().__init__()
        self.client = OpenAI(api_key=api_key)

    def get_models(self):
        return self.client.models.list()

    def generate_text(self, model_name="gpt-3.5-turbo-0125", max_tokens=1024, temperature=0.7, top_p=1.00) -> str:
        response = self.client.chat.completions.create(
            model=model_name,
            messages=self.get_messages(),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    @classmethod
    def prompt_template(cls, role: str, message: str) -> dict[str, str]:
        return {
            "role": role,
            "content": message
        }
