import replicate
from llm.base_llm import BaseLLM


class Llama(BaseLLM):
    def __init__(self, api_key):
        super().__init__()
        self.client = replicate.Client(api_token=api_key)

    def get_models(self):
        return {
            "llama-2-7b": "meta/llama-2-7b-chat",
            "llama-2-13b": "meta/llama-2-13b-chat",
            "llama-2-70b": "meta/llama-2-70b-chat",
            "llama-3-8b": "meta/meta-llama-3-8b-instruct",
            "llama-3-70b": "meta/meta-llama-3-70b-instruct",
        }

    def generate_text(self, model_name="meta/meta-llama-3-8b-instruct", max_tokens=2048, temperature=0.7,
                      top_p=1.0) -> str:

        system = self.messages[0]
        # system = "You are a chess grandmaster. I will give you the move sequence, and you will return your response in JSON format, using 'move' and 'thoughts' as keys. You will return nothing else but this JSON object, with no extra explanations."
        # prompt = "<|begin_of_text|>" + system + self.messages[-1] + "<|start_header_id|>assistant<|end_header_id|>"

        generated_text = self.client.run(
                model_name,
                input={
                    "top_k": 50,
                    "top_p": top_p,
                    "prompt": self.messages[-1],
                    "max_tokens": max_tokens,
                    "stop": None,
                    "min_tokens": 1024,
                    "temperature": temperature,
                    "prompt_template": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "presence_penalty": 1.15,
                    "frequency_penalty": 0.2
                }
        )
        return "".join(generated_text)
    @classmethod
    def prompt_template(cls, role: str, content: str) -> str:
        # return f"<|start_header_id|>{role}<|end_header_id|>{content}<|eot_id|>"
        return content