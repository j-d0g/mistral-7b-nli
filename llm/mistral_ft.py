from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfFolder


class MistralNLI():
    def __init__(self, hf_token, base_model="TheBloke/Mistral-7B-v0.1-GPTQ", peft_model="jd0g/mistral-7b-nli_cot_qkv"):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=False,
            revision="main"
        ).cuda()
        HfFolder.save_token(hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        self.peft_model = PeftModel.from_pretrained(self.base_model, peft_model)
        self.models = {
            "TheBloke/Mistral-7B-v0.1-GPTQ": ["Mistral-7B-NLI-v0.1", "Mistral-7B-NLI-v0.2", "mistral-7b-nli_cot_qkv",
                                              "mistral-7b-gs"]}

    def get_models(self):
        return self.models

    def generate_text(self, prompt, model_name=None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model = self.peft_model if model_name == "peft" else self.base_model

        response = model.generate(
            input_ids=inputs["input_ids"].to("cuda"),
            max_new_tokens=256,
            early_stopping=True
        )

        generated_text = self.tokenizer.batch_decode(response)[0]
        return generated_text

    @classmethod
    def prompt_template(cls, system: str, prompt: str = None):
        return f'''[INST] {system} \n{prompt} \n[/INST]'''