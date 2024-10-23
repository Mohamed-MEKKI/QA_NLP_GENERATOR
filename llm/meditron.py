from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
from config import variables
import torch


class BaseModel:
    def __init__(self, model_id: str, access_token: str):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=access_token)
        self.model_config = AutoConfig.from_pretrained(
            model_id, use_auth_token=access_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=self.model_config,
            device_map='auto',
            use_auth_token=access_token
        )

    def model_eval(self):
        self.model.eval()

    def get_memory_footprint(self):
        print(self.model.get_memory_footprint())

    def len_tokens_text(self, text):
        # tokenize the text
        tokens = self.tokenizer.encode(text)
        # print(tokens)
        return (len(tokens))

    def text_generation_pipeline(self):
        text_gen_pipeline = pipeline(
            model=self.model, tokenizer=self.tokenizer,
            return_full_text=True,
            task='text-generation',
            temperature=0.01,
            max_new_tokens=512,
            repetition_penalty=1.1
        )
        return text_gen_pipeline

    def meditron(self):
       return HuggingFacePipeline(pipeline=self.text_generation_pipeline())

