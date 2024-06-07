import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from models.utils import load_config
import re

class EvaluationModel:
    def __init__(self, config_path="config/default_config.yaml"):
        # Load configuration
        self.CONFIG = load_config(config_path)
        
        # Initialize the Evaluation Model
        model_name = self.CONFIG['EvaluationModelParams']['MODEL_PATH']

        if not os.path.exists(model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {model_name}

            Please follow the instructions in the docs/download_baseline_model_weights document 
            to download and check in the model weights.
            """
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        # Get the device where the model is located
        self.device = next(self.llm.parameters()).device        

    def respond(self, query):
        
        inputs = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
        
        # Generate prediction from model
        outputs = self.llm.generate(inputs, max_new_tokens=8)
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Find the index where generated text starts
        start_index = len(query)
        # Get the generated text
        generated_part = output_text[start_index:].strip()
        
        # Use regular expression to find JSON part
        accuracy_json_match = re.search(r'({.*})', generated_part)
        if accuracy_json_match:
            accuracy_json = accuracy_json_match.group(1)
        else:
            raise ValueError("JSON data not found in the generated text.")
        
        return accuracy_json