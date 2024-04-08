import os
from typing import List

import numpy as np
import torch
from models.utils import trim_predictions_to_max_token_length
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

######################################################################################################
######################################################################################################
###
### IMPORTANT !!!
### Before submitting, please follow the instructions in the docs below to download and check in :
### the model weighs. 
### 
###  https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/download_baseline_model_weights.md
### 
###
### DISCLAIMER: This baseline has NOT been tuned for performance
###             or efficiency, and is provided as is for demonstration.
######################################################################################################


# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.

# Please refer to https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api
# for more information on the MockAPI.
#
# **Note**: This environment variable will not be available for Task 1 evaluations.
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

class ChatModel:
    def __init__(self):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        self.prompt_template = """You are given a quesition and references which may or may not help answer the question. Your goal is to answer the question in as few words as possible.
### Question
{query}

### Answer"""
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        model_name = "models/meta-llama/Llama-2-7b-chat-hf"
        
        if not os.path.exists(model_name):
            raise Exception(f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
            """)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )

        self.generation_pipe = pipeline(task="text-generation",
                                        model=self.llm,
                                        tokenizer=self.tokenizer,
                                        max_new_tokens=75)

    def generate_answer(self, query: str, search_results: List[str]) -> str:
        """
        Generate an answer based on a provided query and a list of pre-cached search results.

        Parameters:
        - query (str): The user's question or query input.
        - search_results (List[str]): A list containing the text content from web pages
          retrieved as search results for the query. Each element in the list is a string
          representing the HTML text of a web page.

        Returns:
        - (str): A plain text response that answers the query. This response is limited to 75 tokens.
          If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 10 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """

        final_prompt = self.prompt_template.format(query=query)
        result = self.generation_pipe(final_prompt)[0]['generated_text']
        answer = result.split("### Answer")[1].strip()
                
        # Trim prediction to a max of 75 tokens
        trimmed_answer = trim_predictions_to_max_token_length(answer)
        
        return trimmed_answer
