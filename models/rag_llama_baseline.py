import os
from typing import List

import numpy as np
import torch
from blingfire import text_to_sentences_and_offsets
from bs4 import BeautifulSoup
from models.utils import trim_predictions_to_max_token_length
from sentence_transformers import SentenceTransformer
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


class RAGModel:
    def __init__(self):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        self.sentence_model = SentenceTransformer('models/sentence-transformers/all-MiniLM-L6-v2', device='cuda')

        self.num_context = 10
        self.max_ctx_sentence_length = 1000 # characters

        self.prompt_template = """You are given a quesition and references which may or may not help answer the question. 
You are to respond with just the answer and no surrounding sentences.
If you are unsure about the answer, respond with "I don't know".
### Question
{query}

### References 
{references}

### Answer"""
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        model_name = "models/meta-llama/Llama-2-7b-chat-hf"

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
                                        max_new_tokens=10)


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

        all_sentences = []

        for html_text in search_results:
            soup = BeautifulSoup(html_text['page_result'], features="html.parser")
            text = soup.get_text().replace('\n', '')
            if len(text) > 0:
              offsets = text_to_sentences_and_offsets(text)[1]
              for ofs in offsets:
                  sentence = text[ofs[0]:ofs[1]]
                  all_sentences.append(sentence[:self.max_ctx_sentence_length])
            else:
                all_sentences.append('')

        all_embeddings = self.sentence_model.encode(all_sentences, normalize_embeddings=True)
        query_embedding = self.sentence_model.encode(query, normalize_embeddings=True)[None, :]

        cosine_scores = (all_embeddings * query_embedding).sum(1)
        top_sentences = np.array(all_sentences)[(-cosine_scores).argsort()[:self.num_context]]

        references = ''
        for snippet in top_sentences:
            references += '<DOC>\n' + snippet + '\n</DOC>\n'

        references = ' '.join(references.split()[:500])
        final_prompt = self.prompt_template.format(query=query, references=references)
        result = self.generation_pipe(final_prompt)[0]['generated_text']
        answer = result.split("### Answer\n")[1]
                                
        # Trim prediction to a max of 75 tokens
        trimmed_answer = trim_predictions_to_max_token_length(answer)
        
        return trimmed_answer
