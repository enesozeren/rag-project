import json
import os
from datetime import datetime

from loguru import logger
from models.user_config import UserModel
from openai import APIConnectionError, OpenAI, RateLimitError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)

def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES

def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    #todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model_name, messages=messages)
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def log_response(messages, response):
    """Save the response from the API to a file."""
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    with open(f"api_responses/{file_name}", 'w') as f:
        json.dump({"messages": messages, "response": response}, f)

def evaluate_response(response):
    """Evaluate the response to determine if it's missing or correct."""
    is_missing = "Missing: True" in response
    is_correct = "Accuracy: True" in response
    return is_missing, is_correct

def evaluate(dataset_path, model_name):
    qa = load_json_file(os.path.join(dataset_path, "qa.json"))
    web_results = load_json_file(os.path.join(dataset_path, "web.json"))
    openai_client = OpenAI()
    participant_model = UserModel()
    character_limit = 50  # todo: Make character limit dynamic

    n_miss, n_correct, n_exact = 0, 0, 0
    system_message = get_system_message()

    for query_dict, query_web_search_results in tqdm(zip(qa, web_results), total=len(qa)):
        query, ground_truth = query_dict['q'], query_dict['fact_ans']
        prediction = participant_model.generate_answer(query, query_web_search_results, character_limit=character_limit)[:character_limit]
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"},
        ]
        response = attempt_api_call(openai_client, model_name, messages)
        if response:
            log_response(messages, response)
            miss, correct = evaluate_response(response)
            n_miss += miss
            n_correct += correct
            n_exact += (prediction.strip() == ground_truth.strip())

    results = {
        "Exact Accuracy": n_exact / len(qa),
        "Accuracy": n_correct / len(qa),
        "Missing": n_miss / len(qa),
        "Total": len(qa)
    }
    logger.info(results)

if __name__ == '__main__':
    DATASET_PATH = "example_data/"    
    MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-4-0125-preview")
    evaluate(DATASET_PATH, MODEL_NAME)
