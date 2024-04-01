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

def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and ((model_resp["accuracy"] is True) or (isinstance(model_resp["accuracy"], str) and model_resp["accuracy"].lower() == "true")):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1

def evaluate(dataset_path, model_name):
    qa = load_json_file(os.path.join(dataset_path, "qa.json"))
    web_results = load_json_file(os.path.join(dataset_path, "web.json"))
    openai_client = OpenAI()
    participant_model = UserModel()

    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()

    for query_dict, query_web_search_results in tqdm(zip(qa, web_results), total=len(qa)):
        query, ground_truth = query_dict['query'], query_dict['answer'].strip().lower()
        prediction = participant_model.generate_answer(query, query_web_search_results)
        prediction = prediction.strip().lower()
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"},
        ]
        if prediction == "i don't know":
            n_miss += 1
            continue
        if prediction == ground_truth:
            n_correct_exact += 1
            n_correct += 1
            continue

        response = attempt_api_call(openai_client, model_name, messages)
        if response:
            log_response(messages, response)
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1

    n = len(qa)
    results = {
        "score": (2*n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,        
    }
    logger.info(results)
    return results

if __name__ == '__main__':
    DATASET_PATH = "example_data/"
    MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-4-0125-preview")
    evaluate(DATASET_PATH, MODEL_NAME)
