import bz2
import functools
import json
import os
import time
from datetime import datetime

from loguru import logger
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

from generation.prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS

tokenizer = LlamaTokenizerFast.from_pretrained("models/tokenizer")
time_logs = {}


def timer(log_name):
    """
    A decorator for timing functions
    """

    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            runtime = end_time - start_time
            if log_name in time_logs.keys():
                time_logs[log_name].append(runtime)
            else:
                time_logs[log_name] = [runtime]
            return value

        return wrapper_repeat

    return decorator_repeat


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens using Llama2 tokenizer"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """Helper function to create an empty batch."""
        return {
            "interaction_id": [],
            "query": [],
            "search_results": [],
            "query_time": [],
            "answer": [],
        }

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


@timer("generate_prediction")
def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    for batch in tqdm(
        load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"
    ):
        batch_ground_truths = batch.pop(
            "answer"
        )  # Remove answers from batch and store them
        batch_predictions = participant_model.batch_generate_answer(batch)

        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

    return queries, ground_truths, predictions


@timer("evaluate_predictions")
def evaluate_predictions(queries, ground_truths, predictions, evaluation_model):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()

    for _idx, prediction in enumerate(
        tqdm(predictions, total=len(predictions), desc="Evaluating Predictions")
    ):
        query = queries[_idx]
        ground_truth = ground_truths[_idx].strip()
        # trim prediction to 75 tokens using Llama2 tokenizer
        prediction = trim_predictions_to_max_token_length(prediction)
        prediction = prediction.strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        prompt_for_eval_model = (
            f"""System:\n{INSTRUCTIONS}\nExamples:\n{IN_CONTEXT_EXAMPLES}"""
        )
        prompt_for_eval_model += f"\nUser:\nQuestion: {query}\nGround truth: {ground_truth}\nPrediction: {prediction}\nAsistant:"

        if "i don't know" in prediction_lowercase:
            n_miss += 1
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            continue

        response = evaluation_model.respond(prompt_for_eval_model)
        if response:
            log_response(prompt_for_eval_model, response)
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
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
