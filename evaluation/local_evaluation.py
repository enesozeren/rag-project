import argparse
import logging
import time
from datetime import datetime

import yaml

from . import evaluation_utils
from .evaluation_utils import time_logs, timer
from .model import EvaluationModel
from models.user_config import UserModel
from models.utils import load_config
import os

if __name__ == "__main__":
    """
    Run the local evaluation script with a configuration file.
    EX: 
        python local_evaluation.py 
        --config=config/default_config.yaml
        --data_path=example_data/dev_data.jsonl.bz2
    
    You can find the logs in the logs folder.
    """
    parser = argparse.ArgumentParser(description="Script to run with a config file.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file."
    )

    args = parser.parse_args()
    config_path = args.config
    # The test dataset path
    dataset_path = args.data_path

    # Start the timer
    start_time = time.perf_counter()

    # Log the model/user_config.py content
    user_config_path = os.path.join("models", "user_config.py")
    with open(user_config_path, "r") as user_config_file:
        user_config_content = user_config_file.read()

    # Load the configuration and log it
    config = load_config(config_path)

    # modelname
    model_name = config["EmbeddingModelParams"]["MODEL_PATH"]
    model_name = model_name[model_name.rfind("/") + 1 :]

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a log file
    logging.basicConfig(
        filename=f"logs/ex_{model_name}_{time_str}.log", level=logging.INFO
    )

    logging.info("\n---DATASET PATH---:\n%s", dataset_path)
    logging.info("\n---MODELS/USER_CONFIG.PY FILE---:\n%s", user_config_content)
    logging.info("\n---CONFIG PATH---:\n%s", config_path)
    logging.info(
        "\n---CONFIGS---:\n%s",
        yaml.dump(config, default_flow_style=False, sort_keys=False),
    )

    # Generate predictions
    participant_model = UserModel(config_path=config_path)
    queries, ground_truths, predictions = evaluation_utils.generate_predictions(
        dataset_path, participant_model
    )

    # Log queries, ground truths, and predictions
    rag_output_file = os.path.join("outputs", f"rag_outputs_{time_str}.txt")
    with open(rag_output_file, "w") as file:
        file.write("Queries:\n")
        for query in queries:
            file.write(query + "\n")
        file.write("\nGround Truths:\n")
        for ground_truth in ground_truths:
            file.write(ground_truth + "\n")
        file.write("\nPredictions:\n")
        for prediction in predictions:
            file.write(prediction + "\n")

    # Initialize evaluation model
    evaluation_model = EvaluationModel(
        config_path, participant_model.llm, participant_model.tokenizer
    )

    # Evaluate Predictions
    evaluation_results = evaluation_utils.evaluate_predictions(
        queries, ground_truths, predictions, evaluation_model
    )

    duration = time.perf_counter() - start_time

    # Log the evaluation results
    logging.info(
        "\n---EVALUATION RESULTS---:\n%s",
        yaml.dump(evaluation_results, default_flow_style=False, sort_keys=False),
    )

    time_logs["total_duration"] = [duration]
    time_logs = {key: sum(value) / 60 for key, value in time_logs.items()}
    time_logs = {
        key: "{:.2f}".format(value) + " Min" for key, value in time_logs.items()
    }
    logging.info("\n ---EXPERIMENT DURATION---\n%s", yaml.dump(time_logs))
