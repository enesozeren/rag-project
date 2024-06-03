import logging
import yaml
import argparse
from datetime import datetime
from models.utils import load_config
from evaluation import evaluation_utils

from models.evaluation_model import EvaluationModel
from models.user_config import UserModel


if __name__ == "__main__":
    '''
    Run the local evaluation script with a configuration file.
    EX: 
        python local_evaluation.py --config=config/default_config.yaml
    
    You can find the logs in the logs folder.
    '''
    parser = argparse.ArgumentParser(description='Script to run with a config file.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the configuration file.')

    args = parser.parse_args()
    config_path = args.config

    # Create a log file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(filename=f'logs/experiment_{current_time}.log', level=logging.INFO)

    # The test dataset path
    DATASET_PATH = "example_data/subsampled_crag_task_1_dev_v3_release.jsonl.bz2"
    # Log the dataset path
    logging.info('\n---DATASET PATH---:\n%s', DATASET_PATH)

    # Log the model/user_config.py content
    user_config_path = 'models/user_config.py'
    with open(user_config_path, 'r') as user_config_file:
        user_config_content = user_config_file.read()
    logging.info('\n---MODELS/USER_CONFIG.PY FILE---:\n%s', user_config_content)

    # Load the configuration and log it
    config = load_config(config_path)
    logging.info('\n---CONFIG PATH---:\n%s', config_path)
    logging.info('\n---CONFIGS---:\n%s', yaml.dump(config, default_flow_style=False, sort_keys=False))

    # Generate predictions
    participant_model = UserModel(config_path=config_path)
    queries, ground_truths, predictions = evaluation_utils.generate_predictions(
        DATASET_PATH, participant_model
    )

    # Initialize evaluation model
    evaluation_model = EvaluationModel(config_path=config_path)

    # Evaluate Predictions
    evaluation_results = evaluation_utils.evaluate_predictions(
        queries, ground_truths, predictions, evaluation_model
    )

    # Log the evaluation results
    logging.info('\n---EVALUATION RESULTS---:\n%s', yaml.dump(evaluation_results, default_flow_style=False, sort_keys=False))
