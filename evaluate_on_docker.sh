#!/bin/bash

docker run --gpus all \
-v $(pwd):/rag-project -it \
rag:latest \
python /rag-project/local_evaluation.py \
--config=/rag-project/config/default_config.yaml \
--data_path=/rag-project/example_data/dev_data.jsonl.bz2 \