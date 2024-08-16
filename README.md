# Meta KDD Cup '24: LMU Applied DL Team

Ludwig-Maximilians-Universität (LMU) 2024 Summer Semester - Applied Deep Learning Project <br>
Check our report [here](report/rag-report.pdf)

**Team Members**:

- Ercong Nie (Supervisor)
- Daniel Gloukhman
- Enes Özeren
- Julia Broden

**Introduction**:
This project uses the baseline RAG template from Meta KDD CRAG Challenge. We have focused on the first task in the challenge (Retrieval Summarization) and conducted experiments to improve this baseline RAG system.

- Main Components of this RAG system:
  - Embedding Model: Creates the embedding vectors for given information resources and user queries.
  - Retrieval Process: We use cosine similarity to retrieve the most related information for the user query.
  - LLM: Answers the user's query with retrieved information.
- Dataset: We have used subsamples of the dataset from the challenge to evaluate our experiments. The dataset contains question, ground truth answer, html resources which might contain the information required for the right answer but it is not guaranteed.
- Evaluation: We have used another LLM to evaluate the given responses from our RAG System
- We have experimented with embedding models, LLMs, hyperparameters of the LLM and our rag system, chunking, re-ranking of retrieved documents, instruction prompts. Please see the `report` directory for details.

# File Structure

```
.
├── config                                      <- contains configuration files for RAG system
├── docs                                        <- contains documents for code and meta kdd cup challenge
│   ├── baselines.md
│   ├── batch_prediction_interface.md
│   ├── dataset.md
│   ├── download_baseline_model_weights.md
│   ├── hardware-and-system-config.md
│   ├── runtime.md
│   └── submission.md
├── evaluation                                  <- script directory for local evaluation utility functions
│   └── Dockerfile
|   ├── model.py                                <- EvaluationModel for evaluations with given open source model in config
│   ├── evaluate_on_docker.sh
│   ├── evaluate.sh
│   ├── evaluation_utils.py
│   ├── __init__.py
│   ├── model.py                                <- open source evaluation model
│   └── local_evaluation.py                     <- evaluation script with open source models contained in models/ dir
├── example_data                                <- datasets for local evaluation of RAG system
│   ├── dev_data.jsonl.bz2                                  <- 10 samples of query & resources
│   ├── subsampled_crag_task_1_dev_v3_release.jsonl.bz2     <- 333 samples of query & resources
│   └── subsampling_dataset.py                              <- script to create sample datasets
├── generation                                              <- module containing the augumented generation of the RAG-system
│   ├── chatmodel.py
│   ├── __init__.py
│   ├── prompts
├── logs                                        <- contains local evaluation experiment logs
├── models                                      <- directory for model weights
│   ├── user_config.py                          <- To be used to submit our RAG system class to competition
│   ├── utils.py                                <- Util functions for RAG system class
├── prompts
│   └── templates.py                            <- prompt templates for evaluation model
├── report                                      <- Report for our experiments & findings in pdf format
├── retrieval
│   ├── chunk_extractor.py                      <- ChunkExtractor class for creating chunks given html resources
│   ├── __init__.py
│   └── vectordb.py                             <- Module containing the code for retrival in RAG
├── aicrowd.json                                <- submission info json
├── apt.txt
├── README.md
└── requirements.txt
```

# Downloading Model Weights

To run / evaluate the RAG systems with the methods mentioned below, you need to download and save the model weights.
Here we illustrate downloading model weights for our best resulting setup.
You need a huggingface account and access to Llama model weights.

1. **Login via CLI**:

   Authenticate yourself with the Hugging Face CLI using the token created in the previous step. Run:

   ```bash
   huggingface-cli login
   ```

   When prompted, enter the token.

2. **Download LLaMA-3-70B-Instruct Model**:

   Execute the following command to download the `Meta-Llama-3-70B-Instruct` model to a local subdirectory. This command excludes unnecessary files to save space:

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
       meta-llama/Meta-Llama-3-70B-Instruct \
       --local-dir-use-symlinks False \
       --local-dir models/meta-llama/Meta-Llama-3-70B-Instruct \
       --exclude *.pth # These are alternates to the safetensors hence not needed
   ```

3. **Download MiniLM-L6-v2 Model (for sentence embeddings)**:

   Similarly, download the `sentence-transformers/gtr-t5-xl` model using the following command:

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
      sentence-transformers/gtr-t5-xl \
       --local-dir-use-symlinks False \
       --local-dir models/sentence-transformers/gtr-t5-xl \
       --exclude *.bin *.h5 *.ot # These are alternates to the safetensors hence not needed
   ```

4. **Download bge-reranker-v2-m3 Model (for reranking)**:

   Similarly, download the `BAAI/bge-reranker-v2-m3` model using the following command:

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
      BAAI/bge-reranker-v2-m3 \
       --local-dir-use-symlinks False \
       --local-dir models/reranker/BAAI/bge-reranker-v2-m3
   ```

After downloading and saving the model weights with the codes above you can run the evaluation script.

# 📏 Evaluation

Follow the steps here for your experiments:

1. Create a configuration file in config directory to store your parameters and model weight directories (Ex: config/default_config.yaml). You can check one of the existing config files for the syntax.

2. Run the local evaluation file by giving the config file you created

```bash
python -m evaluation.local_evaluation \
--config=config/default_config.yaml \
--data_path=example_data/dev_data.jsonl.bz2
```

3. You can find the logs in the logs folder by checking the date in the log file name

Note: local_evaluation.py script is used for evaluation with open source model instead of a OpenAI GPT api. Do not confuse this with running the evaluations on local machine, you would probably need some GPUs to run local_evaluation.py script. (With Llama 3 70B chat model & all-MiniLM-L6-v2 embedding model we recommand at least 2 nvidia A100 GPUs)

# Logs

When `local_evaluation.py` script executed there will be a log file created in `logs` directory.
Logs contains:

```
1. ---DATASET PATH---               for logging which dataset is used for evaluation
2. ---MODELS/USER_CONFIG.PY FILE--- for logging which class used as rag system
3. ---CONFIG PATH---                for logging which config file used to execute the script
4. ---CONFIGS---                    for logging the parameters in the used config file
5. ---EVALUATION RESULTS---         for logging the metric values after evaluation executed
6. ---EXPERIMENT DURATION---        for profiling purposes
```

# Docker

You can build the docker image with the following command.

```bash
docker build -t rag:latest -f evaluation/Dockerfile .
```

To run the evaluation on docker container use the following bash script. Note that you need gpus for this.

```bash
bash evaluation/evaluate_on_docker.sh
```

# Further Documents

To see further documentation about this repository & Meta KDD challenge, check **docs/** directory.

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024
