import os


class ChatModelParams:
    VLLM_TENSOR_PARALLEL_SIZE: 4  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
    VLLM_GPU_MEMORY_UTILIZATION: 0.85  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

    N_OUT_SEQ = (1,)  # Number of output sequences to return for each prompt.
    TOP_P = (
        0.9,
    )  # Float that controls the cumulative probability of the top tokens to consider.
    TEMPERATURE = (0.1,)  # Randomness of the sampling
    SKIP_SPECIAL_TOKENS = (True,)  # Whether to skip special tokens in the output.
    MAX_TOKENS = (50,)  # Maximum number of tokens to generate per output sequence.

    # Note: We are using 50 max new tokens instead of 75,
    # because the 75 max token limit for the competition is checked using the Llama2 tokenizer.
    # Llama3 instead uses a different tokenizer with a larger vocabulary
    # This allows the Llama3 tokenizer to represent the same content more efficiently,
    # while using fewer tokens.


class EmbeddingModelParams:
    SENTENTENCE_TRANSFORMER_BATCH_SIZE = 128  # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available


class RagSystemParams:
    # Define the number of context sentences to consider for generating an answer.
    NUM_CONTEXT_SENTENCES = 20
    # Set the maximum length for each context sentence (in characters).
    MAX_CONTEXT_SENTENCE_LENGTH = 1000
    # Set the maximum context references length (in characters).
    MAX_CONTEXT_REFERENCES_LENGTH = 4000
    # Batch size you wish the evaluators will use to call the `batch_generate_answer` function
    AICROWD_SUBMISSION_BATCH_SIZE = 8  # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


class APIs:
    # Load the environment variable that specifies the URL of the MockAPI. This URL is essential
    # for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
    # may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
    # the API URL to ensure accurate endpoint communication.

    # Please refer to https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api
    # for more information on the MockAPI.
    #
    # **Note**: This environment variable will not be available for Task 1 evaluations.
    CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")
