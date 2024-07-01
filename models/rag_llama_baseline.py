import os
from typing import Any, Dict, List

import numpy as np
import torch
import vllm
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.evaluation_utils import timer
from models.chunk_extractor import ChunkExtractor
from models.utils import load_config


class RAGModel:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """

    def __init__(self, config_path="config/default_config.yaml"):
        # Load configuration
        self.CONFIG = load_config(config_path)
        self.initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def initialize_models(self):
        # Initialize the Chat Model
        self.model_name = self.CONFIG["ChatModelParams"]["MODEL_PATH"]

        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
            
            https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md
            """
            )

        # Initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=self.CONFIG["ChatModelParams"][
                "VLLM_TENSOR_PARALLEL_SIZE"
            ],
            gpu_memory_utilization=self.CONFIG["ChatModelParams"][
                "VLLM_GPU_MEMORY_UTILIZATION"
            ],
            trust_remote_code=True,
            dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Initialize a sentence transformer model
        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            self.CONFIG["EmbeddingModelParams"]["MODEL_PATH"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            trust_remote_code=True,
        )
        # Initialize a reranking transformer model
        self.reranker = CrossEncoder(
            "models/sentence-transformers/BAAI/bge-reranker-v2-m3"
        )

    @timer("calculate_embeddings")
    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=self.CONFIG["EmbeddingModelParams"][
                "SENTENTENCE_TRANSFORMER_BATCH_SIZE"
            ],
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers.
        #       todo: this can also be done in a Ray native approach.
        #
        return embeddings

    @timer("get_reranking_scores")
    def get_reranking_scores(self, query, chunks):
        rank = self.reranker.rank(query, chunks)
        idx, scores = [], []
        for r in rank:
            idx.append(r["corpus_id"])
            scores.append(r["score"])
        idx = np.array(idx)
        scores = np.array(scores)
        return idx, scores

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        The evaluation timeouts linearly scale with the batch size.
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout


        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = self.CONFIG["RagSystemParams"][
            "AICROWD_SUBMISSION_BATCH_SIZE"
        ]
        return self.batch_size

    @timer("batch_generate_answer")
    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-cosine_scores).argsort()[
                    : 7 * self.CONFIG["RagSystemParams"]["NUM_CONTEXT_SENTENCES"]
                ]
            ]
            # rerank results
            ranking_idx, ranking_scores = self.get_reranking_scores(
                query, retrieval_results
            )
            retrieval_results = retrieval_results[
                ranking_idx[np.argsort(ranking_scores)[::-1]]
            ]
            retrieval_results = retrieval_results[
                : self.CONFIG["RagSystemParams"]["NUM_CONTEXT_SENTENCES"]
            ]

            # You might also choose to skip the steps above and
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)

        # Prepare formatted prompts from the LLM
        formatted_prompts = self.format_prompts(
            queries, query_times, batch_retrieval_results
        )

        # Generate responses via vllm
        responses = self.llm.generate(
            formatted_prompts,
            vllm.SamplingParams(
                n=self.CONFIG["ChatModelParams"]["N_OUT_SEQ"],
                top_p=self.CONFIG["ChatModelParams"]["TOP_P"],
                temperature=self.CONFIG["ChatModelParams"]["TEMPERATURE"],
                skip_special_tokens=self.CONFIG["ChatModelParams"][
                    "SKIP_SPECIAL_TOKENS"
                ],
                max_tokens=self.CONFIG["ChatModelParams"]["MAX_TOKENS"],
            ),
            use_tqdm=False,  # you might consider setting this to True during local development
        )

        # Aggregate answers into List[str]
        answers = []
        for response in responses:
            answers.append(response.outputs[0].text)

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[]):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.

        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        """
        system_prompt = "You are provided with a question and various references. Your task is to answer the question succinctly, using the fewest words possible. If the references do not contain the necessary information to answer the question, respond with 'I don't know'. There is no need to explain the reasoning behind your answers."
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""

            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                for _snippet_idx, snippet in enumerate(retrieval_results):
                    references += f"- {snippet.strip()}\n"

            references = references[
                : self.CONFIG["RagSystemParams"]["MAX_CONTEXT_REFERENCES_LENGTH"]
            ]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            formatted_prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return formatted_prompts
