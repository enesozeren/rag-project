import numpy as np
import torch
from .chunk_extractor import ChunkExtractor
from sentence_transformers import CrossEncoder, SentenceTransformer

from evaluation.evaluation_utils import timer
from models.utils import load_config


class VectorDB:
    """
    An in-memory vector database for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of the retrieval stage of a RAG lifecycle.
    """

    def __init__(
        self,
        config_path="config/default_config.yaml",
    ):
        # Load configuration
        self.CONFIG = load_config(config_path)
        self._initialize_models()
        self.chunk_extractor = ChunkExtractor()

    def _initialize_models(self):
        self.sentence_model = SentenceTransformer(
            self.CONFIG["EmbeddingModelParams"]["MODEL_PATH"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            trust_remote_code=True,
        )
        # Initialize a reranking transformer model
        self.reranker = CrossEncoder(
            self.CONFIG["EmbeddingModelParams"]["RERANKING_MODEL_PATH"]
        )

    def set_data(self, batch_interaction_ids, batch_search_results):
        """
        Removes stored data from memory and sets it to passed data
        """
        if hasattr(self, "_chunk_interaction_ids"):
            del self._chunk_interaction_ids
        if hasattr(self, "embeddings"):
            del self.embeddings
        if hasattr(self, "_chunks"):
            del self._chunks

        self.batch_interaction_ids = batch_interaction_ids
        # Chunk all search results using ChunkExtractor
        self._chunks, self._chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )
        # Calculate all chunk embeddings
        self.embeddings = self.calculate_embeddings(self._chunks)

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
    def _get_reranking_scores(self, query, chunks):
        rank = self.reranker.rank(query, chunks)
        idx, scores = [], []
        for r in rank:
            idx.append(r["corpus_id"])
            scores.append(r["score"])
        idx = np.array(idx)
        scores = np.array(scores)
        return idx, scores

    def rerank(self, query, retrieval_results):
        # rerank results
        ranking_idx, ranking_scores = self._get_reranking_scores(
            query, retrieval_results
        )
        retrieval_results = retrieval_results[
            ranking_idx[np.argsort(ranking_scores)[::-1]]
        ]
        retrieval_results = retrieval_results[
            : self.CONFIG["RagSystemParams"]["NUM_CONTEXT_SENTENCES"]
        ]
        return retrieval_results

    def get_top_k(self, queries, query_times, reranking: bool = True):
        """
        Retrieves the top k most similiar chunks w.r.t cosine similarity of their embeddings to the queries
        """
        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(self.batch_interaction_ids):
            query = queries[_idx]
            query_time = query_times[_idx]  # currently not used
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = self._chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = self._chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = self.embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores = (relevant_chunks_embeddings * query_embedding).sum(1)

            # and retrieve top-N results.

            top_k = (
                self.CONFIG["EmbeddingModelParams"]["TOP_K_BEFORE_RERANKING"]
                if reranking
                else self.CONFIG["RagSystemParams"]["NUM_CONTEXT_SENTENCES"]
            )
            retrieval_results = relevant_chunks[(-cosine_scores).argsort()[:top_k]]

            if reranking:
                retrieval_results = self.rerank(query, retrieval_results)

            # You might also choose to skip the steps above and
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)
        return batch_retrieval_results
