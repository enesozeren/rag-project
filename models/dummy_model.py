from typing import Any, Dict, List
from models.utils import trim_predictions_to_max_token_length


class DummyModel:
    def __init__(self):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        pass

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = 4
        return self.batch_size

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
        search_results = batch["search_results"]
        query_times = batch["query_time"]

        answers = []
        for idx, query in enumerate(queries):
            # Implement logic to generate answers based on search results and query times
            answers.append("i don't know")  # Default placeholder response

        return answers
