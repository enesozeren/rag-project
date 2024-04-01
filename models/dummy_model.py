import os
from typing import List

# Load the environment variable that specifies the URL of the MockAPI. This URL is essential
# for accessing the correct API endpoint in Task 2 and Task 3. The value of this environment variable
# may vary across different evaluation settings, emphasizing the importance of dynamically obtaining
# the API URL to ensure accurate endpoint communication.
#
# **Note**: This environment will not be available for Task 1 evaluations.
CRAG_MOCK_API_URL = os.getenv("CRAG_MOCK_API_URL", "http://localhost:8000")

class DummyModel:
    def __init__(self):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        pass

    def generate_answer(self, query: str, search_results: List[str]) -> str:
        """
        Generate an answer based on a provided query and a list of pre-cached search results.

        Parameters:
        - query (str): The user's question or query input.
        - search_results (List[str]): A list containing the text content from web pages
          retrieved as search results for the query. Each element in the list is a string
          representing the HTML text of a web page.

        Returns:
        - (str): A plain text response that answers the query. This response is limited to 75 tokens.
          If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 10 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        # Default response when unsure about the answer
        answer = "i don't know"
        
        return answer
