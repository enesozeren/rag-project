from typing import List


class DummyModel:
    def __init__(self):
        """ Initialize your models here """
        pass

    def generate_answer(self, query: str, search_results: List[str]) -> str:
        """
        You will be provided with a query and the corresponding pre-cached search results for the query
        
        Inputs - 
            query - String representing the input query
            search_results - List of strings, each comes from scraped HTML text of the search query
        Returns - 
            string response - Your answer in plain text, should be limited to the character limit, 
                              Any longer responses will be trimmed to meet the character limit
        """
        answer = "I'm sorry, I can't help with that."
        return answer