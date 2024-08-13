import ray
import numpy as np
from collections import defaultdict
from bs4 import BeautifulSoup
from blingfire import text_to_sentences_and_offsets
from models.utils import load_config
import justext

class ChunkExtractor:
    def __init__(self, config_path="config/default_config.yaml"):
        # Load configuration
        self.CONFIG = load_config(config_path)

    def _sentence_chunking(self, html_source):
        '''
        Extracts sentences chunks from given HTML source.
        Input:
            html_source (str): HTML content from which to extract text.
        Output:
            List[str]: A list of sentences extracted from the HTML content.
        '''
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(
            " ", strip=True
        )  # Use space as a separator, strip whitespaces

        if not text:
            # Return a list with empty string when no text is extracted
            return [""]

        # Extract offsets of sentences from the text
        _, offsets = text_to_sentences_and_offsets(text)

        # Initialize a list to store sentences
        sentence_chunks = []

        # Iterate through the list of offsets and extract sentences
        for start, end in offsets:
            # Extract the sentence and limit its length
            sentence = text[start:end][: self.CONFIG['RagSystemParams']['MAX_CONTEXT_SENTENCE_LENGTH']]
            sentence_chunks.append(sentence)

        return sentence_chunks
    
    def _paragraph_chunking(self, html_source):
        '''
        Extracts paragraph chunks from given HTML source.
        Input:
            html_source (str): HTML content from which to extract text.
        Output:
            List[str]: A list of paragraphs extracted from the HTML content.
        '''
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        text = soup.get_text(
            " ", strip=False
        )  # Use space as a separator, strip whitespaces
        
        # Split the input text by newline characters
        parts = text.split('\n')
        
        # Filter out the empty strings and whitespace-only strings
        paragraph_chunks = [part.strip() for part in parts if part.strip()]
        # Limit the paragraph length
        paragraph_chunks = [paragraph[: self.CONFIG['RagSystemParams']['MAX_CONTEXT_SENTENCE_LENGTH']] 
                            for paragraph in paragraph_chunks]
        
        return paragraph_chunks
    
    def _sentence_chunking_without_boilerplate(self, html_source):
        '''
        Extracts sentences chunks after removing the boilerplate 
        (such as navigation links, headers, and footers) from given HTML source.
        Input:
            html_source (str): HTML content from which to extract text.
        Output:
            List[str]: A list of sentences extracted from the HTML content.
        '''
        # Use justext to remove boilerplate and extract meaningful text
        paragraphs = justext.justext(html_source, 
                                     justext.get_stoplist("English"),
                                     max_link_density=1)
        paragraph_chunks = [paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate]
        if paragraph_chunks != []:
            # Split paragraphs into sentences
            sentence_chunks = []
            for paragraph in paragraph_chunks:
                # Extract offsets of sentences from the text
                _, offsets = text_to_sentences_and_offsets(paragraph)

                # Iterate through the list of offsets and extract sentences
                for start, end in offsets:
                    # Extract the sentence and limit its length
                    sentence = paragraph[start:end][: self.CONFIG['RagSystemParams']['MAX_CONTEXT_SENTENCE_LENGTH']]
                    sentence_chunks.append(sentence)

        # If no paragraphs are found use the old sentence chunking method
        else:
            sentence_chunks = self._sentence_chunking(html_source)    
        
        return sentence_chunks
    
    def _paragraph_chunking_without_boilerplate(self, html_source):
        '''
        Extracts paragraphs chunks after removing the boilerplate 
        (such as navigation links, headers, and footers) from given HTML source.
        Input:
            html_source (str): HTML content from which to extract text.
        Output:
            List[str]: A list of paragraphs extracted from the HTML content.
        '''
        # Use justext to remove boilerplate and extract meaningful text
        paragraphs = justext.justext(html_source, 
                                     justext.get_stoplist("English"), 
                                     max_link_density=1)
        paragraph_chunks = [paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate]
        # Limit the paragraph length
        paragraph_chunks = [paragraph[: self.CONFIG['RagSystemParams']['MAX_CONTEXT_SENTENCE_LENGTH']] 
                            for paragraph in paragraph_chunks]
        
        if paragraph_chunks == []:
            paragraph_chunks = self._paragraph_chunking(html_source)         
        
        return paragraph_chunks
    
    @ray.remote
    def _extract_chunks(self, interaction_id, html_source):
        """
        Extracts and returns chunks from given HTML source.

        Note: This function is for demonstration purposes only.
        We are treating an independent sentence as a chunk here,
        but you could choose to chunk your text more cleverly than this.

        Parameters:
            interaction_id (str): Interaction ID that this HTML source belongs to.
            html_source (str): HTML content from which to extract text.

        Returns:
            Tuple[str, List[str]]: A tuple containing the interaction ID and a list of sentences extracted from the HTML content.
        """
        if html_source == "":
            # Return a list with empty string when no text is extracted
            return interaction_id, [""]
        
        # Extract chunks from the HTML source
        chunks = self._paragraph_chunking_without_boilerplate(html_source)

        return interaction_id, chunks



    def extract_chunks(self, batch_interaction_ids, batch_search_results):
        """
        Extracts chunks from given batch search results using parallel processing with Ray.

        Parameters:
            batch_interaction_ids (List[str]): List of interaction IDs.
            batch_search_results (List[List[Dict]]): List of search results batches, each containing HTML text.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        # Setup parallel chunk extraction using ray remote
        ray_response_refs = [
            self._extract_chunks.remote(
                self,
                interaction_id=batch_interaction_ids[idx],
                html_source=html_text["page_result"],
            )
            for idx, search_results in enumerate(batch_search_results)
            for html_text in search_results
        ]

        # Wait until all sentence extractions are complete
        # and collect chunks for every interaction_id separately
        chunk_dictionary = defaultdict(list)

        for response_ref in ray_response_refs:
            interaction_id, _chunks = ray.get(
                response_ref
            )  # Blocking call until parallel execution is complete
            chunk_dictionary[interaction_id].extend(_chunks)

        # Flatten chunks and keep a map of corresponding interaction_ids
        chunks, chunk_interaction_ids = self._flatten_chunks(chunk_dictionary)

        return chunks, chunk_interaction_ids

    def _flatten_chunks(self, chunk_dictionary):
        """
        Flattens the chunk dictionary into separate lists for chunks and their corresponding interaction IDs.

        Parameters:
            chunk_dictionary (defaultdict): Dictionary with interaction IDs as keys and lists of chunks as values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing an array of chunks and an array of corresponding interaction IDs.
        """
        chunks = []
        chunk_interaction_ids = []

        for interaction_id, _chunks in chunk_dictionary.items():
            # De-duplicate chunks within the scope of an interaction ID
            unique_chunks = list(set(_chunks))
            chunks.extend(unique_chunks)
            chunk_interaction_ids.extend([interaction_id] * len(unique_chunks))

        # Convert to numpy arrays for convenient slicing/masking operations later
        chunks = np.array(chunks)
        chunk_interaction_ids = np.array(chunk_interaction_ids)

        return chunks, chunk_interaction_ids
