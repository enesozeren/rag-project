import os

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