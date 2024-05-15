## Batch Prediction Interface
- Date: `14-05-2024`

Your submitted models can now make batch predictions on the test set, allowing you to fully utilize the multi-GPU setup available during evaluations.

### Changes to Your Code

1. **Add a `get_batch_size()` Function:**

    - This function should return an integer between `[1, 16]`. The maximum batch size supported at the moment is 16.
    - You can also choose the batch size dynamically.
    - This function is a **required** interface for your model class.

2. **Replace `generate_answer` with `batch_generate_answer`:**

    - Update your code to replace the `generate_answer` function with `batch_generate_answer`.
    - For more details on the `batch_generate_answer` interface, please refer to the inline documentation in [dummy_model.py](../models/dummy_model.py).

    ```python
    # Old Interface
    def generate_answer(self, query: str, search_results: List[Dict], query_time: str) -> str:
        ....
        ....
        return answer

    # New Interface
    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        ....
        ....

        return [answer1, answer2, ......, answerN]
    ```

    - The new function should return a list of answers (`List[str]`) instead of a single answer (`str`).
    - The simplest example of a valid submission with the new interface is as follows:

    ```python
    class DummyModel:
        def get_batch_size(self) -> int: 
            return 4

        def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
            queries = batch["query"]
            answers = ["i dont't know" for _ in queries]
            return answers
    ```    

### Backward Compatibility

To ensure a smooth transition, the evaluators will maintain backward compatibility with the `generate_answer` interface for a short period. However, we strongly recommend updating your code to use the `batch_generate_answer` interface to avoid any disruptions when support for the older interface is removed in the coming weeks.
