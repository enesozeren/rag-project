from models.utils import load_config
import re
import vllm

MAX_TOKENS = 10
TEMPERATURE = 0.0


class EvaluationModel:
    def __init__(self, config_path, llm, tokenizer):
        # Load configuration
        self.CONFIG = load_config(config_path)

        # Load the model and tokenizer
        # If the evaluation model is the same as the chat model, use the same model and tokenizer
        if (
            self.CONFIG["EvaluationModelParams"]["MODEL_PATH"]
            == self.CONFIG["ChatModelParams"]["MODEL_PATH"]
        ):
            self.llm = llm
            self.tokenizer = tokenizer
        # Otherwise, load the evaluation model and tokenizer
        else:
            self.llm = vllm.LLM(
                self.CONFIG["EvaluationModelParams"]["MODEL_PATH"],
                tensor_parallel_size=2,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True,
            )
            self.tokenizer = self.llm.get_tokenizer()

    def respond(self, query):
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )

        response = self.llm.generate(
            [formatted_prompt],
            vllm.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE),
            use_tqdm=False,  # you might consider setting this to True during local development
        )

        output_text = response[0].outputs[0].text

        # Use regular expression to find JSON part
        accuracy_json_match = re.search(r"({.*})", output_text)
        if accuracy_json_match:
            accuracy_json = accuracy_json_match.group(1)
        else:
            print("---EVAL MODEL: No JSON found in the generated text---")
            accuracy_json = "{'Accuracy': 'False'}"

        return accuracy_json
