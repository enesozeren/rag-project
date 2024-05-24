from models.vanilla_llama_baseline import InstructModel

# Initialize the DummyModel class
dummy_model = InstructModel()
output = dummy_model.batch_generate_answer(
    "What is the capital of France?", [], "2022-01-01T00:00:00Z"
)
print(output)
