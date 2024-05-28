# from models.dummy_model import DummyModel
# isort: skip_file

# UserModel = DummyModel

# Uncomment the lines below to use the Vanilla LLAMA baseline
# from models.vanilla_llama_baseline import InstructModel

# UserModel = InstructModel


# Uncomment the lines below to use the RAG LLAMA baseline
from models.rag_llama_baseline import RAGModel

UserModel = RAGModel
