# ----------------------------------------------------------------------------------------------------
# Basic configuration for arXiv queries, model selection, batch sizes, and evaluation settings.
# ----------------------------------------------------------------------------------------------------
ARXIV_QUERIES = [
    "cat:cs.IR",
    "cat:cs.LG",
    "cat:cs.AI"
]

MAX_RESULTS_PER_QUERY = 150

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32

TOP_K_RETRIEVAL = 10

PRECISION_AT_K_LIST = [1, 3, 5, 10]

USE_FAISS = True

RANDOM_SEED = 42
