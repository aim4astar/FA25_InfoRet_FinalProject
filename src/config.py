import os
import warnings
import logging

# Suppress all TensorFlow and system warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'                    #0=all, 1=info, 2=warnings, 3=errors
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')                           #Suppress all Python warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress stderr for TensorFlow
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)



# ----------------------------------------------------------------------------------------------------
# Extended configuration for arXiv queries, model selection, batch sizes, and evaluation settings.
# ----------------------------------------------------------------------------------------------------
ARXIV_QUERIES = [
    "cat:cs.IR", "cat:cs.LG", "cat:cs.AI", "cat:cs.CL", "cat:cs.CV",
    "cat:cs.NE", "cat:stat.ML", "cat:physics.comp-ph", "cat:math.NA"
]

DATASET_PATH = "arxiv-metadata-oai-snapshot.json"
MAX_RESULTS_PER_QUERY = 300
EMBEDDING_MODEL_NAMES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2", 
    "multiqa": "sentence-transformers/multi-qa-mpnet-base-dot-v1"
}
DEFAULT_EMBEDDING_MODEL = "minilm"
BATCH_SIZE = 32
TOP_K_RETRIEVAL = 10
PRECISION_AT_K_LIST = [1, 3, 5, 10]
USE_FAISS = True
RANDOM_SEED = 42
EVALUATION_SAMPLE_SIZE = 300                                            # Number of papers to use for evaluation
PLOT_COLORS = {"tfidf": "blue", "bm25": "green", "bert": "red", "mpnet": "purple", "multiqa": "orange"}
SAVE_EMBEDDINGS_VISUALIZATION = True
MAX_TOTAL_PAPERS = 2500
RESULTS_DIR = "search_results"