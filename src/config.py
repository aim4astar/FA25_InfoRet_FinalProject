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
arxivQueries = [
    "cat:cs.IR", "cat:cs.LG", "cat:cs.AI", "cat:cs.CL", "cat:cs.CV",
    "cat:cs.NE", "cat:stat.ML", "cat:physics.comp-ph", "cat:math.NA"
]

datasetPath = "arxiv-metadata-oai-snapshot.json"
maxResultsPerQuery = 300
embeddingModelNames = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "multiqa": "sentence-transformers/multi-qa-mpnet-base-dot-v1"
}
defaultEmbeddingModel = "minilm"
batchSize = 32
topKRetrieval = 10
precisionAtKList = [1, 3, 5, 10]
useFaiss = True
randomSeed = 42
evaluationSampleSize = 300
saveEmbeddingsVisualization = True
maxTotalPapers = 2500
resultsDir = "search_results"
plotColors = {
    "tfidf": "blue",
    "bm25": "green",
    "bert": "red",
    "mpnet": "purple",
    "multiqa": "orange",
    "lsa": "brown",
    "lda": "cyan"
}
