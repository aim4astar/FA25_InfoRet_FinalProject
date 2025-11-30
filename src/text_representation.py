import os
os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from typing import List
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from config import embeddingModelNames, defaultEmbeddingModel, batchSize


class TfIdfRepresentation:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000
        )
        self.documentMatrix = None

    # ------------------------------------------------------------------------------------------------
    # Fits a TF-IDF vectorizer on corpus texts and stores the resulting sparse document matrix.
    # Input: list of document strings. Output: internal sparse matrix for later similarity queries.
    # ------------------------------------------------------------------------------------------------
    def fitDocuments(self, corpusTexts: List[str]) -> None:
        self.documentMatrix = self.vectorizer.fit_transform(corpusTexts)

    # ------------------------------------------------------------------------------------------------
    # Encodes a query string into the TF-IDF feature space consistent with the fitted vocabulary.
    # Input: query text string. Output: one-row sparse TF-IDF vector ready for similarity scoring.
    # ------------------------------------------------------------------------------------------------
    def encodeQuery(self, queryText: str):
        return self.vectorizer.transform([queryText])


class Bm25Representation:
    def __init__(self):
        self.tokenizedDocs = None
        self.bm25 = None

    # ------------------------------------------------------------------------------------------------
    # Tokenizes corpus texts and builds a BM25 index for probabilistic lexical relevance scoring.
    # Input: list of document strings. Output: internal BM25Okapi model bound to the tokenized docs.
    # ------------------------------------------------------------------------------------------------
    def fitDocuments(self, corpusTexts: List[str]) -> None:
        self.tokenizedDocs = [doc.split() for doc in corpusTexts]
        self.bm25 = BM25Okapi(self.tokenizedDocs)

    # ------------------------------------------------------------------------------------------------
    # Scores a query against the BM25 index so that each document receives a lexical relevance score.
    # Input: query string. Output: list of float scores aligned with the corpus document indices.
    # ------------------------------------------------------------------------------------------------
    def scoreQuery(self, queryText: str) -> List[float]:
        queryTokens = queryText.split()
        scores = self.bm25.get_scores(queryTokens)
        return scores


class BertEmbeddingRepresentation:
    def __init__(self, modelName: str = None, verbose: bool = True):
        if modelName is None:
            modelName = embeddingModelNames[defaultEmbeddingModel]
        elif modelName in embeddingModelNames:
            modelName = embeddingModelNames[modelName]
            
        # Print cache location before loading model
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(repo_id=modelName, local_files_only=False)
        print(f"Model '{modelName}' cache location: {cache_dir}")
            
        self.device = self._selectDevice(verbose)
        self.model = SentenceTransformer(modelName, device=self.device)
        self.modelName = modelName

    # ------------------------------------------------------------------------------------------------
    # Chooses GPU if available, otherwise CPU, and prints a short info message for the user.
    # Input: verbose flag to control logging. Output: device string ("cuda" or "cpu").
    # ------------------------------------------------------------------------------------------------
    def _selectDevice(self, verbose: bool = True) -> str:
        if torch.cuda.is_available():
            if verbose:
                print("Info: GPU detected, using CUDA for embedding computations.")
            return "cuda"
        else:
            if verbose:
                print("Info: GPU not detected, using CPU for embedding computations.")
            return "cpu"

    # ------------------------------------------------------------------------------------------------
    # Encodes a batch of texts into dense embeddings using the SentenceTransformer model.
    # Input: list of strings and optional batch size. Output: 2D numpy array of embeddings.
    # ------------------------------------------------------------------------------------------------
    def encodeDocuments(self, corpusTexts: List[str],
                    batchSizeArg: int = batchSize,
                    showProgressBar: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            corpusTexts,
            batch_size=batchSizeArg,
            convert_to_numpy=True,
            show_progress_bar=showProgressBar
        )
        return embeddings

    # ------------------------------------------------------------------------------------------------
    # Encodes a single query string into a dense embedding compatible with document embeddings.
    # Input: query string. Output: 1D numpy vector representing the semantic content of the query.
    # ------------------------------------------------------------------------------------------------
    def encodeQuery(self, queryText: str) -> np.ndarray:
        embedding = self.model.encode(
            [queryText],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding[0]
        
# ------------------------------------------------------------------------------------------------
# Latent Semantic Analysis (LSA) using TF-IDF + TruncatedSVD.
# Documents and queries are represented in a low-dimensional
# latent topic space, then compared via cosine similarity.
# ------------------------------------------------------------------------------------------------
class LsaRepresentation:
    def __init__(self, nComponents: int = 100, randomState: int = 42):        
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000
        )
        self.svd = TruncatedSVD(n_components=nComponents, random_state=randomState)
        self.normalizer = Normalizer(copy=False)
        self.pipeline = make_pipeline(self.vectorizer, self.svd, self.normalizer)
        self.documentMatrix = None
    
    # ------------------------------------------------------------------------------------------------
    # Fit TF-IDF + SVD pipeline and store the document-topic matrix.
    # ------------------------------------------------------------------------------------------------
    def fitDocuments(self, corpusTexts: List[str]) -> None:
        self.documentMatrix = self.pipeline.fit_transform(corpusTexts)
    
    # ------------------------------------------------------------------------------------------------
    # Encode a query into the same latent topic space. Returns a 2D array (1 * nComponents).
    # ------------------------------------------------------------------------------------------------
    def encodeQuery(self, queryText: str) -> np.ndarray:
        return self.pipeline.transform([queryText])
        

# ------------------------------------------------------------------------------------------------
# Latent Dirichlet Allocation (LDA) using CountVectorizer.
# Each document/query is represented as a topic-distribution vector.
# ------------------------------------------------------------------------------------------------
class LdaRepresentation:
    def __init__(self, nTopics: int = 50, randomState: int = 42):
        self.countVectorizer = CountVectorizer(
            stop_words="english",
            max_features=20000
        )
        self.ldaModel = LatentDirichletAllocation(
            n_components=nTopics,
            random_state=randomState,
            learning_method="batch"
        )
        self.documentMatrix = None
    
    # ------------------------------------------------------------------------------------------------
    # Fit CountVectorizer + LDA and store document-topic distributions.
    # ------------------------------------------------------------------------------------------------
    def fitDocuments(self, corpusTexts: List[str]) -> None:
        countMatrix = self.countVectorizer.fit_transform(corpusTexts)
        self.documentMatrix = self.ldaModel.fit_transform(countMatrix)
    
    # ------------------------------------------------------------------------------------------------
    # Encode a query into a topic-distribution vector (1 * nTopics).
    # ------------------------------------------------------------------------------------------------
    def encodeQuery(self, queryText: str) -> np.ndarray:
        countVec = self.countVectorizer.transform([queryText])
        topicDist = self.ldaModel.transform(countVec)
        return topicDist
        
        