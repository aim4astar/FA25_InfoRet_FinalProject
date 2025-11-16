from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

from config import EMBEDDING_MODEL_NAME, BATCH_SIZE


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
    def __init__(self, modelName: str = EMBEDDING_MODEL_NAME):
        self.device = self._selectDevice()
        self.model = SentenceTransformer(modelName, device=self.device)

    # ------------------------------------------------------------------------------------------------
    # Chooses GPU if available, otherwise CPU, and prints a short info message for the user.
    # Input: none. Output: device string ("cuda" or "cpu") used by the embedding model.
    # ------------------------------------------------------------------------------------------------
    def _selectDevice(self) -> str:
        if torch.cuda.is_available():
            print("Info: GPU detected, using CUDA for embedding computations.")
            return "cuda"
        else:
            print("Info: GPU not detected, using CPU for embedding computations.")
            return "cpu"

    # ------------------------------------------------------------------------------------------------
    # Encodes a batch of texts into dense embeddings using the SentenceTransformer model.
    # Input: list of strings and optional batch size. Output: 2D numpy array of embeddings.
    # ------------------------------------------------------------------------------------------------
    def encodeDocuments(self, corpusTexts: List[str],
                        batchSize: int = BATCH_SIZE) -> np.ndarray:
        embeddings = self.model.encode(
            corpusTexts,
            batch_size=batchSize,
            convert_to_numpy=True,
            show_progress_bar=True
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