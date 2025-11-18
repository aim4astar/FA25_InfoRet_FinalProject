from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from text_representation import TfIdfRepresentation, Bm25Representation, BertEmbeddingRepresentation
from config import TOP_K_RETRIEVAL, PRECISION_AT_K_LIST, USE_FAISS, EVALUATION_SAMPLE_SIZE, EMBEDDING_MODEL_NAMES


class SemanticSearchEngine:
    def __init__(self, papers: List[Dict]):
        self.papers = papers
        self.corpusTexts = [paper["abstract"] for paper in papers]

        self.tfIdfRep = TfIdfRepresentation()
        self.bm25Rep = Bm25Representation()
        self.bertReps = {
            model_key: BertEmbeddingRepresentation(model_key) 
            for model_key in EMBEDDING_MODEL_NAMES.keys()
        }

        self.tfIdfMatrix = None
        self.bm25Ready = False
        self.bertEmbeddings = {}
        self.faissIndices = {}

    # ------------------------------------------------------------------------------------------------
    # Builds TF-IDF, BM25, and BERT indices for all models and optionally initializes FAISS indices.
    # Input: none. Output: populated internal structures ready for search and evaluation.
    # ------------------------------------------------------------------------------------------------
    def buildAllIndices(self) -> None:
        print("Building TF-IDF index...")
        self.tfIdfRep.fitDocuments(self.corpusTexts)
        self.tfIdfMatrix = self.tfIdfRep.documentMatrix

        print("Building BM25 index...")
        self.bm25Rep.fitDocuments(self.corpusTexts)
        self.bm25Ready = True

        # Build embeddings for all BERT models
        for modelKey, bertRep in self.bertReps.items():
            print(f"Building {modelKey.upper()} embeddings (this may take a while for large corpora)...")
            show_progress = (modelKey == list(self.bertReps.keys())[0])
            self.bertEmbeddings[modelKey] = bertRep.encodeDocuments(
                self.corpusTexts, 
                show_progress_bar=show_progress
            )
            self.bertEmbeddings[modelKey] = self._l2Normalize(self.bertEmbeddings[modelKey])

            if USE_FAISS:
                print(f"Building FAISS index for {modelKey.upper()}...")
                dim = self.bertEmbeddings[modelKey].shape[1]
                self.faissIndices[modelKey] = faiss.IndexFlatIP(dim)
                self.faissIndices[modelKey].add(self.bertEmbeddings[modelKey].astype(np.float32))
                print(f"FAISS index built for {modelKey.upper()} with {self.faissIndices[modelKey].ntotal} vectors")

    # ------------------------------------------------------------------------------------------------
    # Normalizes embedding vectors to unit length so inner product corresponds to cosine similarity.
    # Input: 2D numpy array of vectors. Output: normalized 2D numpy array with same shape.
    # ------------------------------------------------------------------------------------------------
    def _l2Normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix / norms

    # ------------------------------------------------------------------------------------------------
    # Routes a query to TF-IDF, BM25, or BERT search and returns top-K ranked documents.
    # Input: query text, model type string, and K. Output: list of (paper dict, score) tuples.
    # ------------------------------------------------------------------------------------------------
    def searchByAbstract(self, queryAbstract: str,
                         modelType: str = "bert",
                         topK: int = TOP_K_RETRIEVAL) -> List[Tuple[Dict, float]]:
        if modelType == "tfidf":
            return self._searchWithTfIdf(queryAbstract, topK)
        elif modelType == "bm25":
            return self._searchWithBm25(queryAbstract, topK)
        elif modelType in self.bertReps:
            return self._searchWithBert(queryAbstract, modelType, topK)
        else:
            # Default to minilm if model type not specified
            return self._searchWithBert(queryAbstract, "minilm", topK)

    # ------------------------------------------------------------------------------------------------
    # Performs TF-IDF based cosine similarity search between the query and document matrix.
    # Input: query text and K. Output: list of top-K papers with their similarity scores.
    # ------------------------------------------------------------------------------------------------
    def _searchWithTfIdf(self, queryAbstract: str,
                         topK: int) -> List[Tuple[Dict, float]]:
        queryVec = self.tfIdfRep.encodeQuery(queryAbstract)
        similarities = cosine_similarity(queryVec, self.tfIdfMatrix).flatten()
        topIndices = np.argsort(similarities)[::-1][:topK]
        results = [(self.papers[i], float(similarities[i])) for i in topIndices]
        return results

    # ------------------------------------------------------------------------------------------------
    # Performs BM25 based lexical relevance ranking over the corpus for the given query text.
    # Input: query text and K. Output: list of top-K papers with their BM25 scores.
    # ------------------------------------------------------------------------------------------------
    def _searchWithBm25(self, queryAbstract: str,
                        topK: int) -> List[Tuple[Dict, float]]:
        scores = np.array(self.bm25Rep.scoreQuery(queryAbstract))
        topIndices = np.argsort(scores)[::-1][:topK]
        results = [(self.papers[i], float(scores[i])) for i in topIndices]
        return results

    # ------------------------------------------------------------------------------------------------
    # Performs BERT embedding based search using FAISS if available, otherwise dot-product search.
    # Input: query text, model key, and K. Output: list of top-K semantically similar papers with scores.
    # ------------------------------------------------------------------------------------------------
    def _searchWithBert(self, queryAbstract: str,
                        modelKey: str,
                        topK: int) -> List[Tuple[Dict, float]]:
        if modelKey not in self.bertReps:
            raise ValueError(f"Model {modelKey} not found. Available models: {list(self.bertReps.keys())}")
            
        queryVec = self.bertReps[modelKey].encodeQuery(queryAbstract)
        queryVec = self._l2Normalize(queryVec.reshape(1, -1))

        if modelKey in self.faissIndices and self.faissIndices[modelKey] is not None:
            distances, indices = self.faissIndices[modelKey].search(
                queryVec.astype(np.float32), topK
            )
            indices = indices[0]
            distances = distances[0]
        else:
            sims = np.dot(self.bertEmbeddings[modelKey], queryVec.T).flatten()
            indices = np.argsort(sims)[::-1][:topK]
            distances = sims[indices]

        results = [(self.papers[int(i)], float(distances[pos]))
                   for pos, i in enumerate(indices)]
        return results

    # ------------------------------------------------------------------------------------------------
    # Computes Precision@K using category labels by treating each paper as a query document.
    # Input: model type, list of K values, and maximum number of queries. Output: dict of K to score.
    # ------------------------------------------------------------------------------------------------
    def computePrecisionAtK(self, modelType: str = "bert",
                            kList: List[int] = None,
                            maxQueries: int = None) -> Dict[int, float]:
        if kList is None:
            kList = PRECISION_AT_K_LIST
        if maxQueries is None:
            maxQueries = min(EVALUATION_SAMPLE_SIZE, len(self.papers))

        numQueries = min(maxQueries, len(self.papers))
        queryIndices = list(range(numQueries))

        print(f"Evaluating {modelType.upper()} on {numQueries} queries...")

        precisionSums = {k: 0.0 for k in kList}

        for idx, query_idx in enumerate(queryIndices):
            progress_points = [0, numQueries//4, numQueries//2, 3*numQueries//4, numQueries-1]
            if idx in progress_points:
                percentage = (idx + 1) / numQueries * 100
                print(f"   Progress: {idx + 1}/{numQueries} queries evaluated ({percentage:.0f}%)...")
                
            queryPaper = self.papers[query_idx]
            queryAbstract = queryPaper["abstract"]
            queryCategory = queryPaper["category"]

            results = self.searchByAbstract(
                queryAbstract=queryAbstract,
                modelType=modelType,
                topK=max(kList) + 1                                                             #+1 to account for self-match exclusion
            )

            # Remove the paper itself from results
            filteredResults = [r for r in results if r[0]["id"] != queryPaper["id"]]

            for k in kList:
                topKResults = filteredResults[:k]
                if not topKResults:
                    continue
                matches = sum(1 for paper, _ in topKResults
                              if paper["category"] == queryCategory)
                precisionK = matches / k
                precisionSums[k] += precisionK

        print(f"   Progress: {numQueries}/{numQueries} queries evaluated (100%) - COMPLETED")
        
        precisionAtK = {k: precisionSums[k] / numQueries for k in kList}
        return precisionAtK