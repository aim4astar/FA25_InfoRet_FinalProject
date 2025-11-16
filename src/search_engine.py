from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from text_representation import TfIdfRepresentation, Bm25Representation, BertEmbeddingRepresentation
from config import TOP_K_RETRIEVAL, PRECISION_AT_K_LIST, USE_FAISS, EVALUATION_SAMPLE_SIZE


class SemanticSearchEngine:
    def __init__(self, papers: List[Dict]):
        self.papers = papers
        self.corpusTexts = [paper["abstract"] for paper in papers]

        self.tfIdfRep = TfIdfRepresentation()
        self.bm25Rep = Bm25Representation()
        self.bertRep = BertEmbeddingRepresentation()

        self.tfIdfMatrix = None
        self.bm25Ready = False
        self.bertEmbeddings = None
        self.faissIndex = None

    # ------------------------------------------------------------------------------------------------
    # Builds TF–IDF, BM25, and BERT indices and optionally initializes a FAISS index for speed.
    # Input: none. Output: populated internal structures ready for search and evaluation.
    # ------------------------------------------------------------------------------------------------
    def buildAllIndices(self) -> None:
        print("Building TF-IDF index...")
        self.tfIdfRep.fitDocuments(self.corpusTexts)
        self.tfIdfMatrix = self.tfIdfRep.documentMatrix

        print("Building BM25 index...")
        self.bm25Rep.fitDocuments(self.corpusTexts)
        self.bm25Ready = True

        print("Building BERT embeddings (this may take a while for large corpora)...")
        self.bertEmbeddings = self.bertRep.encodeDocuments(self.corpusTexts)
        self.bertEmbeddings = self._l2Normalize(self.bertEmbeddings)

        if USE_FAISS:
            print("Building FAISS index for fast similarity search...")
            dim = self.bertEmbeddings.shape[1]
            self.faissIndex = faiss.IndexFlatIP(dim)
            self.faissIndex.add(self.bertEmbeddings.astype(np.float32))
            print(f"FAISS index built with {self.faissIndex.ntotal} vectors")

    # ------------------------------------------------------------------------------------------------
    # Normalizes embedding vectors to unit length so inner product corresponds to cosine similarity.
    # Input: 2D numpy array of vectors. Output: normalized 2D numpy array with same shape.
    # ------------------------------------------------------------------------------------------------
    def _l2Normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix / norms

    # ------------------------------------------------------------------------------------------------
    # Routes a query to TF–IDF, BM25, or BERT search and returns top-K ranked documents.
    # Input: query text, model type string, and K. Output: list of (paper dict, score) tuples.
    # ------------------------------------------------------------------------------------------------
    def searchByAbstract(self, queryAbstract: str,
                         modelType: str = "bert",
                         topK: int = TOP_K_RETRIEVAL) -> List[Tuple[Dict, float]]:
        if modelType == "tfidf":
            return self._searchWithTfIdf(queryAbstract, topK)
        elif modelType == "bm25":
            return self._searchWithBm25(queryAbstract, topK)
        else:
            return self._searchWithBert(queryAbstract, topK)

    # ------------------------------------------------------------------------------------------------
    # Performs TF–IDF based cosine similarity search between the query and document matrix.
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
    # Input: query text and K. Output: list of top-K semantically similar papers with scores.
    # ------------------------------------------------------------------------------------------------
    def _searchWithBert(self, queryAbstract: str,
                        topK: int) -> List[Tuple[Dict, float]]:
        queryVec = self.bertRep.encodeQuery(queryAbstract)
        queryVec = self._l2Normalize(queryVec.reshape(1, -1))

        if self.faissIndex is not None:
            distances, indices = self.faissIndex.search(
                queryVec.astype(np.float32), topK
            )
            indices = indices[0]
            distances = distances[0]
        else:
            sims = np.dot(self.bertEmbeddings, queryVec.T).flatten()
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

        for idx in queryIndices:
            if idx % 50 == 0:
                print(f"   Progress: {idx}/{numQueries} queries evaluated...")
                
            queryPaper = self.papers[idx]
            queryAbstract = queryPaper["abstract"]
            queryCategory = queryPaper["category"]

            results = self.searchByAbstract(
                queryAbstract=queryAbstract,
                modelType=modelType,
                topK=max(kList) + 1                                                             # +1 to account for self-match exclusion
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

        precisionAtK = {k: precisionSums[k] / numQueries for k in kList}
        return precisionAtK