import unittest

from data_loader import preprocessText
from text_representation import BertEmbeddingRepresentation
from search_engine import SemanticSearchEngine


# ----------------------------------------------------------------------------------------------------
# Groups unit tests for preprocessing, embeddings, and basic search to validate the pipeline.
# Input: managed by unittest framework. Output: pass or fail status for each test case.
# ----------------------------------------------------------------------------------------------------
class SemanticSearchTests(unittest.TestCase):

    # ------------------------------------------------------------------------------------------------
    # Verifies that preprocessing lowercases text and collapses whitespace correctly.
    # Input: synthetic raw text. Output: assertion that cleaned text matches expectations.
    # ------------------------------------------------------------------------------------------------
    def testPreprocessTextBasic(self):
        rawText = "  This   IS   a   Test\t\n"
        processedText = preprocessText(rawText)
        self.assertEqual(processedText, "this is a test")

    # ------------------------------------------------------------------------------------------------
    # Confirms that the embedding representation selects either CPU or GPU as a valid device.
    # Input: new BertEmbeddingRepresentation instance. Output: assertion on device string.
    # ------------------------------------------------------------------------------------------------
    def testBertDeviceSelection(self):
        bertRep = BertEmbeddingRepresentation()
        self.assertIn(bertRep.device, ["cuda", "cpu"])

    # ------------------------------------------------------------------------------------------------
    # Checks that encoding multiple texts returns a 2D array with rows equal to input size.
    # Input: list of small sentences. Output: assertion on shape of returned embedding matrix.
    # ------------------------------------------------------------------------------------------------
    def testBertEncodeDocumentsShape(self):
        bertRep = BertEmbeddingRepresentation()
        texts = ["first document", "second document", "third document"]
        embeddings = bertRep.encodeDocuments(texts, batchSize=2)
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertGreater(embeddings.shape[1], 0)

    # ------------------------------------------------------------------------------------------------
    # Builds a tiny synthetic corpus and ensures that BERT search returns at least one result.
    # Input: hard-coded mini corpus. Output: assertions on result count and score type.
    # ------------------------------------------------------------------------------------------------
    def testSearchEngineWithTinyCorpus(self):
        papers = [
            {
                "id": "paper1",
                "title": "Neural Networks for Image Classification",
                "abstract": "deep learning models for image classification using convolutional networks",
                "category": "cs.CV",
                "published": None,
            },
            {
                "id": "paper2",
                "title": "Information Retrieval with BM25",
                "abstract": "retrieval models including tf idf and bm25 for ranking documents",
                "category": "cs.IR",
                "published": None,
            },
            {
                "id": "paper3",
                "title": "Transformers for Natural Language Processing",
                "abstract": "attention based transformer models for language tasks",
                "category": "cs.CL",
                "published": None,
            },
        ]

        searchEngine = SemanticSearchEngine(papers=papers)
        searchEngine.buildAllIndices()

        queryText = "deep learning for images"
        results = searchEngine.searchByAbstract(queryText, modelType="bert", topK=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("title", results[0][0])
        self.assertIsInstance(results[0][1], float)


if __name__ == "__main__":
    unittest.main()