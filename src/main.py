import argparse

from data_loader import buildCorpusFromArxiv
from search_engine import SemanticSearchEngine
from evaluation import runFullEvaluation
from config import TOP_K_RETRIEVAL


# ----------------------------------------------------------------------------------------------------
# Parses command-line arguments for evaluation and interactive search control.
# Input: none (reads sys.argv). Output: argparse.Namespace with configuration flags.
# ----------------------------------------------------------------------------------------------------
def parseArguments():
    parser = argparse.ArgumentParser(
        description="Semantic Similarity Search for arXiv Abstracts"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation and only run interactive search."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        choices=["tfidf", "bm25", "bert"],
        help="Model type to use for interactive search."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=TOP_K_RETRIEVAL,
        help="Number of results to return for interactive query."
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------
# Prints ranked search results in a readable format with basic metadata and score.
# Input: list of (paper dict, score) and max result count. Output: formatted console output.
# ----------------------------------------------------------------------------------------------------
def prettyPrintResults(results, maxCount: int):
    for idx, (paper, score) in enumerate(results[:maxCount]):
        print(f"\nRank {idx + 1} | Score: {score:.4f}")
        print(f"ID: {paper['id']}")
        print(f"Category: {paper['category']}")
        print(f"Title: {paper['title'].strip()}")
        print(f"Abstract: {paper['abstract'][:300]}...")


# ----------------------------------------------------------------------------------------------------
# Coordinates loading data, building indices, running evaluation, and starting interactive search.
# Input: none (uses command-line args). Output: console logs and interactive query loop.
# ----------------------------------------------------------------------------------------------------
def main():
    args = parseArguments()

    print("Step 1: Loading corpus from arXiv...")
    papers = buildCorpusFromArxiv()
    print(f"Info: Loaded {len(papers)} unique papers from arXiv.")

    print("Step 2: Building representations and indices...")
    searchEngine = SemanticSearchEngine(papers=papers)
    searchEngine.buildAllIndices()
    print("Info: Indices built successfully.")

    if not args.no_eval:
        print("\nStep 3: Running model evaluation (Precision@K)...")
        runFullEvaluation(searchEngine)

    print("\nStep 4: Interactive semantic search (type 'exit' to quit).")
    while True:
        userQuery = input("\nEnter an abstract or description: ").strip()
        if userQuery.lower() in ["exit", "quit"]:
            print("Info: Exiting interactive search.")
            break

        results = searchEngine.searchByAbstract(
            queryAbstract=userQuery,
            modelType=args.model,
            topK=args.topk
        )
        prettyPrintResults(results, args.topk)


if __name__ == "__main__":
    main()