import argparse
from typing import List, Tuple, Dict
from datetime import datetime
import os
import json

from data_loader import buildCorpusFromArxiv
from search_engine import SemanticSearchEngine
from evaluation import runFullEvaluation, ensureResultsDirectory
from config import TOP_K_RETRIEVAL, RESULTS_DIR


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
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Maximum number of papers to use (for testing)."
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------
# Prints ranked search results in a readable format with basic metadata and score.
# Input: list of (paper dict, score) and max result count. Output: formatted console output.
# ----------------------------------------------------------------------------------------------------
def prettyPrintResults(results, maxCount: int):
    print(f"\n Found {len(results)} results. Showing top {maxCount}:")
    print("=" * 80)
    for idx, (paper, score) in enumerate(results[:maxCount]):
        print(f"\nRank {idx + 1} | Score: {score:.4f} | Category: {paper['category']}")
        print(f"ID: {paper['id']}")
        print(f"Title: {paper['title'].strip()}")
        print(f"Authors: {', '.join(paper.get('authors', ['Unknown'])[:3])}{'...' if len(paper.get('authors', [])) > 3 else ''}")
        print(f"Abstract: {paper['abstract'][:250]}...")
        print("-" * 80)


# ----------------------------------------------------------------------------------------------------
# Saves search query and results to a log file in the results directory for later analysis.
# Input: query string, results list, model type, and topK value. Output: appends to search log.
# ----------------------------------------------------------------------------------------------------
def saveSearchToLog(query: str, results: List[Tuple[Dict, float]], 
                   modelType: str, topK: int) -> None:
    results_dir = ensureResultsDirectory()
    log_file = os.path.join(results_dir, "search_log.json")
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "model_type": modelType,
        "top_k": topK,
        "results": [
            {
                "rank": idx + 1,
                "paper_id": paper["id"],
                "title": paper["title"],
                "category": paper["category"],
                "score": float(score)
            }
            for idx, (paper, score) in enumerate(results[:topK])
        ]
    }
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


# ----------------------------------------------------------------------------------------------------
# Coordinates loading data, building indices, running evaluation, and starting interactive search.
# Input: none (uses command-line args). Output: console logs and interactive query loop.
# ----------------------------------------------------------------------------------------------------
def main():
    args = parseArguments()
    start_time = datetime.now()
    
    # Ensure results directory exists from the start
    ensureResultsDirectory()
    
    print("Starting Semantic Similarity Search System")
    print("=" * 50)

    print("Step 1: Loading corpus from arXiv...")    
    corpus_data = buildCorpusFromArxiv()
    papers = corpus_data['papers']
    total_fetched = corpus_data['total_fetched']
    duplicates_removed = corpus_data['duplicates_removed']

    print(f"Loaded {len(papers)} unique papers from arXiv.")
    
    # Limit papers if specified for testing
    if args.max_papers and args.max_papers < len(papers):
        papers = papers[:args.max_papers]
        print(f"Info: Limited to {len(papers)} papers for testing.")
    
    print(f"Loaded {len(papers)} unique papers from arXiv.")

    print("\nStep 2: Building representations and indices...")
    searchEngine = SemanticSearchEngine(papers=papers)
    searchEngine.buildAllIndices()
    print("Indices built successfully.")

    if not args.no_eval:
        print("\nStep 3: Running model evaluation (Precision@K)...")
        runFullEvaluation(searchEngine, total_fetched, duplicates_removed)
        print("Evaluation completed.")

    elapsed = datetime.now() - start_time
    print(f"\nTotal setup time: {elapsed.total_seconds():.2f} seconds")

    print("\nInteractive semantic search (type 'exit' to quit).")
    print("Try queries like: 'machine learning transformers attention mechanism'")
    print("   or paste a full abstract from a research paper.")
    
    while True:
        userQuery = input("\n Enter an abstract or description: ").strip()
        if userQuery.lower() in ["exit", "quit"]:
            print("Exiting interactive search. Goodbye!")
            break
        if not userQuery:
            print("Please enter a query.")
            continue

        try:
            results = searchEngine.searchByAbstract(
                queryAbstract=userQuery,
                modelType=args.model,
                topK=args.topk
            )
            prettyPrintResults(results, args.topk)

            saveSearchToLog(userQuery, results, args.model, args.topk)
            print(f"Info: Search saved to log file in '{RESULTS_DIR}' directory.")
            
        except Exception as e:
            print(f" Error during search: {e}")

if __name__ == "__main__":
    main()