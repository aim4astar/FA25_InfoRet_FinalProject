import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'huggingface_cache')
import argparse
from typing import List, Tuple, Dict
from datetime import datetime
import json

from data_loader import buildCorpusFromArxiv, buildCorpusFromJson
from search_engine import SemanticSearchEngine
from evaluation import runFullEvaluation, ensureResultsDirectory
from config import TOP_K_RETRIEVAL, RESULTS_DIR, EMBEDDING_MODEL_NAMES


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
        default="minilm",
        choices=["tfidf", "bm25"] + list(EMBEDDING_MODEL_NAMES.keys()),
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
    parser.add_argument(
        "--use-arxiv-api",
        action="store_true",
        help="Use the live arXiv API instead of the local JSON snapshot."
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
# Saves search query and results to a log file and generates visualization plots.
# Input: query string, search engine instance, model type, and topK value. 
# Output: appends to search log and saves PNG plots.
# ----------------------------------------------------------------------------------------------------
def saveSearchToLog(query: str, 
                   searchEngine,
                   modelType: str, 
                   topK: int):
    results_dir = ensureResultsDirectory()
    log_file = os.path.join(results_dir, "search_log.json")
    
    # Get results for the specified model
    single_model_results = searchEngine.searchByAbstract(
        queryAbstract=query, modelType=modelType, topK=topK
    )
    
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
            for idx, (paper, score) in enumerate(single_model_results[:topK])
        ]
    }
    
    # Save to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Generate single model plot
    single_plot_filename = f"search_results_{modelType}.png"
    single_plot_path = os.path.join(results_dir, single_plot_filename)
    
    from evaluation import plotInteractiveSearchResults
    plotInteractiveSearchResults(query, single_model_results[:topK], modelType, single_plot_path)
    
    # Generate multi-model comparison plots
    multi_plot_filename = "search_comparison_all_models.png"
    multi_plot_path = os.path.join(results_dir, multi_plot_filename)
    
    heatmap_filename = "search_comparison_heatmap.png"
    heatmap_path = os.path.join(results_dir, heatmap_filename)
    
    # Generate Excel export
    excel_filename = "search_results_comparison.xlsx"
    excel_path = os.path.join(results_dir, excel_filename)
    
    # Get results from all BERT models for comparison
    from config import EMBEDDING_MODEL_NAMES
    all_model_results = {}
    
    # Include the current model
    all_model_results[modelType] = single_model_results[:topK]
    
    # Get results from other BERT models
    for bert_model in EMBEDDING_MODEL_NAMES.keys():
        if bert_model != modelType:
            try:
                model_results = searchEngine.searchByAbstract(
                    queryAbstract=query, modelType=bert_model, topK=topK
                )
                all_model_results[bert_model] = model_results[:topK]
            except Exception as e:
                print(f"Warning: Could not get results for {bert_model}: {e}")
    
    from evaluation import plotMultiModelSearchResults, plotMultiModelHeatmap, exportSearchResultsToExcel
    plotMultiModelSearchResults(query, all_model_results, multi_plot_path)
    plotMultiModelHeatmap(query, all_model_results, heatmap_path)
    exportSearchResultsToExcel(query, all_model_results, excel_path)
    
    return single_plot_path, multi_plot_path, heatmap_path, excel_path


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

    if args.use_arxiv_api:
        print("Step 1: Loading corpus from arXiv API...")
        corpus_data = buildCorpusFromArxiv(maxPapers=args.max_papers)
    else:
        print("Step 1: Loading corpus from local JSON dataset...")
        corpus_data = buildCorpusFromJson(maxPapers=args.max_papers)

    papers = corpus_data["papers"]
    total_fetched = corpus_data["total_fetched"]
    duplicates_removed = corpus_data["duplicates_removed"]

    print(f"Loaded {len(papers)} unique papers into the corpus.")

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
            if not hasattr(searchEngine, 'searchByAbstract'):
                print(f"Error: searchEngine is {type(searchEngine)}, expected SemanticSearchEngine")
                print("Trying to reinitialize search engine...")
                searchEngine = SemanticSearchEngine(papers=papers)
                searchEngine.buildAllIndices()
            
            results = searchEngine.searchByAbstract(
                queryAbstract=userQuery,
                modelType=args.model,
                topK=args.topk
            )
            prettyPrintResults(results, args.topk)
            single_plot_path, multi_plot_path, heatmap_path, excel_path = saveSearchToLog(
                userQuery, searchEngine, args.model, args.topk
            )
            print(f"Info: Search saved to log.")
            print(f"Info: Single model results plot saved to '{single_plot_path}'.")
            print(f"Info: Multi-model comparison plot saved to '{multi_plot_path}'.")
            print(f"Info: Multi-model heatmap saved to '{heatmap_path}'.")
            print(f"Info: Detailed results exported to Excel: '{excel_path}'.")
            
        except Exception as e:
            print(f" Error during search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()