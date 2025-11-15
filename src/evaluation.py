import json
from typing import Dict

import matplotlib.pyplot as plt

from search_engine import SemanticSearchEngine


# ----------------------------------------------------------------------------------------------------
# Runs Precision@K evaluation for TF–IDF, BM25, and BERT models and prints the scores.
# Input: SemanticSearchEngine instance. Output: dict mapping model names to Precision@K scores.
# Also triggers saving of results to JSON and plotting Precision@K as a PNG image.
# ----------------------------------------------------------------------------------------------------
def runFullEvaluation(searchEngine: SemanticSearchEngine) -> Dict[str, Dict[int, float]]:
    results = {}
    for modelType in ["tfidf", "bm25", "bert"]:
        print(f"\nRunning evaluation for model: {modelType}")
        precisionScores = searchEngine.computePrecisionAtK(modelType=modelType)
        results[modelType] = {int(k): float(v) for k, v in precisionScores.items()}
        for k, score in precisionScores.items():
            print(f"Precision@{k}: {score:.4f}")

    saveEvaluationResultsToJson(results, "evaluation_results.json")
    plotPrecisionAtK(results, "precision_at_k.png")

    print('\nInfo: Saved evaluation results to "evaluation_results.json".')
    print('Info: Saved Precision@K plot to "precision_at_k.png".')

    return results


# ----------------------------------------------------------------------------------------------------
# Saves the evaluation results dictionary to a JSON file for reproducibility and later analysis.
# Input: nested precision results dict and output file path. Output: JSON file written to disk.
# ----------------------------------------------------------------------------------------------------
def saveEvaluationResultsToJson(results: Dict[str, Dict[int, float]],
                                outputPath: str) -> None:
    with open(outputPath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


# ----------------------------------------------------------------------------------------------------
# Plots Precision@K curves for each model and writes the figure to a PNG file for reporting.
# Input: nested precision results dict and output image path. Output: PNG line plot saved to disk.
# ----------------------------------------------------------------------------------------------------
def plotPrecisionAtK(results: Dict[str, Dict[int, float]],
                     outputPath: str) -> None:
    plt.figure()

    # Sort Ks so the x-axis is ordered
    modelNames = list(results.keys())
    for modelName in modelNames:
        kValues = sorted(results[modelName].keys())
        precisions = [results[modelName][k] for k in kValues]
        plt.plot(kValues, precisions, marker="o", label=modelName.upper())

    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Precision@K Comparison for TF–IDF, BM25, and BERT")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outputPath)
    plt.close()