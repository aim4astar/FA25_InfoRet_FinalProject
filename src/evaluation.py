import json
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from search_engine import SemanticSearchEngine
from config import PLOT_COLORS, SAVE_EMBEDDINGS_VISUALIZATION, RESULTS_DIR


# ----------------------------------------------------------------------------------------------------
# Creates results directory if it doesn't exist to ensure all outputs are organized in one place.
# Input: none. Output: creates directory structure and returns full path to results folder.
# ----------------------------------------------------------------------------------------------------
def ensureResultsDirectory() -> str:
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Info: Created results directory: {RESULTS_DIR}")
    return RESULTS_DIR


# ----------------------------------------------------------------------------------------------------
# Runs Precision@K evaluation for TF–IDF, BM25, and BERT models and prints the scores.
# Input: SemanticSearchEngine instance. Output: dict mapping model names to Precision@K scores.
# Also triggers saving of results to JSON and plotting Precision@K as a PNG image.
# ----------------------------------------------------------------------------------------------------
def runFullEvaluation(searchEngine: SemanticSearchEngine, 
                     total_fetched: int, duplicates_removed: int) -> Dict[str, Dict[int, float]]:
    results_dir = ensureResultsDirectory()
    
    results = {}
    for modelType in ["tfidf", "bm25", "bert"]:
        print(f"\nRunning evaluation for model: {modelType}")
        precisionScores = searchEngine.computePrecisionAtK(modelType=modelType)
        results[modelType] = {int(k): float(v) for k, v in precisionScores.items()}
        for k, score in precisionScores.items():
            print(f"Precision@{k}: {score:.4f}")

    # Save all results in the results directory
    evaluation_json_path = os.path.join(results_dir, "evaluation_results.json")
    precision_plot_path = os.path.join(results_dir, "precision_at_k.png")
    
    saveEvaluationResultsToJson(results, evaluation_json_path)
    plotPrecisionAtK(results, precision_plot_path)
    
    if SAVE_EMBEDDINGS_VISUALIZATION and searchEngine.bertEmbeddings is not None:
        embedding_plot_path = os.path.join(results_dir, "embedding_visualization.png")
        category_plot_path = os.path.join(results_dir, "category_distribution.png")
        corpus_stats_path = os.path.join(results_dir, "corpus_statistics.png")
        summary_plot_path = os.path.join(results_dir, "corpus_summary.png")   
        
        plotCorpusSummary(searchEngine.papers, total_fetched, duplicates_removed, summary_plot_path)
        plotEmbeddingVisualization(searchEngine, embedding_plot_path)
        plotCategoryDistribution(searchEngine.papers, category_plot_path)
        plotCorpusStatistics(searchEngine.papers, corpus_stats_path)

    print(f'\nInfo: Saved evaluation results to "{evaluation_json_path}".')
    print(f'Info: Saved Precision@K plot to "{precision_plot_path}".')

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
    plt.figure(figsize=(10, 6))

    # Sort Ks so the x-axis is ordered
    modelNames = list(results.keys())
    for modelName in modelNames:
        kValues = sorted(results[modelName].keys())
        precisions = [results[modelName][k] for k in kValues]
        plt.plot(kValues, precisions, marker="o", label=modelName.upper(), 
                color=PLOT_COLORS.get(modelName, "black"), linewidth=2, markersize=8)

    plt.xlabel("K", fontsize=12)
    plt.ylabel("Precision@K", fontsize=12)
    plt.title("Precision@K Comparison for TF–IDF, BM25, and BERT", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.xticks(kValues)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------------------------------------
# Creates a 2D visualization of BERT embeddings using t-SNE for cluster analysis.
# Input: SemanticSearchEngine instance and output path. Output: PNG scatter plot.
# ----------------------------------------------------------------------------------------------------
def plotEmbeddingVisualization(searchEngine: SemanticSearchEngine, 
                              outputPath: str) -> None:
    if searchEngine.bertEmbeddings is None:
        print("Warning: No BERT embeddings available for visualization.")
        return

    # Sample papers for visualization to avoid overcrowding
    sample_size = min(800, len(searchEngine.papers))
    indices = np.random.choice(len(searchEngine.papers), sample_size, replace=False)
    sample_embeddings = searchEngine.bertEmbeddings[indices]
    sample_papers = [searchEngine.papers[i] for i in indices]
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
    embeddings_2d = tsne.fit_transform(sample_embeddings)
    
    # Get top 10 categories for coloring to avoid legend overflow
    category_counts = {}
    for paper in sample_papers:
        category = paper["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_category_names = [cat for cat, count in top_categories]
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_category_names)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(top_category_names)}
    
    plt.figure(figsize=(16, 10))
    
    # Plot points for top categories
    for category in top_category_names:
        mask = [paper["category"] == category for paper in sample_papers]
        if sum(mask) > 0:  # Only plot if there are points
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       label=category, alpha=0.7, s=40, color=category_to_color[category])
    
    # Group remaining categories as "Other"
    other_mask = [paper["category"] not in top_category_names for paper in sample_papers]
    if sum(other_mask) > 0:
        plt.scatter(embeddings_2d[other_mask, 0], embeddings_2d[other_mask, 1], 
                   label='Other', alpha=0.5, s=30, color='gray', marker='x')
    
    plt.title("t-SNE Visualization of Research Paper Embeddings\n(Top 10 Categories)", fontsize=16, pad=20)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Info: Saved embedding visualization to {outputPath}")


# ----------------------------------------------------------------------------------------------------
# Plots the distribution of papers across categories as a bar chart.
# Input: list of paper dicts and output path. Output: PNG bar plot.
# ----------------------------------------------------------------------------------------------------
def plotCategoryDistribution(papers: List[Dict], outputPath: str) -> None:
    category_counts = {}
    for paper in papers:
        category = paper["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Sort by count and take top 15 categories for clarity
    categories_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    top_categories = categories_sorted[:15]  # Show only top 15 categories
    categories, counts = zip(*top_categories)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(categories)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
    plt.title("Distribution of Papers Across Top 15 Categories", fontsize=16, pad=20)
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Number of Papers", fontsize=12)
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right', fontsize=10)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Info: Saved category distribution to {outputPath}")


# ----------------------------------------------------------------------------------------------------
# Creates a clean visualization of the corpus loading statistics
# ----------------------------------------------------------------------------------------------------    
def plotCorpusStatistics(papers: List[Dict], outputPath: str) -> None:
    category_counts = {}
    for paper in papers:
        category = paper["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Get top 20 categories
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    categories, counts = zip(*top_categories)
    
    # Create horizontal bar chart for better readability
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(categories))
    
    bars = plt.barh(y_pos, counts, color=plt.cm.viridis(np.linspace(0, 1, len(categories))))
    plt.xlabel('Number of Papers', fontsize=12)
    plt.title('Corpus Statistics: Papers per Category (Top 20)', fontsize=16, pad=20)
    plt.yticks(y_pos, categories, fontsize=10)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{count}', ha='left', va='center', fontsize=9)
    
    plt.gca().invert_yaxis()  # Highest count at top
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Info: Saved corpus statistics to {outputPath}")
    

# ----------------------------------------------------------------------------------------------------
# Creates a summary visualization of corpus statistics
# ----------------------------------------------------------------------------------------------------  
def plotCorpusSummary(papers: List[Dict], total_fetched: int, duplicates_removed: int, outputPath: str) -> None:
    unique_papers = len(papers)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Data for plotting
    categories = ['Total Fetched', 'Duplicates Removed', 'Final Corpus']
    values = [total_fetched, duplicates_removed, unique_papers]
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    bars = plt.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{value}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Corpus Collection Summary', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('Number of Papers', fontsize=12)
    plt.ylim(0, max(values) * 1.15)  # Add space for labels
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.xticks(fontsize=11, rotation=0)
    
    stats_text = f'Unique Papers: {unique_papers}\nDuplicates: {duplicates_removed}\nReduction: {(duplicates_removed/total_fetched)*100:.1f}%'
    plt.text(0.65, 0.85, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Info: Saved corpus summary to {outputPath}")   