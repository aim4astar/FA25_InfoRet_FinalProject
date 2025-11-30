import json
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

from search_engine import SemanticSearchEngine
from config import plotColors, saveEmbeddingsVisualization, resultsDir, embeddingModelNames


# ----------------------------------------------------------------------------------------------------
# Creates results directory if it doesn't exist to ensure all outputs are organized in one place.
# Input: none. Output: creates directory structure and returns full path to results folder.
# ----------------------------------------------------------------------------------------------------
def ensureResultsDirectory() -> str:
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
        print(f"Info: Created results directory: {resultsDir}")
    return resultsDir



# ----------------------------------------------------------------------------------------------------
# Runs Precision@K evaluation for TF-IDF, BM25, and all BERT models and prints the scores.
# Input: SemanticSearchEngine instance. Output: dict mapping model names to Precision@K scores.
# Also triggers saving of results to JSON and plotting Precision@K as a PNG image.
# ----------------------------------------------------------------------------------------------------
def runFullEvaluation(searchEngine: SemanticSearchEngine, 
                     total_fetched: int, duplicates_removed: int) -> Dict[str, Dict[int, float]]:
    resultsDirLocal = ensureResultsDirectory()   
    results = {}

    # Evaluate traditional + topic-model baselines
    for modelType in ["tfidf", "bm25", "lsa", "lda"]:
        print(f"\nRunning evaluation for model: {modelType}")
        precisionScores = searchEngine.computePrecisionAtK(modelType=modelType)
        results[modelType] = {int(k): float(v) for k, v in precisionScores.items()}
        for k, score in precisionScores.items():
            print(f"Precision@{k}: {score:.4f}")


    # Evaluate all BERT models
    for modelKey in embeddingModelNames.keys():
        print(f"\nRunning evaluation for BERT model: {modelKey}")
        precisionScores = searchEngine.computePrecisionAtK(modelType=modelKey)
        results[modelKey] = {int(k): float(v) for k, v in precisionScores.items()}
        for k, score in precisionScores.items():
            print(f"Precision@{k}: {score:.4f}")

    # Save all results in the results directory
    evaluationJsonPath = os.path.join(resultsDirLocal, "evaluation_results.json")
    precisionPlotPath = os.path.join(resultsDirLocal, "precision_at_k.png")
    precisionSummaryPath = os.path.join(resultsDirLocal, "precision_at_k_summary.png")
    
    saveEvaluationResultsToJson(results, evaluationJsonPath)
    plotPrecisionAtK(results, precisionPlotPath)
    plotPrecisionAtKSummary(results, kValue=10, outputPath=precisionSummaryPath)
    
    if saveEmbeddingsVisualization and searchEngine.bertEmbeddings:
        embeddingPlotPath = os.path.join(resultsDirLocal, "embedding_visualization.png")
        categoryPlotPath = os.path.join(resultsDirLocal, "category_distribution.png")
        summaryPlotPath = os.path.join(resultsDirLocal, "corpus_summary.png")
        
        plotCorpusSummary(searchEngine.papers, total_fetched, duplicates_removed, summaryPlotPath)
        plotEmbeddingVisualization(searchEngine, embeddingPlotPath)
        plotCategoryDistribution(searchEngine.papers, categoryPlotPath)

    print(f'\nInfo: Saved evaluation results to "{evaluationJsonPath}".')
    print(f'Info: Saved Precision@K plot to "{precisionPlotPath}".')
    print(f'Info: Saved Precision@K summary plot to "{precisionSummaryPath}".')
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
    plt.figure(figsize=(12, 8))

    # Sort Ks so the x-axis is ordered
    modelNames = list(results.keys())
    for modelName in modelNames:
        kValues = sorted(results[modelName].keys())
        precisions = [results[modelName][k] for k in kValues]
        plt.plot(kValues, precisions, marker="o", label=modelName.upper(), 
                color=plotColors.get(modelName, "black"), linewidth=2, markersize=8)

    plt.xlabel("K", fontsize=12)
    plt.ylabel("Precision@K", fontsize=12)
    plt.title("Precision@K Comparison for TF-IDF, BM25, and BERT Models", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Place legend outside the plot to prevent overlapping
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.xticks(kValues)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ----------------------------------------------------------------------------------------------------
# Creates a 2D visualization of BERT embeddings using t-SNE for cluster analysis.
# Input: SemanticSearchEngine instance and output path. Output: PNG scatter plot.
# ----------------------------------------------------------------------------------------------------
def plotEmbeddingVisualization(searchEngine: SemanticSearchEngine, 
                              outputPath: str) -> None:
    if not searchEngine.bertEmbeddings:
        print("Warning: No BERT embeddings available for visualization.")
        return

    # Use the first available BERT model for visualization
    modelKey = list(searchEngine.bertEmbeddings.keys())[0]
    embeddings = searchEngine.bertEmbeddings[modelKey]
    
    # Sample papers for visualization to avoid overcrowding
    sample_size = min(800, len(searchEngine.papers))
    indices = np.random.choice(len(searchEngine.papers), sample_size, replace=False)
    sample_embeddings = embeddings[indices]
    sample_papers = [searchEngine.papers[i] for i in indices]
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
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
    
    plt.title(f"t-SNE Visualization of Research Paper Embeddings ({modelKey.upper()})\n(Top 10 Categories)", fontsize=16, pad=20)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    
    # Place legend outside to prevent overlapping
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', facecolor='white')
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
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Info: Saved category distribution to {outputPath}")


# ----------------------------------------------------------------------------------------------------
# Creates a clean and professional summary visualization of corpus statistics
# Input: papers list, total fetched count, duplicates removed count, and output path.
# Output: PNG summary plot saved to disk.
# ----------------------------------------------------------------------------------------------------
def plotCorpusSummary(papers: List[Dict], total_fetched: int, duplicates_removed: int, outputPath: str) -> None:
    unique_papers = len(papers)
    
    # Create the plot with professional styling
    plt.figure(figsize=(10, 6))
    
    # Data for plotting
    categories = ['Total Fetched', 'Duplicates Removed', 'Final Corpus']
    values = [total_fetched, duplicates_removed, unique_papers]
    
    # Professional color scheme
    colors = ['#2E86AB', '#A23B72', '#18A558']  # Blue, Purple, Green
    
    # Create bar chart
    x_pos = np.arange(len(categories))
    bars = plt.bar(x_pos, values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    # Customize the plot
    plt.xlabel('Processing Stage', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Number of Papers', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Research Corpus Collection Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    plt.xticks(x_pos, categories, fontsize=11, fontweight='bold')
    
    # Remove spines for cleaner look
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + (max(values) * 0.01),
                f'{value:,}', ha='center', va='bottom', fontsize=12, 
                fontweight='bold', color='#2C3E50')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    plt.ylim(0, max(values) * 1.15)
    
    # Add informative text box
    stats_text = f'• Unique Papers: {unique_papers:,}\n• Duplicates Removed: {duplicates_removed:,}\n• Reduction Rate: {(duplicates_removed/total_fetched)*100:.1f}%'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.9,
                      edgecolor='#D1D5DB', linewidth=1),
             fontweight='bold', color='#2C3E50')
    
    # Add a subtle background
    plt.gca().set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Info: Saved corpus summary to {outputPath}")


# ----------------------------------------------------------------------------------------------------
# Plots a bar-chart summary of Precision@K for each model to highlight differences in retrieval quality.
# Input: nested precision results dict, the K value to summarize, and output image path. 
# Output: PNG bar chart saved to disk.
# ----------------------------------------------------------------------------------------------------
def plotPrecisionAtKSummary(results: Dict[str, Dict[int, float]],
                            kValue: int,
                            outputPath: str) -> None:
    if not results:
        print("Warning: No evaluation results available for Precision@K summary plot.")
        return

    # Collect models that contain the requested K value
    modelNames = []
    precisionValues = []
    for modelName, scoreDict in results.items():
        if kValue in scoreDict:
            modelNames.append(modelName.upper())
            precisionValues.append(scoreDict[kValue])
        else:
            print(f"Warning: Model '{modelName}' does not have Precision@{kValue} computed.")

    if not modelNames:
        print(f"Warning: No models contain Precision@{kValue}; skipping summary plot.")
        return

    plt.figure(figsize=(10, 7))
    
    xPositions = np.arange(len(modelNames))

    # Use configured colors when available, otherwise default to gray
    barColors = [plotColors.get(name.lower(), "gray") for name in modelNames]

    bars = plt.bar(xPositions, precisionValues, color=barColors, alpha=0.85, edgecolor="black")
    plt.xticks(xPositions, modelNames, fontsize=11)
    plt.ylabel(f"Precision@{kValue}", fontsize=12)
    plt.title(f"Precision@{kValue} Comparison Across Retrieval Models", fontsize=14, pad=15)
    plt.ylim(0.0, 1.05 * max(precisionValues))

    # Add numeric labels on top of each bar
    for bar, value in zip(bars, precisionValues):
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 0.01,
                 f"{value:.3f}",
                 ha="center",
                 va="bottom",
                 fontsize=10)

    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"Info: Saved Precision@{kValue} summary bar plot to {outputPath}")    


# ----------------------------------------------------------------------------------------------------
# Creates a visualization of interactive search results showing top matching papers and their scores.
# Input: query string, results list, model type, and output path. Output: PNG bar plot saved to disk.
# ----------------------------------------------------------------------------------------------------
def plotInteractiveSearchResults(query: str, 
                                results: List[Tuple[Dict, float]], 
                                modelType: str,
                                outputPath: str) -> None:
    if not results:
        print("Warning: No results to plot for interactive search.")
        return

    # Prepare data for plotting
    paper_ids = [paper['id'] for paper, score in results]
    scores = [score for paper, score in results]
    categories = [paper['category'] for paper, score in results]
    
    # Create the plot with better styling
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(paper_ids))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(scores)))
    
    bars = plt.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.xlabel('Similarity Score', fontsize=12, fontweight='bold')
    plt.ylabel('Paper ID', fontsize=12, fontweight='bold')
    plt.title(f'Semantic Search Results - {modelType.upper()} Model\nQuery: "{query[:80]}{"..." if len(query) > 80 else ""}"', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis to show paper IDs
    plt.yticks(y_pos, paper_ids, fontsize=10)
    
    # Add score labels and additional info on the bars
    for i, (bar, score, category) in enumerate(zip(bars, scores, categories)):
        width = bar.get_width()
        # Score on the right
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
        # Category inside the bar
        if width > 0.1:  # Only put text if bar is wide enough
            plt.text(width * 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{category}', 
                    ha='left', va='center', fontsize=8, color='white',
                    fontweight='bold')
    
    # Add grid and adjust layout
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.gca().invert_yaxis()  # Highest score at top
    plt.xlim(0, max(scores) * 1.2 if scores else 1)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ----------------------------------------------------------------------------------------------------
# Creates a comparison visualization of search results across all BERT models for the same query.
# Input: query string, results dictionary (model -> results), and output path. 
# Output: PNG comparison plot saved to disk.
# ----------------------------------------------------------------------------------------------------
def plotMultiModelSearchResults(query: str, 
                               resultsDict: Dict[str, List[Tuple[Dict, float]]],
                               outputPath: str) -> None:
    if not resultsDict:
        print("Warning: No results to plot for multi-model search comparison.")
        return

    # Extract model names and rank positions
    model_keys = list(resultsDict.keys())
    ranks = np.arange(1, 11)                                                                                #Top 10
    plt.figure(figsize=(12, 7))

    for model in model_keys:
        results = resultsDict[model][:10]
        scores = [score for (_, score) in results]
        plt.plot(ranks, scores, marker="o", linewidth=2, markersize=6, label=model.upper())

    plt.title(
        f"Multi-Model Semantic Search Comparison\n"
        f"Query: \"{query[:80]}{'...' if len(query) > 80 else ''}\"",
        fontsize=14,
        pad=20
    )

    plt.xlabel("Rank Position", fontsize=12)
    plt.ylabel("Similarity Score", fontsize=12)
    plt.xticks(ranks)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()


# ----------------------------------------------------------------------------------------------------
# Creates a heatmap comparison of search results across all models.
# Input: query string, results dictionary, and output path.
# Output: PNG heatmap plot saved to disk.
# ----------------------------------------------------------------------------------------------------
def plotMultiModelHeatmap(query: str,
                         resultsDict: Dict[str, List[Tuple[Dict, float]]],
                         outputPath: str) -> None:
    if not resultsDict:
        print("Warning: No results to plot for multi-model heatmap.")
        return

    # Create heatmap data
    model_keys = list(resultsDict.keys())
    num_results = 10  # Top 10 results
    
    # Prepare heatmap data
    heatmap_data = []
    
    for model_key in model_keys:
        results = resultsDict[model_key][:num_results]
        scores = [score for paper, score in results]
        heatmap_data.append(scores)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap
    im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Customize heatmap
    plt.xticks(np.arange(num_results))
    plt.yticks(np.arange(len(model_keys)))
    plt.xlabel('Rank Position', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Similarity Scores Heatmap - Multi-Model Comparison\n' +
              f'Query: "{query[:80]}{"..." if len(query) > 80 else ""}"',
              fontsize=14, fontweight='bold', pad=20)
    
    # Set tick labels
    plt.xticks(np.arange(num_results), [f'Rank {i+1}' for i in range(num_results)], rotation=45)
    plt.yticks(np.arange(len(model_keys)), [model.upper() for model in model_keys])
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Similarity Score', rotation=270, labelpad=15)
    
    # Add text annotations on heatmap
    for i in range(len(model_keys)):
        for j in range(num_results):
            if j < len(heatmap_data[i]):
                plt.text(j, i, f'{heatmap_data[i, j]:.2f}',
                        ha="center", va="center", color="black", 
                        fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ----------------------------------------------------------------------------------------------------
# Exports search results to Excel format for detailed analysis
# Input: query string, results dictionary, and output path.
# Output: Excel file saved to disk.
# ----------------------------------------------------------------------------------------------------
def exportSearchResultsToExcel(query: str,
                              resultsDict: Dict[str, List[Tuple[Dict, float]]],
                              outputPath: str) -> None:
    try:
        import pandas as pd
        
        # Create Excel writer
        with pd.ExcelWriter(outputPath, engine='openpyxl') as writer:
            
            # Create summary sheet
            summary_data = []
            for model_key, results in resultsDict.items():
                for rank, (paper, score) in enumerate(results[:10]):
                    summary_data.append({
                        'Model': model_key.upper(),
                        'Rank': rank + 1,
                        'Paper_ID': paper['id'],
                        'Title': paper['title'],
                        'Category': paper['category'],
                        'Score': score,
                        'Authors': ', '.join(paper.get('authors', ['Unknown'])[:3])
                    })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Search_Results', index=False)
            
            # Create comparison sheet
            comparison_data = []
            for rank in range(10):
                row = {'Rank': rank + 1}
                for model_key in resultsDict.keys():
                    if rank < len(resultsDict[model_key]):
                        paper = resultsDict[model_key][rank][0]
                        score = resultsDict[model_key][rank][1]
                        row[f'{model_key.upper()}_Paper_ID'] = paper['id']
                        row[f'{model_key.upper()}_Score'] = score
                        row[f'{model_key.upper()}_Category'] = paper['category']
                    else:
                        row[f'{model_key.upper()}_Paper_ID'] = ''
                        row[f'{model_key.upper()}_Score'] = ''
                        row[f'{model_key.upper()}_Category'] = ''
                comparison_data.append(row)
            
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # Create query info sheet
            query_info = pd.DataFrame([{
                'Query': query,
                'Timestamp': pd.Timestamp.now(),
                'Total_Models': len(resultsDict),
                'Results_Per_Model': 10
            }])
            query_info.to_excel(writer, sheet_name='Query_Info', index=False)
        
    except ImportError:
        print("Warning: pandas or openpyxl not installed. Cannot export to Excel.")
        print("Install with: pip install pandas openpyxl")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")