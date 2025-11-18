import os
import json
import re
import string
import random
from typing import List, Dict
import time

import arxiv
from config import (
    ARXIV_QUERIES,
    MAX_RESULTS_PER_QUERY,
    RANDOM_SEED,
    MAX_TOTAL_PAPERS,
    DATASET_PATH,
)

random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------------------------------------
# Normalizes raw text by lowercasing, collapsing whitespace, and trimming edges.
# Input: text string. Output: cleaned text string for uniform downstream processing.
# ----------------------------------------------------------------------------------------------------
def preprocessText(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
            
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove LaTeX equations and math symbols
    text = re.sub(r'\$.*?\$', '', text)                                                         #Remove inline math
    text = re.sub(r'\\[a-z]+\{.*?\}', '', text)                                                 #Remove simple LaTeX commands
    
    # 3. Handle hyphenated words and contractions properly
    text = re.sub(r'(\w+)-(\w+)', r'\1 \2', text)                                               #Split hyphenated words
    text = re.sub(r"(\w+)'ll", r"\1 will", text)                                                #Expand contractions
    text = re.sub(r"(\w+)'t", r"\1 not", text)
    text = re.sub(r"(\w+)'s", r"\1 is", text)
    text = re.sub(r"(\w+)'re", r"\1 are", text)
    text = re.sub(r"(\w+)'ve", r"\1 have", text)
    
    # 4. Remove URLs and arXiv references
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'arxiv:\d+\.\d+', '', text)
    
    # 5. Remove excessive punctuation but keep meaningful ones
    text = re.sub(r'[^\w\s.,!?;:]', '', text)                                                   #Keep basic punctuation
    
    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    
    # 7. Remove common academic boilerplate that doesn't add semantic value
    boilerplate_phrases = [
        r'in this paper,\s*',
        r'in this work,\s*', 
        r'we propose\s*',
        r'we introduce\s*',
        r'experimental results show\s*',
        r'our method\s*',
        r'this paper presents\s*',
        r'we present\s*',
        r'we show\s*',
        r'we demonstrate\s*'
    ]
    
    for phrase in boilerplate_phrases:
        text = re.sub(phrase, '', text)
    
    # 8. Final cleanup
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    # 9. Remove very short texts that lost meaning
    if len(text.split()) < 3:
        return ""
    
    return text


# ----------------------------------------------------------------------------------------------------
# Fetches papers from the arXiv API for a given query into a list of metadata dictionaries.
# Input: query string and max results. Output: list of dicts with id, title, abstract, and category.
# ----------------------------------------------------------------------------------------------------
def fetchArxivPapers(query: str, maxResults: int) -> List[Dict]:
    try:
        search = arxiv.Search(
            query=query,
            max_results=maxResults,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for result in search.results():
            paperDict = {
                "id": result.get_short_id(),
                "title": result.title,
                "abstract": preprocessText(result.summary),
                "category": result.primary_category,
                "published": result.published,
                "authors": [str(author) for author in result.authors],
            }
            papers.append(paperDict)
        print(f"Fetched {len(papers)} papers for query: {query}")
        return papers
    except Exception as e:
        print(f"Failed to fetch papers for query '{query}': {e}")
        return []


# ----------------------------------------------------------------------------------------------------
# Builds a unified corpus by querying arXiv with multiple queries and removing duplicate paper ids.
# Input: optional list of queries and per-query limit. Output: shuffled list of unique paper dicts.
# ----------------------------------------------------------------------------------------------------
def buildCorpusFromArxiv(queries: List[str] = None,
                            maxResultsPerQuery: int = None,
                            maxPapers: int = None) -> Dict:
    if queries is None:
        queries = ARXIV_QUERIES
    if maxResultsPerQuery is None:
        maxResultsPerQuery = MAX_RESULTS_PER_QUERY
    if maxPapers is None:
        maxPapers = MAX_TOTAL_PAPERS

    allPapers = []
    print(f"Starting arXiv data collection with {len(queries)} queries...")
    print(f"Target: {maxResultsPerQuery} papers per query (max total ≈ {maxPapers})")
    
    for i, query in enumerate(queries):
        print(f"Progress: Query {i+1}/{len(queries)}: {query}")
        queryPapers = fetchArxivPapers(query, maxResultsPerQuery)
        allPapers.extend(queryPapers)
                
        if i < len(queries) - 1:
            time.sleep(2)

        # Early stop with some buffer for duplicates
        if len(allPapers) >= maxPapers * 1.2:
            print(f"Approaching paper limit (~{maxPapers}), stopping early.")
            break

    # Remove duplicate papers
    seenIds = set()
    uniquePapers = []
    duplicate_count = 0
    
    for paper in allPapers:
        if paper["id"] not in seenIds:
            seenIds.add(paper["id"])
            uniquePapers.append(paper)
        else:
            duplicate_count += 1
    
    if len(uniquePapers) > maxPapers:
        print(f"Limiting corpus from {len(uniquePapers)} to {maxPapers} papers")
        uniquePapers = uniquePapers[:maxPapers]

    category_counts = {}
    for paper in uniquePapers:
        category = paper["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\nFinal Corpus Statistics (arXiv API):")
    print(f"Total fetched (with duplicates): {len(allPapers)}")
    print(f"Total unique papers used: {len(uniquePapers)}")
    print(f"Duplicates removed: {duplicate_count}")
    print("Papers per category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(uniquePapers)) * 100
        print(f"  {category}: {count} papers ({percentage:.1f}%)")
    
    if len(uniquePapers) < 1000:
        print("Warning: Corpus size is relatively small for robust evaluation")
    elif len(uniquePapers) >= 2000:
        print("Excellent: Large corpus suitable for comprehensive evaluation")
    
    random.shuffle(uniquePapers)
    corpus_stats = {
        "papers": uniquePapers,
        "total_fetched": len(allPapers),
        "duplicates_removed": duplicate_count,
        "unique_count": len(uniquePapers),
    }
    return corpus_stats
    

# ----------------------------------------------------------------------------------------------------
# Loads papers from a local arXiv metadata JSON snapshot (one JSON object per line),
# applies basic filtering & preprocessing, and returns a corpus_stats dict.
# ----------------------------------------------------------------------------------------------------
def buildCorpusFromJson(dataset_path: str = DATASET_PATH,
                        maxPapers: int = None) -> Dict:
                            
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{dataset_path}'. "
            f"Please place arxiv-metadata-oai-snapshot.json in the project folder."
        )

    print(f"Loading corpus from local JSON: {dataset_path}")
    if maxPapers:
        print(f"Maximum papers to load: {maxPapers}")
    else:
        print("Loading ALL papers (no limit)")

    allPapers = []
    seenIds = set()
    duplicate_count = 0
    total_lines = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            paper_id = record.get("id")
            if not paper_id:
                continue

            # Handle duplicates
            if paper_id in seenIds:
                duplicate_count += 1
                continue

            # Preprocess abstract
            raw_abstract = record.get("abstract", "")
            abstract = preprocessText(raw_abstract)
            if not abstract:
                continue

            # Extract fields
            title = record.get("title", "").strip()
            category_str = record.get("categories", "")
            category = category_str.split()[0] if category_str else "unknown"

            # Authors
            authors_parsed = record.get("authors_parsed")
            if isinstance(authors_parsed, list) and authors_parsed:
                authors = [" ".join(x for x in a if x) for a in authors_parsed]
            else:
                authors = [a.strip() for a in record.get("authors", "").split(" and ")] \
                        if record.get("authors") else ["Unknown"]

            paperDict = {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "category": category,
                "published": record.get("update_date") or None,
                "authors": authors,
            }

            allPapers.append(paperDict)
            seenIds.add(paper_id)

            # Stop early ONLY if the user passed maxPapers
            if maxPapers and len(allPapers) >= maxPapers:
                break

    # Stats
    print("\nLoaded Papers Summary (Local JSON):")
    print(f"Total lines read: {total_lines}")
    print(f"Total unique papers loaded: {len(allPapers)}")
    print(f"Duplicate IDs skipped: {duplicate_count}")
    corpus_stats = {
        "papers": allPapers,
        "total_fetched": total_lines,
        "duplicates_removed": duplicate_count,
        "unique_ids_encountered": len(seenIds),
        "unique_papers_loaded": len(allPapers)
    }

    return corpus_stats