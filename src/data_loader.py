import re
import string
import random
from typing import List, Dict
import time

import arxiv
from config import ARXIV_QUERIES, MAX_RESULTS_PER_QUERY, RANDOM_SEED, MAX_TOTAL_PAPERS

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
                         maxResultsPerQuery: int = None) -> List[Dict]:
    if queries is None:
        queries = ARXIV_QUERIES
    if maxResultsPerQuery is None:
        maxResultsPerQuery = MAX_RESULTS_PER_QUERY

    allPapers = []
    print(f"Starting arXiv data collection with {len(queries)} queries...")
    print(f"Target: {maxResultsPerQuery} papers per query")
    
    for i, query in enumerate(queries):
        print(f"Progress: Query {i+1}/{len(queries)}: {query}")
        queryPapers = fetchArxivPapers(query, maxResultsPerQuery)
        allPapers.extend(queryPapers)
                
        if i < len(queries) - 1:                                                #Avoid rate limiting - longer delay between queries
            time.sleep(2)                                                       #Increased to 2 seconds
                
        if len(allPapers) >= MAX_TOTAL_PAPERS * 1.2:                            #Buffer for duplication
            print(f"Approaching paper limit, stopping early.")
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
    
    if len(uniquePapers) > MAX_TOTAL_PAPERS:                                                #Apply maximum paper limit
        print(f"Limiting corpus from {len(uniquePapers)} to {MAX_TOTAL_PAPERS} papers")
        uniquePapers = uniquePapers[:MAX_TOTAL_PAPERS]

    category_counts = {}
    for paper in uniquePapers:
        category = paper["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"\nFinal Corpus Statistics:")
    print(f"Total unique papers: {len(uniquePapers)}")
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
        'papers': uniquePapers,
        'total_fetched': len(allPapers),
        'duplicates_removed': duplicate_count,
        'unique_count': len(uniquePapers)
    }
    return corpus_stats