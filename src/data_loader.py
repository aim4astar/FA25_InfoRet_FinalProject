import re
import random
from typing import List, Dict

import arxiv
from config import ARXIV_QUERIES, MAX_RESULTS_PER_QUERY, RANDOM_SEED

random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------------------------------------
# Normalizes raw text by lowercasing, collapsing whitespace, and trimming edges.
# Input: text string. Output: cleaned text string for uniform downstream processing.
# ----------------------------------------------------------------------------------------------------
def preprocessText(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ----------------------------------------------------------------------------------------------------
# Fetches papers from the arXiv API for a given query into a list of metadata dictionaries.
# Input: query string and max results. Output: list of dicts with id, title, abstract, and category.
# ----------------------------------------------------------------------------------------------------
def fetchArxivPapers(query: str, maxResults: int) -> List[Dict]:
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
        }
        papers.append(paperDict)
    return papers


# ----------------------------------------------------------------------------------------------------
# Builds a unified corpus by querying arXiv with multiple queries and deduplicating paper ids.
# Input: optional list of queries and per-query limit. Output: shuffled list of unique paper dicts.
# ----------------------------------------------------------------------------------------------------
def buildCorpusFromArxiv(queries: List[str] = None,
                         maxResultsPerQuery: int = None) -> List[Dict]:
    if queries is None:
        queries = ARXIV_QUERIES
    if maxResultsPerQuery is None:
        maxResultsPerQuery = MAX_RESULTS_PER_QUERY

    allPapers = []
    for query in queries:
        queryPapers = fetchArxivPapers(query, maxResultsPerQuery)
        allPapers.extend(queryPapers)

    seenIds = set()
    uniquePapers = []
    for paper in allPapers:
        if paper["id"] not in seenIds:
            seenIds.add(paper["id"])
            uniquePapers.append(paper)

    random.shuffle(uniquePapers)
    return uniquePapers