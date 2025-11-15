# Semantic Similarity Search and Ranking for arXiv Research Papers
### Using TF–IDF, BM25, Sentence-BERT, and FAISS

This project implements a semantic similarity search engine for academic research papers using **arXiv abstract data**. It evaluates and compares traditional keyword-based retrieval models (TF–IDF, BM25) with modern embedding-based semantic search (Sentence-BERT + FAISS). The system is fully modular, GPU-aware, batch-optimized, and supports real-time interactive querying along with quantitative evaluation through **Precision@K**.

---

## 🚀 Features

### 🔍 Multi-Model Retrieval
The search engine provides three retrieval strategies:

| Model | Type | Description |
|-------|------|-------------|
| **TF–IDF** | Traditional | Sparse lexical vector representation via scikit-learn |
| **BM25** | Probabilistic | Strong keyword-based baseline for document ranking |
| **Sentence-BERT** | Semantic | Dense contextual embeddings capturing meaning, not just keywords |

---

### ⚡ High-Performance Search (FAISS)
- Sentence-BERT embeddings indexed using **FAISS** for fast nearest-neighbor lookup.
- Uses **inner-product search** with L2-normalized vectors.
- Automatically falls back to cosine similarity if FAISS is disabled.

---

### 🧱 Modular Architecture (6 Files)
The codebase is divided into exactly **six core implementation files**, aligned with assignment requirements:

```
config.py
data_loader.py
text_representation.py
search_engine.py
evaluation.py
main.py
```

Additional supporting files:

```
test_semantic_search.py
requirements.txt
README.md
```

Each function includes standardized header-style comments for clarity.

---

### ⚙️ GPU/CPU Auto-Detection
The system automatically selects the optimal compute device:

- Uses **CUDA GPU**, if available  
- Falls back to **CPU** with clean messages (no warnings)

---

### 🚀 Batch Processing for Speed
Batch encoding is used for all embedding operations to maximize throughput. This dramatically speeds up processing when handling 100–500 arXiv abstracts.

---

### 📊 Precision@K Evaluation
Implements category-based evaluation using:

- P@1  
- P@3  
- P@5  
- P@10

This measures the proportion of retrieved papers that share the same **arXiv category** as the query.

---

### 🎛 Interactive Search Mode
After evaluation, the program switches to an interactive mode:

```
Enter an abstract or description:
```

Returns ranked papers with:

- Rank
- Similarity score
- arXiv ID
- Category
- Title
- Abstract snippet

---

## 📁 Folder Structure

```
project/
│
├── config.py
├── data_loader.py
├── text_representation.py
├── search_engine.py
├── evaluation.py
├── main.py
│
├── test_semantic_search.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### **1. Full pipeline (evaluation + interactive search)**

```bash
python main.py
```

### **2. Only interactive search (skip evaluation)**

```bash
python main.py --no-eval
```

### **3. Choose retrieval model**

```bash
python main.py --model tfidf
python main.py --model bm25
python main.py --model bert
```

### **4. Change number of results returned**

```bash
python main.py --topk 15
```

---

## 🧪 Running Unit Tests

```bash
python -m unittest test_semantic_search.py
```

This validates:

- Text preprocessing  
- Device selection (GPU/CPU)  
- Embedding shape consistency  
- Overall search engine correctness  

---

## 🔍 How the arXiv API Works

The project uses the official `arxiv` Python library.

Example call:

```python
search = arxiv.Search(
    query="cat:cs.LG",
    max_results=150,
    sort_by=arxiv.SortCriterion.SubmittedDate
)
```

Each returned `result` object contains:

- title  
- summary (abstract)  
- primary_category  
- published date  
- id (via `get_short_id()`)  

Abstracts are normalized and fed into the vectorization/embedding pipeline.

---

## 📐 Alignment With Assignment Requirements

| Requirement | Status | Explanation |
|------------|--------|-------------|
| Use arXiv API | ✔️ | arxiv library used for metadata retrieval |
| TF–IDF, BM25 models | ✔️ | Implemented and benchmarked |
| Neural semantic model | ✔️ | Sentence-BERT embeddings |
| Cosine similarity & FAISS | ✔️ | Both included |
| GPU/CPU, batching | ✔️ | Device auto-selection + batch encoding |
| Precision@K | ✔️ | Category-based evaluation |
| Modular files (≤6) | ✔️ | Exactly six implementation files |
| Header comments | ✔️ | Added before every method |
| Interactive system | ✔️ | CLI interface implemented |

---

## 📚 Possible Extensions

- Persist FAISS index to disk
- Cache arXiv results to speed up development
- Add a simple web interface (Flask or FastAPI)
- Add visualization (TSNE of embeddings)

---

## 📜 License
Free for academic, educational, and research use.
