# Semantic Similarity Search and Ranking of Research Papers Using arXiv Abstract Data
### TF–IDF • BM25 • LSA • LDA • MiniLM • MPNet • MultiQA • FAISS

This project implements a comprehensive **semantic similarity search system** for arXiv research papers, combining classical IR techniques with modern neural embedding models. Inspired by the challenges described in our report :contentReference[oaicite:1]{index=1}, the system retrieves research papers based on **conceptual meaning**, not just keyword overlap.

The system supports **dual data acquisition modes**, full **text preprocessing**, seven **retrieval models**, FAISS-accelerated similarity search, and a complete **Precision@K evaluation pipeline**. It also provides an **interactive semantic search interface**.

---

## 🚀 Key Features

### 🔍 Multi-Model Retrieval (7 Models)
This framework compares **seven distinct representation techniques**, consistent with the models discussed in our mid-project report :contentReference[oaicite:2]{index=2}:

| Category | Models | Description |
|----------|--------|-------------|
| **Lexical** | TF–IDF, BM25 | Keyword-based, sparse representations |
| **Topic Models** | LSA, LDA | Latent semantic structure extraction |
| **Neural Embeddings** | MiniLM, MPNet, MultiQA | Transformer-based dense embeddings (384–768 dims), state-of-the-art semantic search |

---

### ⚡ High-Performance Similarity Search (FAISS)

- Uses **FAISS IndexFlatIP** for fast inner-product search  
- Supports datasets with **hundreds of thousands of embeddings**  
- Automatically falls back to cosine similarity when FAISS is unavailable  
- Embeddings automatically **L2-normalize** for IP-cosine equivalence  

This matches the scalable search discussion in Section II-D of the report :contentReference[oaicite:3]{index=3}.

---

### 🧠 Automatic Device Selection

Sentence-BERT models automatically select:

- **CUDA GPU** (if available)  
- **CPU** fallback  

Mirroring the neural framework described in Section III-C :contentReference[oaicite:4]{index=4}.

---

### 📥 Automatic Dataset Handling (KaggleHub)

Consistent with the “dual data acquisition pipeline” described in Sections III–IV :contentReference[oaicite:5]{index=5}:

- If local dataset file:

```
arxiv-metadata-oai-snapshot.json
```

is missing, the system **automatically downloads** the latest dataset from:

```
Cornell-University/arxiv  (via kagglehub)
```

- No API key required  
- Creates a local copy for reproducibility  
- Provides the same JSONL format used in the report’s methodology  

---

### 🌐 Optional arXiv API Mode

The system can alternatively pull papers live from the arXiv API:

```bash
python main.py --use-arxiv-api
```

Supports all categories listed in the report (cs.LG, cs.AI, cs.CL, stat.ML, math.NA, etc.) :contentReference[oaicite:6]{index=6}.

---

## 🧹 Text Preprocessing Pipeline

Based on the pipeline described in Section IV-A of the report :contentReference[oaicite:7]{index=7}, the system performs:

- Removal of LaTeX expressions & equations  
- URL & reference cleaning  
- Contraction expansion  
- Academic boilerplate filtering  
- Punctuation normalization  
- Lower-casing  
- Duplicate detection based on arXiv ID  

This ensures clean, semantically meaningful text for all models.

---

## 📊 Evaluation: Precision@K

Implements category-based evaluation using:

- P@1  
- P@3  
- P@5  
- P@10  

As described in Section IV-C of the report :contentReference[oaicite:8]{index=8}.

Generated evaluation plots include:

- Precision@K line plot  
- Precision@10 summary bar chart  
- Multi-model similarity heatmap  
- Multi-model comparison (line plot)  
- Embedding t-SNE visualization  
- Category distribution plot  
- Corpus summary plot  

All redundant/duplicate plots (e.g., earlier bar comparison) have been removed.

---

## 🎛 Interactive Search Mode

After evaluation finishes, the system enters:

```
Enter an abstract or description:
```

Returns for each result:

- Rank  
- Similarity score  
- Title  
- arXiv ID  
- Category  
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
├── environment.yml
└── search_results/
```

---

## ⚙️ Installation

### 1. Create the Conda environment
(Updated to reflect modern dependencies)

```bash
conda env create -f environment.yml
```

Activate:

```bash
conda activate semantic-search-env
```

---

### 2. Verify installation (optional)

```bash
python -c "import torch, faiss, sentence_transformers, kagglehub; print('Environment OK!')"
```

---

## ▶️ Running the Project

### 1. Full pipeline (evaluation + interactive search)

```bash
python main.py
```

### 2. Only interactive search

```bash
python main.py --no-eval
```

### 3. Choose a model

```bash
python main.py --model tfidf
python main.py --model bm25
python main.py --model bert
```

### 4. Limit dataset size (faster development)

```bash
python main.py --max-papers 50000
```

### 5. Use arXiv API instead of dataset

```bash
python main.py --use-arxiv-api
```

---

## 🧪 Running Tests

```bash
python -m unittest test_semantic_search.py
```

Tests cover:

- Preprocessing  
- Device selection  
- Embedding integrity  
- Retrieval correctness  

---

## 🔧 Windows Note (Important)

To prevent HuggingFace symlink errors (`WinError 1314`), the system sets:

```python
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
```

included at the top of `text_representation.py`.

---

## 📚 Possible Extensions

As suggested in Section VII of the report :contentReference[oaicite:9]{index=9}:

- Persist FAISS index  
- Add reranking models (cross-encoders)  
- Build a web UI (Flask/FastAPI)  
- Add embedding-based clustering  
- Integrate citation graph retrieval  

---
