# Semantic Similarity Search and Ranking for arXiv Research Papers
### Using TF–IDF, BM25, Sentence-BERT, and FAISS

This project implements a semantic similarity search engine for academic research papers using **arXiv abstract data**. It evaluates and compares traditional keyword-based retrieval models (TF–IDF, BM25) with modern embedding-based semantic search (Sentence-BERT + FAISS). The system is fully modular, GPU-aware, batch-optimized, supports automatic dataset download via **KaggleHub**, and provides real-time interactive querying along with quantitative evaluation through **Precision@K**.

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

### ⚙️ GPU/CPU Auto-Detection
The system automatically selects the optimal compute device:

- Uses **CUDA GPU**, if available  
- Falls back to **CPU** with clean messages

---

### 📥 Automatic Dataset Download (KaggleHub)
You **do NOT need to manually download the dataset**.

If the file:

```
arxiv-metadata-oai-snapshot.json
```

is not found, the system automatically downloads it using:

```python
kagglehub.dataset_download("Cornell-University/arxiv")
```

No credentials or authentication needed.

---

### 🌐 Optional arXiv API Mode
Instead of using the dataset snapshot, you can fetch papers directly from the arXiv API:

```bash
python main.py --use-arxiv-api
```

---

### 📊 Precision@K Evaluation
Implements category-based evaluation using:

- P@1  
- P@3  
- P@5  
- P@10

This measures the proportion of retrieved papers that share the same **arXiv category** as the query.

The system generates:

- Precision@K line chart  
- Precision@10 bar chart  
- Multi-model comparison line plot  
- Multi-model similarity heatmap  
- Embedding t-SNE plot  
- Category distribution visualization  
- Corpus summary statistics  
- Single-model search result visualization  

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
├── environment.yml
└── results/
```

---

## ⚙️ Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate semantic-search-env
```

---

### 2. Optional: Verify installation

```bash
python -c "import torch, faiss, sentence_transformers, kagglehub; print('Environment OK!')"
```

---

## ▶️ Running the Project

### **1. Full pipeline (evaluation + interactive search)**

```bash
python main.py
```

---

### **2. Only interactive search (skip evaluation)**

```bash
python main.py --no-eval
```

---

### **3. Choose retrieval model**

```bash
python main.py --model tfidf
python main.py --model bm25
python main.py --model bert
```

---

### **4. Change number of results returned**

```bash
python main.py --topk 15
```

---

### **5. Limit dataset size (faster for development)**

```bash
python main.py --max-papers 50000
```

---

### **6. Use arXiv API instead of dataset**

```bash
python main.py --use-arxiv-api
```

---

## 🧪 Running Unit Tests

```bash
python -m unittest test_semantic_search.py
```

Tests include:

- Text preprocessing  
- Device selection  
- Embedding shape verification  
- Retrieval correctness  

---

## 🔧 Technical Notes

### ✔ Windows Symlink Fix  
The project includes:

```python
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
```

This prevents Windows symlink errors such as:

```
WinError 1314: A required privilege is not held by the client
```

---

### ✔ Dataset Size  
The arXiv snapshot is very large.  
Use `--max-papers` for debugging.

---

## 📚 Possible Extensions

- Persist FAISS index to disk  
- Add web UI (Flask/FastAPI)  
- Add cross-encoder reranking  
- Add embedding-based clustering  
- Integrate citation graph analysis  

---
