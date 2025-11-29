# Semantic Similarity Search and Ranking for arXiv Research Papers
### Using TF–IDF, BM25, LSA, LDA, MiniLM, MPNet, MultiQA, and FAISS

This project implements a full semantic search system for arXiv research papers using their abstracts. It compares traditional information-retrieval techniques (TF–IDF, BM25, topic models) with state-of-the-art transformer-based embedding models (MiniLM, MPNet, MultiQA). The system evaluates how well each model retrieves papers with similar meaning and provides an interactive semantic search interface.

The pipeline includes dataset handling, preprocessing, vectorization, similarity search (FAISS), evaluation using Precision@K, and result visualization.

---

## 🚀 What This Project Does

### 🔍 1. Loads and Processes the arXiv Metadata Dataset
- Automatically downloads the dataset via **KaggleHub** if not found locally  
- Cleans and normalizes abstracts  
- Removes LaTeX, URLs, special characters, and duplicates  
- Extracts titles, categories, and abstract text  

---

### 📚 2. Creates Multiple Text Representations (7 Models)

| Category | Models | Description |
|----------|--------|-------------|
| **Lexical Models** | TF–IDF, BM25 | Keyword-based sparse vectors |
| **Topic Models** | LSA, LDA | Latent semantic topic structure |
| **Transformer Embeddings** | MiniLM, MPNet, MultiQA | Dense contextual semantic vectors |

These models work together to compare keyword relevance vs. conceptual understanding.

---

### ⚡ 3. Performs High-Speed Similarity Search (FAISS)

- Embeddings indexed using **FAISS IndexFlatIP** for efficient nearest-neighbor search  
- Uses inner-product (cosine-equivalent) similarity  
- Falls back to sklearn cosine similarity when FAISS isn't available  
- Supports fast search even on thousands of abstracts  

---

### 📊 4. Evaluates Models Using Precision@K

The evaluation tests how often each model returns papers from the **same arXiv category** as the query.

Metrics:

- Precision@1  
- Precision@3  
- Precision@5  
- Precision@10  

Generated plots include:

- Precision@K line plot  
- Precision@10 bar chart  
- Multi-model comparison line plot  
- Multi-model heatmap  
- Embedding t-SNE visualization  
- Category distribution plot  
- Corpus summary statistics  
- Single-model search results plot  

All plots are saved in the `search_results/` directory.

---

### 🎛 5. Interactive Semantic Search

After evaluation, the system enters an interactive mode:

```
Enter an abstract or description:
```

It returns the top-ranked research papers with:

- Rank  
- Similarity score  
- Title  
- arXiv ID  
- Category  
- Abstract snippet  

This allows users to test the semantic capabilities of each model.

---

## 📁 Project Structure

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
├── test_semantic_search.py
└── search_results/
```

---

## ⚙️ Installation

### 1. Create the Conda environment

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

### **1. Full pipeline (evaluation + search)**

```bash
python main.py
```

---

### **2. Only interactive semantic search**

```bash
python main.py --no-eval
```

---

### **3. Choose a retrieval model**

```bash
python main.py --model tfidf
python main.py --model bm25
python main.py --model bert
```

(Use lowercase names as defined in `config.py`.)

---

### **4. Limit number of papers loaded**

```bash
python main.py --max-papers 50000
```

---

### **5. Use arXiv API instead of dataset**

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
- Embedding output checks  
- Search engine behavior  

---

## 🛠 Windows Notes

To prevent symlink errors during model downloads, the system sets:

```python
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
```

This ensures compatibility with Windows environments.

---

## 💡 Future Improvements

- Save FAISS index to disk  
- Add cross-encoder reranking  
- Add a UI (Flask/FastAPI)  
- Create clustering over embeddings  
- Add citation-graph-based similarity  

---
