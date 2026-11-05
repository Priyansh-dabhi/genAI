# Vector Database & Similarity Search From Scratch

Welcome! This project breaks down how **Vector Similarity Search** works behind the scenes, using both **pure manual Python code** and modern AI libraries (`sentence-transformers`, `torch`, `numpy`).

---

## 🧠 Core Concepts in Simple Words

### 1. What is an Embedding?
Computers cannot understand human meaning directly. An **Embedding Model** transforms sentences into a list of floating-point numbers (a vector).
- Similar sentences get vectors that point in almost the exact same direction.
- Unrelated sentences point in completely different directions.

### 2. The 3 Similarity Metrics Compared

| Metric | Intuition | Formula | Range | When to Use |
| :--- | :--- | :--- | :--- | :--- |
| **Dot Product** | Measures alignment + length | $\sum (A_i \times B_i)$ | $(-\infty, +\infty)$ | When vector length (importance/frequency) matters |
| **Cosine Similarity** | Measures **angle** between directions (ignores length) | $\frac{A \cdot B}{\|A\| \times \|B\|}$ | $[-1.0, +1.0]$ | **Most common for text search** (sentence length shouldn't penalize relevance) |
| **Euclidean (L2) Distance** | Physical straight-line distance between 2 points | $\sqrt{\sum (A_i - B_i)^2}$ | $[0, +\infty)$ | Image clustering, spatial coordinates, KNN |

---

## 🎯 The Golden Formula: Why Normalized Dot Product == Cosine Similarity

Cosine similarity formula:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

If you normalize vector $A$ and vector $B$ so their lengths $(\|A\|, \|B\|)$ are equal to $1.0$:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{1.0 \times 1.0} = A \cdot B$$

That is why in production vector databases, we **pre-normalize all vectors**, and then vector search is just **one lightning-fast matrix multiplication** (`@` in Python)!

---

## 🚀 How to Run the Code

Open PowerShell in this folder (`p:\GenAI-Projects\Vector Database`) and run:

```powershell
python similarity_search_by_hand.py
```

---

## 🔬 What `similarity_search_by_hand.py` Does

1. **Part 1: Math by Hand (No Libraries)**
   - Manual L2 Norm (length calculation using `sqrt(sum(x^2))`)
   - Manual Dot Product (`sum(a * b)`)
   - Manual Cosine Similarity
   - Manual L2 Distance
   - Verifies with simple 2D vectors `[3, 4]` and `[6, 8]`.

2. **Part 2: Real Sentence Search (The IBM Lab Exercise)**
   - Takes 4 company documents.
   - Embeds query: *"Who is responsible for a coding project and fixing others' mistakes?"*
   - Normalizes using PyTorch/NumPy.
   - Calculates `@` dot product similarity across all documents.
   - Uses `.argmax()` to find and print the winning document:
     > *"Bugs introduced by the intern had to be squashed by the lead developer."*

