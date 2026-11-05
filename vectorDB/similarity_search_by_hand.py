"""
=============================================================================
Vector Similarity Search "By Hand"
=============================================================================
This script demonstrates:
1. Pure Python math from scratch (Dot Product, Norm, Cosine Similarity, L2 Distance)
2. Real-world vector search using sentence embeddings and matrix operations
=============================================================================
"""

import math
import numpy as np

# =============================================================================
# PART 1: UNDERSTANDING THE MATH "BY HAND" (NO LIBRARIES)
# =============================================================================

print("=" * 60)
print("PART 1: THE MATHEMATICAL FOUNDATIONS BY HAND")
print("=" * 60)

# Let's take two simple 2D vectors to see the math clearly
# Vector A: pointing towards [3, 4]
# Vector B: pointing towards [6, 8]  (same direction as A, just 2x longer)
# Vector C: pointing towards [-4, 3] (perpendicular / 90 degrees to A)
vec_a = [3.0, 4.0]
vec_b = [6.0, 8.0]
vec_c = [-4.0, 3.0]

def manual_vector_length(v):
    """
    Computes the L2 Norm (length/magnitude) of a vector:
    ||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)
    """
    sum_of_squares = sum(x ** 2 for x in v)
    return math.sqrt(sum_of_squares)

def manual_dot_product(v1, v2):
    """
    Computes the Dot Product between two vectors:
    v1 . v2 = (v1[0]*v2[0]) + (v1[1]*v2[1]) + ...
    """
    return sum(x * y for x, y in zip(v1, v2))

def manual_normalize(v):
    """
    Scales a vector so its length is exactly 1.0 (Unit Vector):
    unit_v = v / ||v||
    """
    length = manual_vector_length(v)
    if length == 0:
        return v
    return [x / length for x in v]

def manual_cosine_similarity(v1, v2):
    """
    Cosine Similarity = (v1 . v2) / (||v1|| * ||v2||)
    Range: -1.0 (opposite) to +1.0 (identical direction)
    """
    dot = manual_dot_product(v1, v2)
    norm1 = manual_vector_length(v1)
    norm2 = manual_vector_length(v2)
    return dot / (norm1 * norm2)

def manual_l2_distance(v1, v2):
    """
    Euclidean (L2) Distance = straight-line distance between two points:
    sqrt((v1[0] - v2[0])^2 + (v1[1] - v2[1])^2 + ...)
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

print(f"Vector A: {vec_a}")
print(f"  Length of A: {manual_vector_length(vec_a):.2f}")  # sqrt(3^2 + 4^2) = 5.0
print(f"  Normalized A: {manual_normalize(vec_a)}")          # [3/5, 4/5] = [0.6, 0.8]

print(f"\nVector B: {vec_b}")
print(f"  Length of B: {manual_vector_length(vec_b):.2f}")  # sqrt(6^2 + 8^2) = 10.0
print(f"  Normalized B: {manual_normalize(vec_b)}")          # [6/10, 8/10] = [0.6, 0.8]

# Notice: A and B have the EXACT same normalized vector!
# Because Cosine Similarity only cares about direction, sim(A, B) = 1.0!
print(f"\nCosine Similarity between A and B (Same direction): {manual_cosine_similarity(vec_a, vec_b):.4f}")
print(f"Cosine Similarity between A and C (90 degrees / orthogonal): {manual_cosine_similarity(vec_a, vec_c):.4f}")
print(f"L2 Distance between A and B (Physical distance): {manual_l2_distance(vec_a, vec_b):.4f}")

# The Big Insight:
# If vectors are already normalized (length = 1), Dot Product == Cosine Similarity!
norm_a = manual_normalize(vec_a)
norm_b = manual_normalize(vec_b)
print(f"\nDot product of normalized A and normalized B: {manual_dot_product(norm_a, norm_b):.4f}")
print("-> Exactly equal to Cosine Similarity!")

print("\n" + "=" * 60)
print("PART 2: REAL-WORLD SEMANTIC SEARCH (THE LAB TASK)")
print("=" * 60)

# Our database of documents
documents = [
    "The sales team exceeded their quarterly targets by twenty percent.",
    "Bugs introduced by the intern had to be squashed by the lead developer.",
    "A new coffee machine was installed in the office kitchen on the second floor.",
    "The marketing campaign generated thousands of new leads on social media.",
]

# The user's query
query = "I want to learn from experianced developers."

print(f"Query: \"{query}\"\n")
print("Documents in database:")
for i, doc in enumerate(documents):
    print(f"  [{i}] {doc}")

print("\n--- Running Embedding & Vector Search ---")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    
    # 1. Load an open-source, lightweight embedding model
    print("\nLoading sentence transformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Embed the documents
    doc_embeddings = model.encode(documents)
    # Normalize document embeddings manually using PyTorch (like in the lab)
    normalized_embeddings_manual = torch.nn.functional.normalize(
        torch.from_numpy(doc_embeddings)
    ).numpy()
    
    # STEP 1: Embed the query
    query_embedding = model.encode([query])
    print(f"\nStep 1: Query encoded into vector of shape: {query_embedding.shape}")
    
    # STEP 2: Normalize the query embedding
    normalized_query_embedding = torch.nn.functional.normalize(
        torch.from_numpy(query_embedding)
    ).numpy()
    print(f"Step 2: Normalized query embedding. Length is now: {np.linalg.norm(normalized_query_embedding[0]):.4f}")
    
    # STEP 3: Calculate cosine similarity using dot product (@ operator)
    # Shape of normalized_embeddings_manual: (4, 384)
    # Shape of normalized_query_embedding.T: (384, 1)
    # Result: (4, 1) matrix of similarity scores
    cosine_similarity_q3 = normalized_embeddings_manual @ normalized_query_embedding.T
    print("\nStep 3: Cosine Similarity Scores for each document:")
    for i, score in enumerate(cosine_similarity_q3.flatten()):
        print(f"  Doc [{i}] score: {score:.4f} -> \"{documents[i][:45]}...\"")
        
    # STEP 4: Find the position of the highest similarity score
    highest_cossim_position = cosine_similarity_q3.argmax()
    print(f"\nStep 4: Highest score is at index: {highest_cossim_position}")
    
    # STEP 5: Retrieve the winning document
    best_doc = documents[highest_cossim_position]
    print(f"\nStep 5: Retrieved Document:")
    print(f"  >>> \"{best_doc}\" <<<")
    print(f"  Similarity Score: {cosine_similarity_q3[highest_cossim_position][0]:.4f}")

except ImportError:
    print("\n[Note] 'sentence-transformers' or 'torch' is not installed yet.")
    print("Run: pip install sentence-transformers torch numpy")
    print("Then re-run this script to see the real AI model in action!")

