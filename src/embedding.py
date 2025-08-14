from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# We are choosing a specific, well-performing model from Hugging Face.
MODEL_NAME = "all-MiniLM-L6-v2"

# We initialize the model once and reuse it to save time and resources.
print("Initializing sentence-transformer model...")
model = SentenceTransformer(MODEL_NAME)
print("Model initialized.")

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text chunks.

    Args:
        texts: A list of text chunks to be embedded.

    Returns:
        A numpy array of embeddings.
    """
    if not texts or not isinstance(texts, list):
        return np.array([])

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print("Embeddings generated successfully.")
    
    return embeddings