import faiss
import numpy as np
import os
from typing import List, Optional

def create_and_save_index(embeddings: np.ndarray, file_path: str = "faiss_index.bin"):
    """
    Creates a FAISS index from embeddings and saves it to a file.

    Args:
        embeddings: A numpy array of text embeddings.
        file_path: The path where the FAISS index file will be saved.
    """
    if embeddings.size == 0:
        print("Embeddings array is empty. No index will be created.")
        return

    # The dimension of our vectors is the number of columns in our embeddings array.
    dimension = embeddings.shape[1]
    
    # We are using the IndexFlatL2, a basic but effective index for L2 distance (Euclidean distance).
    print(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    
    # Add the embeddings to the index.
    index.add(embeddings)
    
    print(f"Index created with {index.ntotal} vectors.")
    
    # Save the index to a file.
    faiss.write_index(index, file_path)
    print(f"Index saved to {file_path}")

def load_index(file_path: str = "faiss_index.bin") -> Optional[faiss.Index]:
    """
    Loads a FAISS index from a file.

    Args:
        file_path: The path to the FAISS index file.

    Returns:
        The loaded FAISS index, or None if the file doesn't exist.
    """
    if not os.path.exists(file_path):
        print(f"Index file not found at {file_path}")
        return None
    
    print(f"Loading index from {file_path}...")
    index = faiss.read_index(file_path)
    print("Index loaded successfully.")
    return index