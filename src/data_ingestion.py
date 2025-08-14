import fitz  # PyMuPDF library
from typing import List

def process_pdf(file_path: str) -> List[str]:
    """
    Extracts text from a PDF and splits it into chunks using PyMuPDF.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A list of text chunks.
    """
    print(f"Processing PDF file: {file_path}...")
    
    # 1. Extract text from PDF
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except Exception as e:
        # PyMuPDF can raise a RuntimeError for corrupted files
        print(f"An error occurred while reading the PDF: {e}")
        return []

    if not text:
        print("Warning: No text could be extracted from the PDF.")
        return []
        
    print(f"Extracted {len(text)} characters from the PDF.")

    # 2. Split text into chunks
    chunks = chunk_text(text)
    
    print(f"Split text into {len(chunks)} chunks.")
    return chunks

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits a long text into smaller, overlapping chunks.
    """
    if not text:
        return []
        
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    
    return chunks

# --- This block allows to test the module directly ---
if __name__ == '__main__':
    sample_pdf_path = "1706.03762.pdf" 
    
    chunks = process_pdf(sample_pdf_path)
    
    if chunks:
        print("\n--- Sample Chunk ---")
        print(chunks[0])
        print("\n--------------------")
        print(f"Total chunks created: {len(chunks)}")