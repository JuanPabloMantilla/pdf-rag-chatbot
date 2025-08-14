import click
import pickle
import requests  
import json      

# --- Local Imports ---
from data_ingestion import process_pdf
from embedding import generate_embeddings, model as embedding_model
from vector_store import create_and_save_index, load_index
from config import API_KEY

# --- Constants ---
CHUNKS_FILE = "pdf_chunks.pkl"
INDEX_FILE = "faiss_index.bin"
TOP_K = 3
MODEL_NAME = "gemini-1.5-flash"

if not API_KEY:
    raise ValueError("API_KEY not found. Please set the GOOGLE_API_KEY environment variable.")


@click.group()
def cli():
    """A command-line tool to chat with your PDF."""
    pass

@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True, dir_okay=False))
def index(pdf_path: str):
    """Processes a PDF file, creates embeddings, and builds a FAISS index."""
    click.echo(f"Starting to index PDF: {pdf_path}")
    chunks = process_pdf(pdf_path)
    if not chunks:
        click.echo(click.style("Failed to extract any text chunks from the PDF.", fg="red"))
        return
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    click.echo(click.style(f"Text chunks saved to {CHUNKS_FILE}", fg="green"))
    embeddings = generate_embeddings(chunks)
    if embeddings.size == 0:
        click.echo(click.style("Failed to generate embeddings.", fg="red"))
        return
    create_and_save_index(embeddings, file_path=INDEX_FILE)
    click.echo(click.style("\nPDF successfully indexed!", fg="green", bold=True))

@cli.command()
def chat():
    """Starts an interactive chat session about the indexed PDF."""
    try:
        index = load_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
    except FileNotFoundError:
        click.echo(click.style("Index not found. Please run the 'index' command first.", fg="red"))
        return
    
    click.echo(click.style("🚀 Chat session started (using Google Gemini). Ask questions about your PDF.", fg="cyan"))
    click.echo(click.style("Type 'quit' or 'exit' to end the session.", fg="cyan"))

    while True:
        question = click.prompt(click.style("You", fg="yellow", bold=True))
        if question.lower() in ['quit', 'exit']:
            click.echo(click.style("Goodbye!", fg="magenta"))
            break

        question_embedding = embedding_model.encode([question])
        distances, indices = index.search(question_embedding, TOP_K)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        # Construct the prompt for the Gemini REST API
        prompt = f"""
        Context from the document:
        ---
        {context}
        ---
        Based on the context provided above, please answer the following question.

        Question: {question}
        """

        # --- Make the direct HTTP request ---
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            response_json = response.json()
            
            # Safely extract the text from the response
            if 'candidates' in response_json and response_json['candidates']:
                first_candidate = response_json['candidates'][0]
                if 'content' in first_candidate and 'parts' in first_candidate['content'] and first_candidate['content']['parts']:
                    answer = first_candidate['content']['parts'][0]['text']
                else:
                    answer = "I could not find a text part in the response."
            else:
                answer = "No response candidates found. The content may have been blocked."

        except requests.exceptions.RequestException as e:
            answer = f"API Request Failed: {e}"
        except Exception as e:
            answer = f"An unexpected error occurred: {e}"

        click.echo(click.style("Bot:", fg="blue", bold=True) + f" {answer.strip()}")


if __name__ == '__main__':
    cli()