 # Project: "Chat with your PDF" - A Mini RAG System
 
 This project is a command-line tool that allows you to have a conversation with your PDF documents. It implements a complete Retrieval-Augmented Generation (RAG) pipeline from scratch, demonstrating a practical application of modern large language models (LLMs).
 
 ---
 
 ## 🎯 Motivation
 
 As an aspiring AI/ML Engineer, I identified a critical skill gap in my portfolio: hands-on experience with LLM APIs and modern architectures like RAG. This project was built to bridge that gap. It showcases the ability to engineer a complete AI system that goes beyond simple API calls, involving data preprocessing, vector embeddings, similarity search, and sophisticated prompt engineering. This repository serves as a demonstration of both technical depth and the ability to build production-ready AI tools.
 
 ---
 
 ## 🏛️ Architecture: The RAG Pipeline
 
 The application follows a classic Retrieval-Augmented Generation workflow. This approach enhances the LLM's knowledge by providing it with relevant context from a specific document, reducing hallucinations and enabling it to answer questions about data it wasn't trained on.
 
 #### 1. Indexing Pipeline
 The `index` command processes and stores the knowledge from a PDF.
 
 `[📄 Your PDF]` -> **1. Parse & Chunk** -> `[Text Chunks]` -> **2. Embed** -> `[Vector Embeddings]` -> **3. Index & Store** -> `[💾 FAISS Index]` + `[💾 Chunks File]`
 
 #### 2. Chat Pipeline
 The `chat` command uses the stored knowledge to answer questions.
 
 `[❓ Your Question]` -> **1. Embed** -> `[Query Vector]` -> **2. Search** -> `[Relevant Chunks]` -> **3. Prompt LLM** -> `[🤖 Final Answer]`
 
 ---
 
 ## 🛠️ Tech Stack
 
 * **Python 3.8+**
 * **Document Parsing**: `PyMuPDF` for robust PDF text extraction.
 * **Embeddings**: `sentence-transformers` (using the `all-MiniLM-L6-v2` model from Hugging Face) for generating semantic vector embeddings.
 * **Vector Store**: `FAISS` (Facebook AI Similarity Search) for efficient similarity search in the vector space.
 * **LLM**: Google Gemini (`gemini-1.5-flash`) accessed via a direct REST API call.
 * **CLI Framework**: `Click` for creating a clean and user-friendly command-line interface.
 
 ---
 
 ## 🚀 Setup and Installation
 
 Follow these steps to set up and run the project locally.
 
 1.  **Clone the Repository**
     ```bash
     git clone [https://github.com/your-username/chat-with-your-pdf.git](https://github.com/your-username/chat-with-your-pdf.git)
     cd chat-with-your-pdf
     ```
 
 2.  **Create and Activate a Virtual Environment**
     ```bash
     # Create the environment
     python -m venv .venv
 
     # Activate it (Windows)
     .venv\Scripts\activate
     
     # Activate it (macOS/Linux)
     source .venv/bin/activate
     ```
 
 3.  **Install Dependencies**
     ```bash
     pip install -r requirements.txt
     ```
 
 4.  **Set Up Your API Key**
     * Create a `.env` file in the root directory of the project.
     * Add your Google Gemini API key to the file:
         ```
         GOOGLE_API_KEY="your_api_key_here"
         ```
 
 ---
 
 ## Usage
 
 The tool has two main commands: `index` and `chat`.
 
 #### 1. Index a PDF
 First, you need to process your PDF file. This will create a `faiss_index.bin` file and a `pdf_chunks.pkl` file in your directory.
 
 ```bash
 python src/cli.py index "path/to/your/document.pdf"
 ```
 
 #### 2. Chat with Your PDF
 Once the PDF is indexed, you can start asking questions.
 
 ```bash
 python src/cli.py chat
 ```
 
 The application will load the index and prompt you for questions. To end the session, simply type `quit` or `exit`.
 ```
 🚀 Chat session started (using Google Gemini). Ask questions about your PDF.
 Type 'quit' or 'exit' to end the session.
 You: What is the main idea of this document?
 Bot: ...
 ```