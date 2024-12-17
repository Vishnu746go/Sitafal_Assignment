# Sitafal_Assignment
# 📄 **Chat with PDF using Retrieval-Augmented Generation (RAG) Pipeline**

## **Overview**

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to interact with semi-structured data in multiple PDF files. Using state-of-the-art **FAISS** for similarity search, **SentenceTransformer** for embeddings, and **GPT-2** (via Hugging Face Transformers) for response generation, the system answers user queries and performs comparisons between data across multiple PDFs.

---

## **Features**

1. **Data Ingestion**:
   - Extracts and processes text from multiple PDF files.
   - Splits the extracted text into chunks for better retrieval granularity.
   - Converts chunks into vector embeddings using a pre-trained SentenceTransformer model (`all-MiniLM-L6-v2`).
   - Stores the embeddings in a **FAISS vector database** for efficient similarity-based retrieval.

2. **Query Handling**:
   - Accepts natural language queries from the user.
   - Retrieves the most relevant text chunks from the PDFs using similarity search.
   - Generates detailed and context-aware responses using GPT-2 (`distilgpt2`).

3. **Comparison Queries**:
   - Processes comparison-type queries across multiple PDFs.
   - Extracts and aggregates data from relevant PDFs to generate structured comparison responses.

4. **Response Generation**:
   - Uses a retrieval-augmented prompt format to generate responses.

---

## **Technologies Used**

- **Python**: Core programming language.
- **PyPDF2**: For extracting text from PDF files.
- **LangChain**: For splitting text into chunks.
- **FAISS**: For building a vector database and performing similarity search.
- **SentenceTransformer**: For generating embeddings.
- **Hugging Face Transformers**: For GPT-2 text generation.
- **DistilGPT2**: Lightweight GPT-2 model for fast text generation.

---
