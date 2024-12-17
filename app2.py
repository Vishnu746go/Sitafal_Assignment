import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Step 2: Split text into chunks
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Step 3: Generate embeddings using SentenceTransformer
def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model
    embeddings = model.encode(chunks)
    return embeddings

# Step 4: Create FAISS index
def create_faiss_index(embeddings):
    embedding_array = embeddings.astype('float32')
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    return index

# Main data ingestion function (handles multiple PDFs)
def process_pdfs(pdf_paths):
    faiss_indexes = {}
    chunks = {}
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)
        chunk_list = chunk_text(text)
        chunk_embeddings = generate_embeddings(chunk_list)
        faiss_index = create_faiss_index(chunk_embeddings)
        faiss_indexes[os.path.basename(pdf_path)] = faiss_index
        chunks[os.path.basename(pdf_path)] = chunk_list
    print("FAISS indexes created successfully for all PDFs!")
    return faiss_indexes, chunks

# Initialize Hugging Face Model and Tokenizer (for LLM)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 2: Query the FAISS index from multiple PDFs (with source tracking)
def query_index(user_query, faiss_indexes, chunks, embedding_model, top_k=5):
    query_embedding = embedding_model.encode([user_query])  # Use embedding model for the query
    relevant_chunks = {}

    for pdf_name, faiss_index in faiss_indexes.items():
        distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        relevant_chunks[pdf_name] = [(chunks[pdf_name][i], pdf_name) for i in indices[0]]

    return relevant_chunks

# Step 3: Generate a response using Hugging Face
def generate_response(user_query, relevant_chunks, max_input_tokens=900, max_new_tokens=100):
    context = ""
    for pdf_name, chunk_list in relevant_chunks.items():
        context += f"From {pdf_name}:\n" + "\n\n".join([chunk[0] for chunk in chunk_list[:3]]) + "\n\n"

    input_tokens = tokenizer.encode(context, truncation=True, max_length=max_input_tokens)
    truncated_context = tokenizer.decode(input_tokens)

    prompt = f"Context:\n{truncated_context}\n\nQuestion: {user_query}\n\nAnswer:"

    response = text_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        truncation=True
    )
    return response[0]["generated_text"]

# Step 4: Main function for handling user queries
def handle_user_query(user_query, faiss_indexes, chunks, embedding_model):
    relevant_chunks = query_index(user_query, faiss_indexes, chunks, embedding_model, top_k=5)
    response = generate_response(user_query, relevant_chunks)
    return response

# Handling comparison query
def handle_comparison_query(comparison_query, faiss_indexes, chunks, embedding_model):
    comparison_terms = comparison_query.split("compare") 
    relevant_chunks = {}

    for term in comparison_terms:
        term = term.strip()
        relevant_chunks.update(query_index(term, faiss_indexes, chunks, embedding_model, top_k=5))

    comparison_data = process_comparison_data(relevant_chunks)
    comparison_response = generate_comparison_response(comparison_data)
    return comparison_response

def process_comparison_data(relevant_chunks):
    comparison_data = {}
    for pdf_name, chunks in relevant_chunks.items():
        comparison_data[pdf_name] = [chunk[0] for chunk in chunks]
    return comparison_data

def generate_comparison_response(comparison_data):
    response = "Comparison Results:\n\n"
    for pdf_name, chunks in comparison_data.items():
        response += f"From {pdf_name}:\n" + "\n".join(chunks) + "\n\n"
    return response

# Main function to handle both user and comparison queries
if __name__ == "__main__":
    pdf_paths = ["AttritonPrediction_BasePaper.pdf", "AttritionPrediction_Using_XAI_1.pdf"]  # Add multiple PDFs
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    faiss_indexes, chunks = process_pdfs(pdf_paths)
    
    # Handle User Query
    user_query = input("Enter your question: ")
    print("\nHandling user query...\n")
    response = handle_user_query(user_query, faiss_indexes, chunks, embedding_model)
    print(f"\nResponse:\n{response}")
    
    # Handle Comparison Query
    comparison_query = input("\nEnter a comparison query: ")
    print("\nHandling comparison query...\n")
    comparison_response = handle_comparison_query(comparison_query, faiss_indexes, chunks, embedding_model)
    print(f"\nComparison Response:\n{comparison_response}")
