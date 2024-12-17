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

# Step 3: Generate embeddings
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

# Main data ingestion function
def process_pdf(pdf_path):
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting text into chunks...")
    chunks = chunk_text(text)

    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = generate_embeddings(chunks)

    print("Creating FAISS index...")
    faiss_index = create_faiss_index(embeddings)

    print("FAISS index created successfully!")
    return faiss_index, chunks

# Initialize Hugging Face Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 2: Query the FAISS index
def query_index(user_query, faiss_index, chunks, model, top_k=5):
    query_embedding = model.encode([user_query])
    distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Step 3: Generate a response using Hugging Face
def generate_response(user_query, relevant_chunks, max_input_tokens=900, max_new_tokens=100):
    """
    Generate a response using a Hugging Face model with token truncation.
    
    Args:
        user_query (str): The user's natural language question.
        relevant_chunks (list): The most relevant chunks retrieved from the FAISS index.
        max_input_tokens (int): Maximum input tokens allowed.
        max_new_tokens (int): Tokens to be generated.
        
    Returns:
        str: The generated response.
    """
    # Combine the top 2-3 relevant chunks into a single context
    context = "\n\n".join(relevant_chunks[:3])  # Take the top 3 relevant chunks
    
    # Tokenize the combined context and truncate it aggressively
    input_tokens = tokenizer.encode(context, truncation=True, max_length=max_input_tokens)
    truncated_context = tokenizer.decode(input_tokens)  # Convert back to text
    
    # Prepare the prompt
    prompt = f"Context:\n{truncated_context}\n\nQuestion: {user_query}\n\nAnswer:"
    
    # Generate the response
    response = text_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        truncation=True  # Ensure truncation is explicit
    )
    return response[0]["generated_text"]


# Step 4: Main function for handling user queries
def handle_user_query(user_query, faiss_index, chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    relevant_chunks = query_index(user_query, faiss_index, chunks, model, top_k=5)
    response = generate_response(user_query, relevant_chunks)
    return response

if __name__ == "__main__":
    pdf_path = "AttritonPrediction_BasePaper.pdf"
    faiss_index, chunks = process_pdf(pdf_path)
    user_query = "What are the key factors in employee attrition?"
    print("\nHandling user query...\n")
    response = handle_user_query(user_query, faiss_index, chunks)
    print(f"\nResponse:\n{response}")
