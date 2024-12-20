# -*- coding: utf-8 -*-
pip install sentence-transformers faiss-cpu langchain requests fitz numpy

!pip install --force-reinstall pymupdf

#Importing the libraries and packages required
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import requests
import io
import time
import os
import pickle

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Explanation:
#1.The SentenceTransformer model (all-MiniLM-L6-v2) is used to generate embeddings for textual data.
#2.This is a lightweight transformer-based model optimized for semantic similarity tasks.

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    if pdf_path.startswith("http"):  # Handle URLs
        response = requests.get(pdf_path)
        response.raise_for_status()  # Raise an exception if download fails
        pdf_data = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_data, filetype="pdf")  # Open from bytes
    else:
        doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

#Explanation:
#1.If the pdf_path starts with "http", the PDF is fetched via HTTP, converted to a byte stream, and processed.
#2.For local files, it directly reads the PDF using fitz.

# Function to chunk text into smaller pieces for embeddings
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

#Explanation:
#1.Splits the extracted text into smaller chunks (default size: 500 words).
#2.These smaller chunks ensure that the embeddings are more manageable and relevant for semantic search.

# Function to create embeddings for chunks
def create_embeddings(chunks, embeddings_cache_path="embeddings.pkl"):
    if os.path.exists(embeddings_cache_path):
        with open(embeddings_cache_path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

    embeddings = model.encode(chunks)  # Generate embeddings using SentenceTransformers model

    # Cache embeddings to a file
    with open(embeddings_cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings

#Explanation:
#1.Purpose: Converts text chunks into dense vector representations (embeddings).
#2.If embeddings are cached (stored as a .pkl file), it loads them to save computation time.
#3.Otherwise, embeddings are created using the SentenceTransformer model and stored in a pickle file for future use.


# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)  # L2 similarity
    np_embeddings = np.array(embeddings, dtype='float32')
    index.add(np_embeddings)
    return index

#Explanation:
#1.Function: store_embeddings_in_faiss(embeddings)
#2.Purpose: Adds the embeddings to a FAISS index for similarity search.
#3.Creates an IndexFlatL2 index for L2 (Euclidean distance) similarity.
#4.Converts the embeddings into a NumPy array (float32 type) and adds them to the index.


# Function to perform similarity search on embeddings
def search_embeddings(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])  # Generate embedding for the query using SentenceTransformers

    # Search the FAISS index
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    # Fetch the most relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

#Explanation:
#1.Function: search_embeddings(query, index, chunks, top_k=3)
#2.Purpose: Finds the top k text chunks most relevant to the user’s query.
#3.Steps:
#Encodes the user query into an embedding using the same SentenceTransformer model.
#Performs a similarity search on the FAISS index to find the closest embeddings to the query.
#Returns the corresponding text chunks.

# Function to generate a response (simplified here without LangChain)
def generate_response(user_query, relevant_chunks):
    context = "\n".join(relevant_chunks)  # Combine the relevant chunks
    response = f"Based on the provided context, here's the response to your query: {user_query}\n\nContext:\n{context}"
    return response

#Explanation:
#1.Function: generate_response(user_query, relevant_chunks)
#2.Purpose: Combines the most relevant text chunks into a context and formulates a response.
#The response structure includes:
#The user’s query.
#The most relevant chunks from the PDF content.

# Main pipeline function
def run_pipeline(pdf_path, user_query):
    # Step 1: Extract and chunk text
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Step 2: Create and store embeddings
    embeddings = create_embeddings(chunks)
    index = store_embeddings_in_faiss(embeddings)

    # Step 3: Retrieve relevant chunks for the query
    relevant_chunks = search_embeddings(user_query, index, chunks)

    # Step 4: Generate response
    response = generate_response(user_query, relevant_chunks)
    return response

#Explanation:
#Function: run_pipeline(pdf_path, user_query)
#1.Combines all steps into a single pipeline:
#2.Text Extraction: Extract text from the given PDF (URL or local).
#3.Text Chunking: Split the text into manageable pieces.
#4.Embedding Creation: Generate and/or load cached embeddings.
#5.Index Storage: Store embeddings in FAISS for efficient retrieval.
#6.Similarity Search: Retrieve the most relevant chunks for the user query.
#7.Response Generation: Generate a meaningful response based on the relevant chunks.

# Running the pipeline
if __name__ == "__main__":
    # Path to the PDF file you want to process
    pdf_path = "https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-presentations/tables-charts-and-graphs-with-examples-from.pdf"  # Replace with the actual path to your PDF

    # The query that you want to ask based on the content of the PDF
    user_query = "From page 2 get the exact unemployment information based on type of degree input"  # Replace with your own query

    # Run the pipeline
    response = run_pipeline(pdf_path, user_query)

    # Print the response
    print("Response:", response)

#Explanation:
#1.Input PDF:
#A remote PDF document located at https://www.hunter.cuny.edu/dolciani/....
#This PDF is downloaded, processed, and text is extracted.
#User Query:"From page 2 get the exact unemployment information based on type of degree input"
#The pipeline processes the query to fetch the relevant content from the PDF.

#Output:The system identifies and retrieves the text chunks that are semantically most relevant to the query and includes them in the response.

"""**OUTCOMES:**

Use Case

This pipeline can be used for:

1.Extracting insights from large PDF documents.

2.Building question-answering systems for reports, research papers, or manuals.

3.Automating document analysis tasks for various industries.
"""