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
model = SentenceTransformer('all-MiniLM-L6-v2')
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
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
def create_embeddings(chunks, embeddings_cache_path="embeddings.pkl"):
    if os.path.exists(embeddings_cache_path):
        with open(embeddings_cache_path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

    embeddings = model.encode(chunks)  # Generate embeddings using SentenceTransformers model

    with open(embeddings_cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings
# Function to store embeddings in FAISS
def store_embeddings_in_faiss(embeddings):
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)  # L2 similarity
    np_embeddings = np.array(embeddings, dtype='float32')
    index.add(np_embeddings)
    return index

def search_embeddings(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])  # Generate embedding for the query using SentenceTransformers

    # Search the FAISS index
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    # Fetch the most relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

# Function to generate a response (simplified here without LangChain)
def generate_response(user_query, relevant_chunks):
    context = "\n".join(relevant_chunks)  # Combine the relevant chunks
    response = f"Based on the provided context, here's the response to your query: {user_query}\n\nContext:\n{context}"
    return response
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

#Output:The system identifies and retrieves the text chunks that are semantically most relevant to the query and includes them in the response.Response: Based on the provided context, here's the response to your query: From page 2 get the exact unemployment information based on type of degree input

#Context:
#Tables, Charts, and Graphs with Examples from History, Economics, Education, Psychology, Urban Affairs and Everyday Life REVISED: MICHAEL LOLKUS 2018 Tables, Charts, and Graphs Basics We use charts and graphs to visualize data. This data can either be generated data, data gathered from an experiment, or data collected from some source. A picture tells a thousand words so it is not a surprise that many people use charts and graphs when explaining data. Types of Visual Representations of Data Table of Yearly U.S. GDP by Industry (in millions of dollars) Year 2010 2011 2012 2013 2014 2015 All Industries 26093515 27535971 28663246 29601191 30895407 31397023 Manufacturing 4992521 5581942 5841608 5953299 6047477 5829554 Finance, Insurance, Real Estate, Rental, Leasing 4522451 4618678 4797313 5031881 5339678 5597018 Arts, Entertainment, Recreation, Accommodation, and Food Service 964032 1015238 1076249 1120496 1189646 1283813 Other 15614511 16320113 16948076 17495515 18318606 18686638 Source: U.S. Bureau of Labor Statistics 19% 18% 4% 59% 2015 U.S. GDP (in millions of dollars) Manufacturing Finance, insurance, real estate, rental, and leasing Arts, entertainment, recreation, accommodation, and food services Other • The chart below is called a pie chart. It shows what percent “of the pie” a particular category occupies out of the whole. • If total GDP in 2015 is the entire pie, then manufacturing makes up 19% of that pie and finance makes up 18%. Notice that visually speaking, since 19% and 18% are so close to each other in value, their respective slices of the pie are similarly sized. Pie charts can be misleading when the slices do not correspond with the percent contribution to the whole pie. Notice the pie chart below is not very intuitive. Example from Everyday Life 19% 10% 15% 5% 26% 25% Family Budget of $31,000 Other Recreation Transportation Clothing housing Food The following chart shows how a family spends its yearly income of $31,000. How much money does this family spend on transportation? Solution The chart indicates that 15% of the income is spent on transportation. We must answer the question: 15% of $31,000 is what? Writing as an equation and solving, we get n = 0.15 x 31,000 = 4650 So the family spends $4650 on transportation yearly. 0 5 10 15 20 25 30 35 All industries Manufacturing Finance, insurance, real estate, rental, and leasing Arts, entertainment, recreation, accommodation, and food services Other Dollars 2015 GDP (in trillions of dollars) • The graph below is called a bar graph. • It shows each of the variables independent of each other, each with its own bar. • 2015 GDP for all industries was $31.397023; looking at the graph, the bar for all industries is just above $30. • One is still be able compare each variable with the other by comparing bars. • The graph below is called a line graph. It shows how a variable evolves with respect to another variable. In the line graph below, we show how GDP has evolved by year. 0 5 10 15 20 25 30 35
#1947 1950 1953 1956 1959 1962 1965 1968 1971 1974 1977 1980 1983 1986 1989 1992 1995 1998 2001 2004 2007 2010 2013 Dollars Year Yearly Total GDP (in trillions of dollars) Yearly Total GDP When to use a Line Graph, Pie Chart, or Bar Graph? We use the pie chart here to compare parts of a whole. In our example, we compared components of US GDP. The line chart is useful when you want to show how a variable changes over time. For our purposes, we used it show how GDP changed over time. Bar graphs are good for comparing different groups of variables. We used it to compare different components of US GDP. We did the same with the pie chart; depending on your purposes you may choose to use a pie chart or a bar graph. x y 0 0 1 3 2 6 3 9 4 12 5 15 6 18 7 21 8 24 • If given a table of data, we should be able to plot it. Below is some sample data; plot the data with x on the x-axis and y on the y-axis. 0 5 10 15 20 25 30 0 1 2 3 4 5 6 7 8 • Below is a plot of the data on the table from the previous slide. Notice that this plot is a straight line meaning that a linear equation must have generated this data. • What if the data is not generated by a linear equation? We can fit the data using a linear regression and use that line as an approximation to the data. Regressions are beyond the scope of this workshop. Example from Urban Affairs What kind of bar graph is this? Whose life expectancy has changed the most since 1925? In 1925, about how much longer was a woman expected to live than a man? Example from History In what years were the affiliations for Republicans and Independents the same? During what time period did the party affiliations have the most change? Example from Education What percent of the total class received grades of 72 or 77? Which grade showed the largest difference between males and females? Example from Psychology What do you notice is different in this graph than the others reviewed so far?
#1947 1950 1953 1956 1959 1962 1965 1968 1971 1974 1977 1980 1983 1986 1989 1992 1995 1998 2001 2004 2007 2010 2013 Dollars Year Yearly Total GDP (in trillions of dollars) Yearly Total GDP When to use a Line Graph, Pie Chart, or Bar Graph? We use the pie chart here to compare parts of a whole. In our example, we compared components of US GDP. The line chart is useful when you want to show how a variable changes over time. For our purposes, we used it show how GDP changed over time. Bar graphs are good for comparing different groups of variables. We used it to compare different components of US GDP. We did the same with the pie chart; depending on your purposes you may choose to use a pie chart or a bar graph. x y 0 0 1 3 2 6 3 9 4 12 5 15 6 18 7 21 8 24 • If given a table of data, we should be able to plot it. Below is some sample data; plot the data with x on the x-axis and y on the y-axis. 0 5 10 15 20 25 30 0 1 2 3 4 5 6 7 8 • Below is a plot of the data on the table from the previous slide. Notice that this plot is a straight line meaning that a linear equation must have generated this data. • What if the data is not generated by a linear equation? We can fit the data using a linear regression and use that line as an approximation to the data. Regressions are beyond the scope of this workshop. Example from Urban Affairs What kind of bar graph is this? Whose life expectancy has changed the most since 1925? In 1925, about how much longer was a woman expected to live than a man? Example from History In what years were the affiliations for Republicans and Independents the same? During what time period did the party affiliations have the most change? Example from Education What percent of the total class received grades of 72 or 77? Which grade showed the largest difference between males and females? Example from Psychology What do you notice is different in this graph than the others reviewed so far?

"""**OUTCOMES:**

Use Case

This pipeline can be used for:

1.Extracting insights from large PDF documents.

2.Building question-answering systems for reports, research papers, or manuals.

3.Automating document analysis tasks for various industries.
"""
