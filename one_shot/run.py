'''
Implementing a one-shot RAG. Take in a user's question and 
then answer it using Gemini's LLM taking into account Paul
Graham's essays available at 
"https://paulgraham.com/articles.html".


'''

import requests
from dotenv import load_dotenv
import json, os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


def create_vector_store(_chunks, google_api_key): # Renamed chunks to _chunks to satisfy cache_resource
    """Creates embeddings and the Chroma vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = Chroma.from_documents(_chunks, embeddings)
    return vector_store


# --- Load and Process Essays ---
def load_and_process_essays(file_path):
    """Loads essays from JSON, creates Document objects with metadata."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for item in data:
        content = item.get('content', '') # Use .get for safety
        title = item.get('title', 'Untitled') # Use .get for safety
        url = item.get('url', '')
        metadata = {"title": title, "source": url} 
        documents.append(Document(page_content=content, metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


# Function to get the response from the Gemini API
def answer_with_pg_essays(question):
    # Get the Gemini API key from an environment variable for security
    load_dotenv(dotenv_path='../.env')
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable not set."


    essay_file_path = "../data/pg_essays.json" 
    chunks = load_and_process_essays(essay_file_path)
    vector_store = create_vector_store(chunks, api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks, could potentially be from the same essay.
    relevant_essays = retriever.invoke(question)  # <- this is the missing piece


    # Create the prompt with the relevant essays
#    context = "\n\n".join([doc.page_content for doc in relevant_essays])	#no title in this input
    context = "\n\n".join([f"Title: {essay.metadata['title']}\nContent: {essay.page_content}" for essay in relevant_essays])
    prompt = f"Based on the relevant esssays provided, answer the question:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Define your Gemini API endpoint and headers
    #api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"  # no longer available 
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
    }
    params = {
        "key": api_key  # Include the API key as a query parameter
    }
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 1500,  # Adjust based on the desired response length
        }
    }

    try:
        # Send the request to Gemini
        response = requests.post(api_url, params=params, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

        # Extract and return the answer from the response
        response_json = response.json()
        if 'candidates' in response_json and response_json['candidates']:
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Error: No response content found in Gemini API response."

    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except ValueError as e:
        return f"Error: Could not decode JSON response - {e}"
    except KeyError as e:
        return f"Error: Missing key in JSON response - {e}"



if __name__ == "__main__":
    question = input("Enter a question that you would like Paul Graham (based on his essays) to answer: ")
    answer = answer_with_pg_essays(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
