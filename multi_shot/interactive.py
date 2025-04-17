'''
Implementing a multi-shot RAG. Take in a user's question 
and then answer it using Gemini's LLM taking into account 
Paul Graham's essays available at 
"https://paulgraham.com/articles.html".

Using Streamlit for the GUI interface.

run the script with the following command
streamlit run interactive.py
'''

from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json, os


# --- Load Environment Variables ---
load_dotenv(dotenv_path='../.env')
google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    st.error("Please set your Google API key (use environment variables or Streamlit secrets).")
    st.stop()

# --- Load and Process Essays ---
@st.cache_data(show_spinner="Loading and processing essays...")
def load_and_process_essays(file_path):
    """Loads essays from JSON, creates Document objects with metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Essay file not found at {file_path}. Please ensure the path is correct.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. Please ensure the file is valid JSON.")
        st.stop()


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

# --- Create Embeddings and Vector Store ---
# Using cache_resource for objects that shouldn't be re-serialized (like vector stores)
# hash_funcs ignores the list of chunks for hashing purposes if they are large/complex
@st.cache_resource(hash_funcs={list: lambda _: None}, show_spinner="Creating embeddings and vector store...")
def create_vector_store(_chunks, google_api_key): # Renamed chunks to _chunks to satisfy cache_resource
    """Creates embeddings and the Chroma vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = Chroma.from_documents(_chunks, embeddings)
    return vector_store

# --- Initialize Chatbot Components ---
def initialize_chatbot(google_api_key):
    """Initializes all components needed for the chatbot chain."""
    # Ensure the file path is correct relative to where you run the streamlit app
    essay_file_path = "../data/pg_essays.json" # Might need adjustment depending on execution context
    chunks = load_and_process_essays(essay_file_path)
    vector_store = create_vector_store(chunks, google_api_key)

#    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key) # Using latest 1.5 pro model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' # Explicitly set the output key for memory
    )
    # -------------------------------

    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks, could potentially be from the same essay

    # --- Custom Prompt for Tech Advisor Persona ---
    template = """As a helpful and knowledgeable tech advisor, please provide guidance based on the following context. Avoid explicitly stating "According to essay X..." or similar phrases. Instead, synthesize the information into a coherent and advisory response. Use the context to answer the user's question directly and provide actionable insights where possible.

    Context:
    {context}

    Question: {question}"""
    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])


    # Create the conversational chain
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True, # We want source documents back
        combine_docs_chain_kwargs={'prompt': QA_PROMPT},	#pass custom prompt
        verbose=False # Set to True for debugging chain steps
    )
    return chat_chain, memory # Return memory if needed elsewhere, though chain manages it

# --- Format Source Documents for Display ---
def format_source_documents(source_documents):
    """Formats source documents into markdown links."""
    if not source_documents:
        return ""
    sources = "Relevant essays:\n"
    seen_sources = set()	#make sure no replicates
    formatted_sources = []
    for doc in source_documents:
        source_url = doc.metadata.get('source', '#')
        if source_url not in seen_sources:
             formatted_sources.append(f"- [{doc.metadata.get('title', 'Unknown Title')}]({source_url})")
             seen_sources.add(source_url)

    return sources + "\n".join(formatted_sources)

# --- Streamlit Interface ---
st.title("Paul Graham Essay Chatbot")
st.caption("Ask questions whose answers are informed by Paul Graham's essays.")

# Initialize chat chain and messages in session state if they don't exist
if "chat_chain" not in st.session_state:
    st.session_state["chat_chain"], _ = initialize_chatbot(google_api_key) # Don't need to store memory separately now
    st.session_state["messages"] = []


# Display previous chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new chat input
if prompt := st.chat_input("Ask Paul Graham..."):
    # Add user message to state and display it
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query through the chat chain and display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Get the result from the chain (includes 'answer' and 'source_documents')
        # The chain automatically uses and updates the memory object configured within it
        #result = st.session_state["chat_chain"]({"question": prompt})
        result = st.session_state["chat_chain"].invoke({"question": prompt})

        # Extract answer and source documents
        answer = result.get('answer', 'Sorry, I could not generate a response.')
        source_documents = result.get("source_documents", [])

        # Display the main answer (simulating typing)
        # Note: For true token-by-token streaming, you'd need a streaming callback handler
        # This simulates typing word-by-word from the complete answer.
        words = answer.split()
        for i in range(len(words)):
            full_response += words[i] + " "
            response_placeholder.markdown(full_response + "▌")
            # Add a small delay here if needed for better visual effect:
            # import time
            # time.sleep(0.05)
        response_placeholder.markdown(full_response) # Final response without cursor

        # Format and display source documents below the answer
        sources_markdown = format_source_documents(source_documents)
        if sources_markdown:
            st.markdown("---") # Add a separator
            st.markdown(sources_markdown)

        # Add the full assistant response (answer + sources) to session state
        assistant_content = full_response
        if sources_markdown:
            assistant_content += "\n\n---\n" + sources_markdown # Append formatted sources

        st.session_state["messages"].append({
            "role": "assistant",
            "content": assistant_content
        })
