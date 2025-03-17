import time
import threading
import queue

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


# --- Custom Callback Handler Using a Queue ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, token_queue):
        self.token_queue = token_queue
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        # Append token to text and put cumulative text into the queue.
        self.text += token
        self.token_queue.put(self.text)


# --- Helper Functions ---

def get_vector_store(url):
    """Return a vectorstore for the provided URL by loading and splitting the content."""
    loader = WebBaseLoader(url)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vector_store


def get_retreiver_chain(vector_store):
    """Return a history-aware retriever chain (non-streaming, for retrieval)."""
    llm = ChatOpenAI()  # No streaming needed for retrieval
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_retriver_chain = create_history_aware_retriever(llm, retriever, prompt)
    return history_retriver_chain


def get_conversational_rag(history_retriever_chain, callback_handler):
    """Return the conversational retrieval-augmented generation chain with streaming enabled."""
    # Enable streaming on the LLM and pass in our custom callback handler.
    llm = ChatOpenAI(streaming=True, callbacks=[callback_handler])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a skilled and accomplished EMDR therapist. You are keen to help your patient uncover the root cause of their fears, anxiety or depression. "
         "Your name is Nicola Hills and you have fifteen years of experience in EMDR therapy. "
         "Use the following examples of common questions and answers as a framework for helping your patient. "
         "The initial goal is discovery of the root cause of the trauma, this is crucial, guide your questions to unearth the root cause. This is most important. "
         "Using gentle, short and incremental questions get the patient to articulate why they feel the way that they do. "
         "Key Takeaways: Validation First, Curiosity and Exploration, EMDR-Specific Integration, Practical Tools, and Hope and Reassurance. "
         "The responses should be a maximum of two sentences. "
         "Don't rush questions, take your time and be sensitive to the patient's situation. "
         "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    return conversational_retrieval_chain


def get_response(user_input):
    """
    Run the chain in a background thread while streaming tokens via a queue.
    The main thread polls the queue and updates the UI placeholder.
    """
    token_queue = queue.Queue()
    streamlit_handler = StreamlitCallbackHandler(token_queue)

    placeholder = st.empty()  # Placeholder for streaming output

    # Capture vector_store and chat_history from session_state in the main thread.
    vector_store_local = st.session_state.vector_store
    chat_history_local = st.session_state.chat_history

    def run_chain():
        history_retriever_chain = get_retreiver_chain(vector_store_local)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain, streamlit_handler)
        conversation_rag_chain.invoke({
            "chat_history": chat_history_local,
            "input": user_input
        })

    # Run the chain in a background thread.
    thread = threading.Thread(target=run_chain)
    thread.start()

    full_text = ""
    # Poll the queue while the background thread is alive or tokens remain.
    while thread.is_alive() or not token_queue.empty():
        try:
            full_text = token_queue.get(timeout=0.1)
            placeholder.markdown(full_text)
        except queue.Empty:
            pass
        time.sleep(0.1)  # Small delay to allow UI updates
    thread.join()
    return full_text


# --- Streamlit App ---

st.header("Chat with websites")

# Initialize session state if not already set.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, my name is Nicola. What has brought you here today?")
    ]

# Sidebar: URL input.
with st.sidebar:
    st.header("Paste your URL")
    website_url = st.text_input("Enter URL")

if website_url is None or website_url.strip() == "":
    st.info("Please enter a website URL")
else:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store(website_url)

    user_input = st.chat_input("Type your message here...")
    if user_input is not None and user_input.strip() != "":
        # Append and display the human message.
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("Human"):
            st.write(user_input)
        # Display the AI's streaming response.
        with st.chat_message("AI"):
            response = get_response(user_input)
        st.session_state.chat_history.append(AIMessage(content=response))

    # Render the full conversation from session state.
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("Human"):
                st.write(message.content)
