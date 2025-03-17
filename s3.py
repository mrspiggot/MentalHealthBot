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
from dotenv import load_dotenv

import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Hardcoded URL for the document.
HARD_CODED_URL = "https://my.clevelandclinic.org/health/treatments/22641-emdr-therapy"
# ---------------------------

# --- Custom Callback Handler Using a Queue ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, token_queue):
        self.token_queue = token_queue
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.token_queue.put(self.text)

# --- Helper Functions ---
def get_vector_store(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vector_store

def get_retreiver_chain(vector_store):
    llm = ChatOpenAI()  # Non-streaming for retrieval
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_retriver_chain = create_history_aware_retriever(llm, retriever, prompt)
    return history_retriver_chain

def get_conversational_rag(history_retriever_chain, callback_handler):
    llm = ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[callback_handler])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a skilled and accomplished EMDR therapist. You are keen to help your patient uncover the root cause of their fears, anxiety or depression. "
         "Your name is Nicola Hills and you have fifteen years of experience in EMDR therapy. "
         "Use the following examples of common questions and answers as a framework for helping your patient. "
         "The initial goal is discovery of the root cause of the trauma, this is crucial, guide your questions to unearth the root cause. This is most important. "
         "Using gentle, short and incremental questions get the patient to articulate why they feel the way that they do. "
         "Key Takeaways: Validation First, Curiosity and Exploration, EMDR-Specific Integration, Practical Tools, and Hope and Reassurance. "
         "The responses should be a maximum of two sentences. They should always be either a question or a request for the patient to do something."
         "Don't rush questions, take your time and be sensitive to the patient's situation. "
         "Answer the user's questions based on the below context:\n\n{context}"
         "Be sure to guide your questions to get to the root cause of the patient's issues so that the next phase of therapy can be conducted."
         "You need to get to 'WHY' the patient feels the way they do and then provide a thorough treatment for that 'WHY'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain, document_chain)
    return conversational_retrieval_chain

def get_response(user_input):
    token_queue = queue.Queue()
    streamlit_handler = StreamlitCallbackHandler(token_queue)

    # Capture current session data in local variables.
    vector_store_local = st.session_state.vector_store
    chat_history_local = st.session_state.chat_history

    def run_chain():
        history_retriever_chain = get_retreiver_chain(vector_store_local)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain, streamlit_handler)
        conversation_rag_chain.invoke({
            "chat_history": chat_history_local,
            "input": user_input
        })

    thread = threading.Thread(target=run_chain)
    thread.start()

    full_text = ""
    placeholder = st.empty()  # For streaming output.
    while thread.is_alive() or not token_queue.empty():
        try:
            full_text = token_queue.get(timeout=0.1)
            placeholder.markdown(full_text)
        except queue.Empty:
            pass
        time.sleep(0.1)
    thread.join()
    return full_text

# ---------------------------
# Streamlit App
# ---------------------------
st.title("Hi. I'm Nicola, how can I help you today?")

# Sidebar: Display EMDR Therapy title, subtitle, and logo.
with st.sidebar:
    st.image("ee-logo-4.png", width=300)
    st.title("Executive EMDR")
    st.markdown("<small>Powered by Lucidate</small>", unsafe_allow_html=True)
    st.image("color_lucidate.png", width=75)

# Initialize session state if not already set.
if "chat_history" not in st.session_state:
    # Start with an empty conversation.
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store(HARD_CODED_URL)

# Chat container: Render the entire conversation in one container.
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="nicola.jpeg"):
                st.markdown(msg.content)
        else:
            with st.chat_message("user"):
                st.markdown(msg.content)

# st.chat_input is anchored at the bottom.
user_input = st.chat_input("Type your message here...")

if user_input and user_input.strip() != "":
    # Append and display the user's message.
    new_user_msg = HumanMessage(content=user_input)
    st.session_state.chat_history.append(new_user_msg)
    with st.chat_message("user"):
        st.markdown(user_input)
    # Generate and stream the AI's response.
    with st.chat_message("assistant", avatar="nicola.jpeg"):
        response = get_response(user_input)
    new_ai_msg = AIMessage(content=response)
    st.session_state.chat_history.append(new_ai_msg)

# Auto-scroll to the bottom using JavaScript.
st.markdown(
    """
    <script>
    window.addEventListener('load', function() {
        const chatContainers = window.parent.document.querySelectorAll('[data-baseweb="chat"]');
        if (chatContainers.length > 0) {
            const container = chatContainers[chatContainers.length - 1];
            container.scrollTop = container.scrollHeight;
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)
