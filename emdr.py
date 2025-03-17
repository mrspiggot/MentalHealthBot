from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
#Return vectorstore for the URL
def get_vector_store(url):
  loader = WebBaseLoader(url)
  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
  chunks = text_splitter.split_documents(data)
  vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
  return vector_store

#Returns history_retriever_chain
def get_retreiver_chain(vector_store):
  llm=ChatOpenAI()
  retriever = vector_store.as_retriever()
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}"),
      ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
  ])
  history_retriver_chain = create_history_aware_retriever(llm,retriever,prompt)

  return history_retriver_chain

#Returns conversational rag
def get_conversational_rag(history_retriever_chain):
  llm = ChatOpenAI(
        model_name="gpt-4o",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
        openai_api_key=openai_api_key
    )
  answer_prompt=ChatPromptTemplate.from_messages([
      ("system","You are a skilled and accomplished EMDR therapist. You are keen to help your patirnt uncover the root cause of their fears, anxiety or depression."
                "Your name is Nicola Hills and you have fifteen years of experience in EMDR therapy."
        "Use the following examples of common questions and answers as a framework for helping your patient."
        "The initial goal is discovery of the root cause of the trauma, this is crucial, guide your questions to unearth the root cause. This is most important"
        "Using gentle, short and incremental questions get the patient to articulate why they feel the way that they do"
        " Here are Key Takeaways About These Responses"
           " Validation First: Most responses begin by validating the client’s experience—naming or reflecting the emotion, sensation, or belief."
           " Curiosity and Exploration: Therapists often invite clients to notice the body’s reactions or explore past experiences that might be contributing to current feelings."
           " EMDR-Specific Integration: References to bilateral stimulation and memory reprocessing are woven in, reflecting how an EMDR therapist might guide the client."
           " Practical Tools: Suggestions for grounding, breathing exercises, or container techniques help clients manage distressing symptoms."
           " Hope and Reassurance: Therapists remind clients of the potential for change and that experiencing temporary discomfort can be part of the healing process."
            " The responses should be a maximum of two sentences"
        "Don't rush questions, take your time and be sensitive to the patient's situation."
       "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}")
  ])

  document_chain = create_stuff_documents_chain(llm,answer_prompt)

  #create final retrieval chain
  conversational_retrieval_chain = create_retrieval_chain(history_retriever_chain,document_chain)

  return conversational_retrieval_chain

#Returns th final response
def get_response(user_input):
  history_retriever_chain = get_retreiver_chain(st.session_state.vector_store)
  conversation_rag_chain = get_conversational_rag(history_retriever_chain)
  response = conversation_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input":user_input
    })
  return response["answer"]



#Streamlit app

st.header("Chat with websites")

chat_history=[]
vector_store=[]


# Sidebar
# URL pasting in sidebar on the left
with st.sidebar:
  st.header("Paste your URL")
  website_url = st.text_input("Enter URL")

if website_url is None or website_url.strip()=="":
  st.info("Please enter a website URL")
else:
  #session state
  if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="Hi, may name is Nicola, what has brought you here today?")
    ]
   #create conversation chain
  if vector_store not in st.session_state:
      st.session_state.vector_store = get_vector_store(website_url)

  user_input=st.chat_input("Type your message here...")
  if user_input is not None and user_input.strip()!="":
    response = get_response(user_input)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

  for message in st.session_state.chat_history:
      if isinstance(message,AIMessage):
        with st.chat_message("AI"):
          st.write(message.content)
      else:
        with st.chat_message("Human"):
          st.write(message.content)



