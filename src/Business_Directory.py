import os
import openai
import nest_asyncio
import chromadb
import streamlit as st
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer

# Load and process data
with open("docs/MSME_final.txt", 'r', encoding='utf-8') as file:
    content = file.read()

chunks = [chunk.strip() for chunk in content.split('--------------------------------------') if chunk.strip()]

# Set up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

# Create nodes
nodes = [TextNode(text=chunk) for chunk in chunks]

# Set up Chroma
nest_asyncio.apply()
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart2")

# Set up vector store and index
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

# Set up chat engine
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as give"
        " detailed explanations of different MSME schemes in the form of:"
        " scheme description, nature of assistance, how to apply and who can apply"
    ),
)

# Streamlit UI
st.title("MSME Business Directory Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about MSME schemes?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat_engine.chat(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(str(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": str(response)})

# Add a button to reset the conversation
if st.button("Reset Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()
