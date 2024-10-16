import os
import getpass
import openai
import nest_asyncio
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.memory import ChatMemoryBuffer
from flask import render_template


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

app.template_folder = 'templates'
# Load and process data
with open("C:/Users/gupta/Manipal Hackathon/Business DIrectory/MSME_final.txt", 'r', encoding='utf-8') as file:
    content = file.read()

chunks = [chunk.strip() for chunk in content.split('--------------------------------------') if chunk.strip()]

# Set up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-proj-RSCEG593QOTh03shUM80T3BlbkFJHrJETp1Lk7beobdoUWY6")

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
chat_engine1 = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as give"
        " detailed explanations of different MSME schemes"
    ),
)

chat_engine2 = index.as_chat_engine(
    chat_mode="context",
    system_prompt=(
        "You are a chatbot, able to have normal interactions, as well as give"
        " detailed explanations of different MSME schemes in the form of:"
        " scheme description, nature of assistance, how to apply and who can apply"
    ),
)

@app.route('/')
def home():
    return render_template("index.html");



@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')  # Get the message sent by the frontend
    conversation_count = data.get('conversation_count', 0)

    # Initialize the response
    if conversation_count == 0:  # If it's the first message, send the initial prompt
        initial_response = "Rural SMEs"
    else:
        # Process user messages
        initial_response = message

    # Use chat_engine2 to generate a response
    chat_response = chat_engine2.chat(initial_response)

    response = {
        'response': str(chat_response),
        'conversation_count': conversation_count + 1
    }
    
    return jsonify(response)

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    try:

        initial_topic = "Credit Guarantee Scheme"
        
        # Generate an initial message about the Credit Guarantee Scheme
        initial_message = initial_topic
        response = chat_engine2.chat(initial_message)
        
        return jsonify({
            'message': str(response),
            'conversation_count': 1,  # Set to 1 as we've just had the first interaction
            'initial_topic': initial_topic
        })
    except Exception as e:
        print("Flask error")
        app.logger.error(f"Error in reset_conversation: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

# Add this to your existing JavaScript in index.html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
