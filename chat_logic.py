from dotenv import load_dotenv
load_dotenv()
import os
import torch
import sqlite3
import json
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMessage
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.agent import ReActAgent  # Updated import
import re
from sqlalchemy import create_engine

# --- CUDA Check ---
if torch.cuda.is_available():
    print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA not available. Falling back to CPU.")

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect("chat_sessions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            chat_history TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- Pinecone Setup ---
os.environ["PINECONE_API_KEY"] = "pcsk_5MuNo8_B963pXPrNXk5rjmqzHHbSgmrpoie2tmBpm7McA1r3DxWDuEsBHERgMWBnpYp7Ri"
os.environ["PINECONE_INDEX_NAME"] = "chatbot"
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Qwen via Ollama ---
llm = Ollama(
    model="llama3:latest",
    temperature=0.3,
    max_tokens=1024,
    request_timeout=100.0
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.chunk_overlap = 100

# --- Global Agent Instances ---
doc_agent = None
sql_agent = None
current_session_id = None

# --- Document Tracking ---
def get_existing_document_ids():
    """Retrieve IDs of documents already indexed in Pinecone."""
    try:
        stats = pinecone_index.describe_index_stats()
        # Check if the default namespace exists and has vectors
        namespaces = stats.get('namespaces', {})
        if '' in namespaces and namespaces[''].get('vector_count', 0) > 0:
            # Fetch all IDs in the default namespace
            query_response = pinecone_index.query(
                vector=[0] * 1024,  # Dummy vector
                top_k=10000,  # Adjust based on expected number of documents
                include_metadata=False,
                namespace=''
            )
            return {match['id'] for match in query_response['matches']}
        return set()
    except Exception as e:
        print(f"Error fetching existing document IDs: {str(e)}")
        return set()

# --- Delete Embeddings ---
def delete_embeddings_for_file(filename):
    """Delete embeddings associated with a specific file from Pinecone."""
    try:
        pinecone_index.delete(filter={"filename": filename})
        print(f"Deleted embeddings for {filename} from Pinecone")
    except Exception as e:
        print(f"Error deleting embeddings for {filename}: {str(e)}")

# --- Rebuild Agents ---
def rebuild_agent(sql_file_path=None):
    global doc_agent, sql_agent
    print(f"Rebuilding agents. SQL file path: {sql_file_path}")

    # ---- 📄 Document Agent ----
    documents = SimpleDirectoryReader("docs").load_data()
    existing_ids = get_existing_document_ids()
    
    # Filter out documents that are already indexed
    new_documents = [
        doc for doc in documents
        if doc.id_ not in existing_ids
    ]
    
    if new_documents:
        print(f"Indexing {len(new_documents)} new documents")
        index = VectorStoreIndex.from_documents(
            new_documents,
            vector_store=vector_store,
            show_progress=True
        )
    else:
        print("No new documents to index. Using existing index.")
        index = VectorStoreIndex(vector_store=vector_store)

    query_engine = index.as_query_engine(similarity_top_k=5, verbose=True)

    doc_metadata = ToolMetadata(
        name="document_retriever",
        description="Fetches data from uploaded files."
    )
    doc_tool = QueryEngineTool(query_engine=query_engine, metadata=doc_metadata)
    doc_memory = ChatMemoryBuffer.from_defaults(llm=llm, chat_history=[])

    doc_agent = ReActAgent.from_tools(
        tools=[doc_tool],
        llm=llm,
        memory=doc_memory,
        verbose=True,
        max_iterations=5,
        system_prompt="""You are a document-based AI assistant. Respond only based on the uploaded document.
The document is already loaded into memory, so do not try to access files by name.
If information is missing, say 'not found in document'. Avoid assumptions."""
    )

    # ---- 🧠 SQL Agent (Only if SQL file provided) ----
    if sql_file_path:
        print(f"Initializing SQL agent with {sql_file_path}")
        sql_agent = create_sql_agent_from_sql_file(sql_file_path, llm)
    else:
        print("No SQL file provided. SQL agent not initialized.")

def create_sql_agent_from_sql_file(sql_file_path, llm):
    print(f"Connecting to database: {sql_file_path}")
    engine = create_engine(f"sqlite:///{sql_file_path}")
    sql_database = SQLDatabase(engine)
    inspector = sql_database._inspector
    tables = inspector.get_table_names()
    print(f"Detected tables: {tables}")
    if not tables:
        print("No tables found in database!")

    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=tables,
        llm=llm,
        verbose=True
    )

    sql_metadata = ToolMetadata(
        name="sql_retriever",
        description="Queries any database for information. Use this tool to answer questions by generating SQL queries based on the database schema. If data is missing, say 'not found in database'."
    )

    sql_tool = QueryEngineTool(query_engine=sql_query_engine, metadata=sql_metadata)
    sql_memory = ChatMemoryBuffer.from_defaults(llm=llm, chat_history=[])

    sql_agent = ReActAgent.from_tools(
        tools=[sql_tool],
        llm=llm,
        memory=sql_memory,
        verbose=True,
        max_iterations=20,
        system_prompt="""You are a SQL-based AI assistant. Respond only based on SQL database queries.
Use the sql_retriever tool to generate appropriate SQL queries from user questions.
For questions about customers/orders, join tables and filter. If unclear, ask. If not found, say so."""
    )

    return sql_agent

# --- Session Management ---
def create_new_session():
    global current_session_id, doc_agent, sql_agent
    if doc_agent is None or sql_agent is None:
        rebuild_agent()
    conn = sqlite3.connect("chat_sessions.db")
    cursor = conn.cursor()
    session_name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    cursor.execute(
        "INSERT INTO sessions (session_name, created_at, chat_history) VALUES (?, ?, ?)",
        (session_name, datetime.now(), json.dumps([]))
    )
    current_session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    doc_agent.memory.reset()
    if sql_agent:
        sql_agent.memory.reset()
    return current_session_id

def load_session(session_id):
    global current_session_id, doc_agent, sql_agent
    if doc_agent is None:
        rebuild_agent()

    conn = sqlite3.connect("chat_sessions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT chat_history FROM sessions WHERE session_id = ?", (session_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        current_session_id = session_id
        chat_history_json = json.loads(result[0])
        chat_history = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in chat_history_json]
        doc_agent.memory = ChatMemoryBuffer.from_defaults(llm=llm, chat_history=chat_history)
        if sql_agent:
            sql_agent.memory = ChatMemoryBuffer.from_defaults(llm=llm, chat_history=chat_history)
        return chat_history_json
    return []

def save_session(session_id, chat_history):
    if session_id is None:
        return
    serializable_history = [{"role": msg.role, "content": msg.content} for msg in chat_history]
    conn = sqlite3.connect("chat_sessions.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sessions SET chat_history = ? WHERE session_id = ?",
        (json.dumps(serializable_history), session_id)
    )
    conn.commit()
    conn.close()

def get_sessions():
    conn = sqlite3.connect("chat_sessions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, session_name, created_at FROM sessions ORDER BY created_at DESC")
    sessions = [{"id": row[0], "name": row[1], "created_at": row[2]} for row in cursor.fetchall()]
    conn.close()
    return sessions

def delete_session(session_id):
    conn = sqlite3.connect("chat_sessions.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def rename_session(session_id, new_name):
    conn = sqlite3.connect("chat_sessions.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET session_name = ? WHERE session_id = ?", (new_name, session_id))
    conn.commit()
    conn.close()

# --- Initialize Agents Only ---
rebuild_agent()

# --- Inference ---
def ask_bot(message: str) -> str:
    global current_session_id, doc_agent, sql_agent
    try:
        # --- Greeting Handling ---
        greetings = ["hello", "hi", "hey", "good day", "how are you", "what's up"]
        msg = message.strip().lower()
        if re.fullmatch(r"(hi|hello|hey|good day|how are you|what's up)[!?\.]?", msg):
            if current_session_id is None:
                current_session_id = create_new_session()
            chat_history = doc_agent.memory.get()
            chat_history.append(ChatMessage(role="user", content=message))
            chat_history.append(ChatMessage(role="assistant", content="Hello! How can I assist you today? (Documents or Database?)"))
            save_session(current_session_id, chat_history)
            return "Hello! How can I assist you today? (Documents or Database?)"

        # --- Session Init ---
        if current_session_id is None:
            current_session_id = create_new_session()

        # --- Decide Agent ---
        print(f"Processing message: {message}")
        if any(keyword in msg for keyword in ["database", "table", "sql", "customer", "order", "agent"]):
            print(f"Routing to SQL agent. sql_agent is {sql_agent is not None}")
            if sql_agent is not None:
                response = sql_agent.chat(message)
                chat_history = sql_agent.memory.get()
                print(f"SQL agent response: {response}")
            else:
                print("SQL agent is None")
                chat_history = doc_agent.memory.get()
                chat_history.append(ChatMessage(role="user", content=message))
                chat_history.append(ChatMessage(role="assistant", content="⚠️ No database loaded. Please upload a .db, .csv, or .excel file to enable SQL queries."))
                save_session(current_session_id, chat_history)
                return "⚠️ No database loaded. Please upload a .db, .csv, or .excel file to enable SQL queries."
        else:
            print("Routing to document agent")
            response = doc_agent.chat(message)
            chat_history = doc_agent.memory.get()


        # --- Save & Return ---
        save_session(current_session_id, chat_history)
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, "response"):
            return str(response.response).strip()
        return str(response).strip()

    except Exception as e:
        print(f"Error in ask_bot: {str(e)}")
        if doc_agent:
            doc_agent.memory.reset()
            save_session(current_session_id, [])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error: {str(e)}"