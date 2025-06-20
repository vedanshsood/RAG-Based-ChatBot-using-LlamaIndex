from dotenv import load_dotenv
load_dotenv()
import os
import torch
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

os.environ["PINECONE_API_KEY"] = "pcsk_5MuNo8_B963pXPrNXk5rjmqzHHbSgmrpoie2tmBpm7McA1r3DxWDuEsBHERgMWBnpYp7Ri"
os.environ["PINECONE_INDEX_NAME"] = "chatbot"
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"

pc = Pinecone(api_key= os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True
)
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"
    prompt += "<|assistant|>\n"
    return prompt
llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    messages_to_prompt=messages_to_prompt,
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.3, "do_sample": True},
    device_map="auto"
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 256
def build_agent():
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, show_progress=True)

    query_engine = index.as_query_engine()
    query_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="document_retriever",
            description="Fetches data from uploaded files."
        )
    )

    memory = ChatMemoryBuffer.from_defaults(llm=llm, chat_history=[])
    agent = ReActAgent.from_tools(
        tools=[query_tool],
        llm=llm,
        memory=memory,
        verbose=False,
        max_iterations=5,
        system_prompt="""You are a helpful, friendly, and professional AI assistant..."""
    )
    return agent
agent_instance = build_agent()

def ask_bot(message: str) -> str:
    try:
        response = agent_instance.chat(message)
    except Exception as e:
        agent_instance.memory.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error: {str(e)}"