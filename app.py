import os
from flask import Flask, request, render_template, redirect
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === Config ===
UPLOAD_DIR = "uploaded_files"
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load LLM ===
def load_llm():
    return LlamaCPP(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_new_tokens=256,
        context_window=2048,
        verbose=True,
    )

# === Load documents & create query engine ===
def create_query_engine():
    if not os.listdir(UPLOAD_DIR):
        return None

    documents = SimpleDirectoryReader(UPLOAD_DIR).load_data()

    llm = load_llm()
    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Local embedding

    # Set global settings
    Settings.llm = llm
    Settings.node_parser = parser
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

# === Flask App ===
app = Flask(__name__)
chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history

    if request.method == "POST":
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if uploaded_file.filename != "":
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
                uploaded_file.save(file_path)

        user_input = request.form.get("message")
        if user_input:
            # Check for greetings
            greetings = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]
            if user_input.strip().lower() in greetings:
                response_text = "Hello! How can I help you today?"
            else:
                llm = load_llm()
                query_engine = create_query_engine()
                if query_engine:
                    response_text = query_engine.query(user_input).response
                else:
                    response_text = llm.complete(user_input).text

            chat_history.append(("You", user_input))
            chat_history.append(("Bot", str(response_text)))

        return redirect("/")

    return render_template("chat.html", chat_history=chat_history)

# === Run app ===
if __name__ == "__main__":
    app.run(debug=True)
