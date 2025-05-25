# 🧠 Local Document-Aware Chatbot (Flask + LlamaIndex + Llama.cpp)

This is a lightweight chatbot web app that runs **fully offline** using [LlamaIndex](https://github.com/jerryjliu/llama_index), [Flask](https://flask.palletsprojects.com/), and [llama.cpp](https://github.com/ggerganov/llama.cpp). You can chat with uploaded documents (PDF or `.txt`) using local LLMs like TinyLLaMA, Mistral, etc.

## ✅ Features

- 💬 Simple web-based chat UI built with Flask
- 🧾 Upload PDFs or `.txt` files as context
- 🤖 Local LLM response generation (no internet/API keys required)
- 💡 Remembers chat history in session

---

## 📦 Requirements

- Python 3.10 or higher
- A compatible `.gguf` model (e.g., TinyLLaMA or Mistral)
- `llama-cpp-python`, `llama-index`, `Flask`, and others

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗂 Directory Structure

```
chatbot/
├── app.py                # Flask backend
├── templates/
│   └── chat.html         # Chat UI
├── static/
│   └── style.css         # Optional styling
├── uploads/              # Uploaded user documents
├── models/               # Local LLM GGUF model files
└── README.md
```

---

## ⚙️ Configuration

### 1. Download a GGUF Model

Download a quantized `.gguf` model compatible with `llama.cpp`.  
Examples:
- [TinyLLaMA](https://huggingface.co/cmp-nct/tiny-llama-gguf)
- [Mistral](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

Place the model file in the `models/` directory and update the model path in `app.py`:

```python
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
```

### 2. Run the App

```bash
python app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000)

---

## 📂 Uploading Files

- You can upload `.pdf` and `.txt` files to be used as context.
- These files are processed and used to answer queries contextually.

---

## ❌ No API Key Needed

This chatbot uses **only local models** — no need for OpenAI or any other API key.

---

## 🔧 Future Ideas

- Add chat session memory with file-based storage
- Improve formatting and UI
- Switch to other local embedding models (like `BGE` or `MiniLM`)

---

## 🧑‍💻 Author

Built with ❤️ and LLMs by [Your Name].
