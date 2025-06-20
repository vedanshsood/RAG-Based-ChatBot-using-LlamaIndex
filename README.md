
# 📚 Document Chatbot Assistant

An interactive chatbot that allows users to chat with uploaded documents using a clean UI and FastAPI backend. The app includes a chat interface with a "Bot is typing..." animation, file upload, and real-time responses.

---

## 🚀 Features

- 💬 Chat interface with user and bot message bubbles
- ⏳ Typing indicator animation (dots) while bot is generating a response
- 📁 Document upload capability
- ⚡ FastAPI backend with POST endpoints for chat and file uploads
- 🎨 Stylish and responsive frontend using plain HTML, CSS, and JavaScript

---

## 📂 Project Structure

```
project-root/
│
├── chatbot app/
│   ├── main.py              # FastAPI backend logic
│   ├── templates/
│   │   └── index.html       # Frontend HTML interface
│   └── static/
│       └── style.css        # Styles for the chat UI
│
├── uploads/                 # Uploaded documents (optional)
└── README.md                # Project documentation
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document-chatbot.git
cd document-chatbot
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn python-multipart
```

You may also need:
```bash
pip install llama-index openai  # If your agent uses LlamaIndex or OpenAI APIs
```

### 4. Run the Application

```bash
uvicorn chatbot\ app.main:app --reload
```

Then open your browser at:  
👉 **http://127.0.0.1:8000**

---

## 💡 How It Works

1. Upload a document using the file input.
2. Start chatting by typing a question in the chat box.
3. The backend processes the message and returns a response.
4. A typing animation displays while the bot thinks.
5. Responses appear dynamically in a styled chat bubble.

---

## 🧪 TODO (Optional Improvements)

- [ ] Add streaming support using async generators
- [ ] Implement document parsing with embeddings
- [ ] Add conversation history memory
- [ ] Support multiple file formats (PDF, DOCX)

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Support

If you encounter issues, please open an issue or pull request. Contributions welcome!
