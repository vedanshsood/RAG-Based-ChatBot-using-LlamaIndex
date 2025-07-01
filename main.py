from fastapi import FastAPI, UploadFile, Request, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uvicorn
import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from chat_logic import ask_bot, rebuild_agent, create_new_session, load_session, get_sessions, delete_session, rename_session, delete_embeddings_for_file

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = Path("chatbot.db")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    # Rebuild agent on page load to ensure existing documents are not re-indexed
    rebuild_agent(str(DB_PATH) if DB_PATH.exists() else None)
    return Path("templates/index.html").read_text(encoding='utf-8')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = UPLOAD_DIR / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith((".db", ".csv", ".xlsx", ".xls")):
        try:
            print(f"Processing file: {file_location}")
            if file.filename.endswith(".db"):
                # Copy the uploaded .db file to chatbot.db
                shutil.copyfile(file_location, DB_PATH)
                print(f"Copied {file.filename} to {DB_PATH}")
            elif file.filename.endswith((".csv", ".xlsx", ".xls")):
                # Read the file into a pandas DataFrame
                if file.filename.endswith(".csv"):
                    df = pd.read_csv(file_location)
                else:  # .xlsx or .xls
                    df = pd.read_excel(file_location)

                # Create a SQLite engine and write the DataFrame to chatbot.db
                engine = create_engine(f"sqlite:///{DB_PATH}")
                table_name = file.filename.split(".")[0]  # Use filename (without extension) as table name
                df.to_sql(table_name, engine, if_exists="replace", index=False)
                print(f"Converted {file.filename} to table {table_name} in {DB_PATH}")

            print(f"Rebuilding agent with {DB_PATH}")
            rebuild_agent(str(DB_PATH))
            return {"message": f"{file.filename} processed and chatbot.db loaded into SQL agent"}
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return {"error": f"Processing failed: {str(e)}"}
    else:  # Handle .pdf and other documents
        print(f"Uploaded document: {file.filename}")
        rebuild_agent(str(DB_PATH) if DB_PATH.exists() else None)
        return {"message": f"{file.filename} uploaded successfully"}

@app.delete("/delete")
async def delete_file(filename: str = Form(...)):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        if filename.endswith((".db", ".csv", ".xlsx", ".xls")):
            # Delete embeddings from Pinecone
            delete_embeddings_for_file(filename)
            # If deleting the database file, also remove chatbot.db and reset SQL agent
            if filename.endswith(".db") and DB_PATH.exists():
                DB_PATH.unlink()
            rebuild_agent(None)  # Rebuild without SQL agent
        return {"message": f"{filename} deleted"}
    return {"error": "File Not Found"}

@app.post("/chat")
async def chat(message: str = Form(...)):
    response = ask_bot(message)
    return {"response": response}

@app.get("/files")
def list_uploaded_files():
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"files": files}

@app.post("/new_session")
async def new_session():
    session_id = create_new_session()
    return {"session_id": session_id}

@app.get("/sessions")
def list_sessions():
    sessions = get_sessions()
    return {"sessions": sessions}

@app.post("/load_session")
async def load_session_endpoint(session_id: int = Form(...)):
    chat_history = load_session(session_id)
    return {"chat_history": chat_history}

@app.delete("/delete_session")
async def delete_session_endpoint(session_id: int = Form(...)):
    delete_session(session_id)
    return {"message": f"Session {session_id} deleted"}

@app.post("/rename_session")
async def rename_session_endpoint(session_id: int = Form(...), new_name: str = Form(...)):
    rename_session(session_id, new_name)
    return {"message": f"Session {session_id} renamed to {new_name}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)