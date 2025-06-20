from fastapi import FastAPI, UploadFile, Request,  File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil 
import uvicorn
import os 
from chat_logic import ask_bot

app = FastAPI()
app.mount("/static", StaticFiles(directory= "static"), name="static")

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("docs")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return Path("templates/index.html").read_text(encoding='utf-8')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = UPLOAD_DIR / file.filename
    with open(file_location,"wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": f"{file.filename} uploaded successfully"}

@app.delete("/delete")
async def delete_file(filename: str = Form(...)):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"message": f"{filename} deleted"}
    return {"error": "File Not Found"}

@app.post("/chat")
async def chat(message: str = Form(...)):
    response = ask_bot(message)  
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port=8000, reload = True)