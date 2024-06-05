from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import rag

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/api/ask")
async def ask(chat_message: ChatMessage):
    try:
        result = rag.answer_query(chat_message.message)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")