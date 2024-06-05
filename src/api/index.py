from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import rag

origins = [
    "http://localhost:3000", 
    "https://lena-rag.vercel.app/",  
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    print(chat_message)
    try:
        result = rag.answer_query(chat_message.message)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")