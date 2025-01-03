from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
from openai_api import OpenAIHandler
from tts_handler import TTSHandler

# 환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 핸들러 초기화
openai_handler = OpenAIHandler()
tts_handler = TTSHandler()

class ChatRequest(BaseModel):
    message: str
    style: str  # 스타일 파라미터 추가

class TTSRequest(BaseModel):
    text: str
    voice: Dict[str, Any]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = openai_handler.get_completion(request.message, request.style)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        audio_content = await tts_handler.generate_speech(
            request.text, 
            request.voice
        )
        return Response(
            content=audio_content,
            media_type="audio/mpeg"
        )
    except Exception as e:
        return {"error": str(e)} 