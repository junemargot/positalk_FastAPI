from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import load_dotenv
from openai_api import OpenAIHandler
from huggingface_api import HuggingFaceHandler
from tts_handler import TTSHandler

# 환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 개발 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI와 TTS 핸들러 초기화
openai_handler = OpenAIHandler()
tts_handler = TTSHandler()

# HuggingFace 핸들러는 지연 초기화
huggingface_handler = None

class ChatRequest(BaseModel):
    message: str
    style: str
    model: str

class TTSRequest(BaseModel):
    text: str
    voice: Dict[str, Any]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if request.model == 'openai':
            response = openai_handler.get_completion(request.message, request.style)
            return {"response": response if response else "OpenAI 처리 중 오류가 발생했습니다."}
        else:  # huggingface
            global huggingface_handler
            if huggingface_handler is None:
                huggingface_handler = HuggingFaceHandler()
            response = await huggingface_handler.get_completion(request.message, request.style)
            return {"response": response if response else "HuggingFace 처리 중 오류가 발생했습니다."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        audio_content = await tts_handler.generate_speech(request.text, request.voice)
        return Response(content=audio_content, media_type="audio/mp3")
    except Exception as e:
        return {"error": str(e)} 