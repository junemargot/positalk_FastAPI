from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
from openai_api import OpenAIHandler
from tts_handler import TTSHandler
from huggingface_api import HuggingFaceHandler

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
huggingface_handler = HuggingFaceHandler()
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
        print(f"\n[요청] 메시지: '{request.message}', 스타일: {request.style}")
        
        # 먼저 HuggingFace 모델 시도
        print("[시도] HuggingFace 모델 사용 시도...")
        response = await huggingface_handler.get_completion(request.message, request.style)
        
        # HuggingFace 모델이 실패하거나 타임아웃된 경우 OpenAI 모델 사용
        if response is None:
            print("[전환] HuggingFace 실패/타임아웃 -> OpenAI 모델로 전환")
            response = openai_handler.get_completion(request.message, request.style)
        else:
            print("[성공] HuggingFace 모델 사용 완료")
            
        print(f"[응답] '{response}'")
        return {"response": response}
    except Exception as e:
        print(f"[에러] {str(e)}")
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