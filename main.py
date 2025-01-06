import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
from openai_api import OpenAIHandler
# from deepseek_api import DeepSeekAIHandler  # 주석 처리
from gemini_api import GeminiHandler
from huggingface_api import HuggingFaceHandler
from tts_handler import TTSHandler

# 환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 핸들러 초기화
openai_handler = OpenAIHandler()
try:
    # deepseek_handler = DeepSeekAIHandler()  # 주석 처리
    gemini_handler = GeminiHandler()
except Exception as e:
    print(f"모델 초기화 중 에러 발생: {e}")
    raise
huggingface_handler = HuggingFaceHandler()
tts_handler = TTSHandler()

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
        if request.model == 'openai-gpt':
            response = openai_handler.get_completion(request.message, request.style)
            return {"response": response if response else "OpenAI 처리 중 오류가 발생했습니다."}
        # elif request.model == 'deepseek':  # 주석 처리
        #     response = deepseek_handler.get_completion(request.message, request.style)
        #     return {"response": response if response else "Deepseek 처리 중 오류가 발생했습니다."}
        elif request.model == 'gemini':
            response = gemini_handler.get_completion(request.message, request.style)
            return {"response": response if response else "Gemini 처리 중 오류가 발생했습니다."}
        elif request.model == 'huggingface':
            response = await huggingface_handler.get_completion(request.message, request.style)
            return {"response": response if response else "HuggingFace 처리 중 오류가 발생했습니다."}
        else:
            return {"error": "지원하지 않는 모델입니다."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        audio_content = await tts_handler.generate_speech(request.text, request.voice)
        return Response(content=audio_content, media_type="audio/mp3")
    except Exception as e:
        return {"error": str(e)} 