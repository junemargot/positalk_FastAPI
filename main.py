import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
from openai_api import OpenAIHandler
from gemini_api import GeminiHandler
from polyglot_ko_api import PolyglotKoHandler
from kogpt2_handler import KoGPT2Handler
from qwen_1_5_1_8b import TestHandler as Qwen18BHandler
from qwen_2_5_1_5b_instruct import HuggingFaceHandler as Qwen15BHandler
from qwen_2_5_7b_instruct import HuggingFaceHandler as Qwen7BHandler
import text_style_converter_qwen25_3b_instruct as qwen3b
from tts_handler import TTSHandler

# 환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 핸들러 초기화
openai_handler = OpenAIHandler()
try:
    gemini_handler = GeminiHandler()
    polyglot_handler = PolyglotKoHandler()
    kogpt2_handler = KoGPT2Handler()
    qwen18b_handler = Qwen18BHandler()
    qwen15b_handler = Qwen15BHandler()
    qwen7b_handler = Qwen7BHandler()
    qwen3b.init_pipeline()  # Qwen 3B 초기화
except Exception as e:
    print(f"모델 초기화 중 에러 발생: {e}")
    raise

tts_handler = TTSHandler()

class ChatRequest(BaseModel):
    message: str
    style: str
    model: str
    subModel: str = 'gpt-3.5-turbo'

class TTSRequest(BaseModel):
    text: str
    voice: Dict[str, Any]

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        print(f"요청 데이터: {request}")
        
        if request.model == "openai-gpt":
            response = await openai_handler.get_completion(
                request.message, 
                request.style,
                request.subModel
            )
        elif request.model == "gemini":
            response = gemini_handler.get_completion(
                request.message,
                request.style
            )
        elif request.model == "huggingface":  # polyglot-ko
            response = await polyglot_handler.get_completion(
                request.message,
                request.style
            )
        elif request.model == "kogpt2":
            response = await kogpt2_handler.get_completion(
                request.message,
                request.style
            )
        elif request.model == "qwen18b":
            response = await qwen18b_handler.get_completion(
                request.message,
                request.style
            )
        elif request.model == "qwen15b":
            response = await qwen15b_handler.get_completion(
                request.message,
                request.style
            )
        elif request.model == "qwen7b":
            response = await qwen7b_handler.get_completion(
                request.message,
                request.style
            )
        elif request.model == "qwen3b":
            response = qwen3b.convert_style(request.message, request.style)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 모델입니다.")
            
        if response is None:
            raise HTTPException(status_code=500, detail="응답 생성 실패")
                
        return {"response": response}
            
    except Exception as e:
        print(f"서버 에러: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        audio_content = await tts_handler.generate_speech(request.text, request.voice)
        return Response(content=audio_content, media_type="audio/mp3")
    except Exception as e:
        return {"error": str(e)} 