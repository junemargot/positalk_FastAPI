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
from bllossom_handler import BllossomHandler
import text_style_converter_qwen25_3b_instruct as qwen3b
import heegyu
import formal_9unu
import gentle_9unu
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

# 전역 변수로 현재 로드된 핸들러와 모델 이름 저장
current_handler = None
current_model = None

async def cleanup_unused_handlers():
    """이전 핸들러를 메모리에서 해제하는 함수"""
    global current_handler, current_model
    
    if current_handler is not None:
        print(f"이전 모델 제거: {current_model}")
        current_handler = None
        current_model = None

async def get_handler(model_name: str):
    """필요할 때만 핸들러를 초기화하고 반환하는 함수"""
    global current_handler, current_model
    
    # 같은 모델을 요청한 경우 기존 핸들러 재사용
    if current_model == model_name and current_handler is not None:
        return current_handler
    
    # 다른 모델 요청 시 이전 핸들러 정리
    await cleanup_unused_handlers()
    
    try:
        # 새로운 핸들러 초기화
        if model_name == "openai-gpt":
            current_handler = OpenAIHandler()
        elif model_name == "gemini":
            current_handler = GeminiHandler()
        elif model_name == "polyglot-ko":
            current_handler = PolyglotKoHandler()
        elif model_name == "kogpt2":
            current_handler = KoGPT2Handler()
        elif model_name == "qwen18b":
            current_handler = Qwen18BHandler()
        elif model_name == "qwen15b":
            current_handler = Qwen15BHandler()
        elif model_name == "qwen7b":
            current_handler = Qwen7BHandler()
        elif model_name == "qwen3b":
            qwen3b.init_pipeline()
            current_handler = qwen3b
        elif model_name == "bllossom":
            current_handler = BllossomHandler()
        elif model_name == "h9":
            return None
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        current_model = model_name
        print(f"새로운 모델 로드: {model_name}")
        return current_handler
        
    except Exception as e:
        print(f"핸들러 초기화 중 에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # 요청된 모델의 핸들러 가져오기
        handler = await get_handler(request.model)
        
        # 모델별 응답 생성
        if request.model == "qwen3b":
            response = handler.convert_style(request.message, request.style)
        elif request.model == "openai-gpt":
            response = await handler.get_completion(
                request.message, 
                request.style,
                request.subModel
            )
        elif request.model == "gemini":
            response = handler.get_completion(
                request.message,
                request.style
            )
        elif request.model == "h9":
            # 스타일에 따라 다른 모듈 사용
            if request.style == 'formal':
                formal_9unu.init_pipeline()
                response = formal_9unu.convert(request.message)
            elif request.style == 'gentle':
                gentle_9unu.init_pipeline()
                response = gentle_9unu.convert(request.message)
            else:
                heegyu.init_pipeline()
                # heegyu 모듈 사용
                response = heegyu.transfer_text_style(
                    text=request.message,
                    target_style=request.style
                )
        else:
            response = await handler.get_completion(
                request.message,
                request.style
            )
            
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