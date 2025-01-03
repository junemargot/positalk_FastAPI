from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from openai_api import OpenAIHandler

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 핸들러 초기화
openai_handler = OpenAIHandler()

# API 요청을 위한 모델 정의
class ChatRequest(BaseModel):
    message: str
    style: str

# API 엔드포인트
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = openai_handler.get_completion(request.message, request.style)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# Favicon 엔드포인트 추가
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("../../positalk_react/build/favicon.ico")

# 정적 파일 서빙 설정
app.mount("/", StaticFiles(directory="static", html=True), name="static") 