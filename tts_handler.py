import requests
import urllib.parse
from fastapi import HTTPException
import os
from dotenv import load_dotenv

class TTSHandler:
    def __init__(self):
        load_dotenv()
        self.client_id = os.getenv('NAVER_CLIENT_ID')
        self.client_secret = os.getenv('NAVER_CLIENT_SECRET')
        self.api_url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"

    async def generate_speech(self, text, voice):
        try:
            # 스타일별 화자 매핑
            voice_mapping = {
                'formal': {
                    'speaker': 'nara',      # 전문적인 여성
                    'speed': '0',
                    'pitch': '0'
                },
                'casual': {
                    'speaker': 'njinho',    # 친근한 남성
                    'speed': '0',
                    'pitch': '0'
                },
                'polite': {
                    'speaker': 'nsujin',    # 공손한 여성
                    'speed': '0',
                    'pitch': '0'
                },
                'cute': {
                    'speaker': 'ndain',     # 귀여운 여성(아동)
                    'speed': '0',
                    'pitch': '0'
                }
            }

            # 프론트엔드의 select 값으로 스타일 가져오기
            current_style = voice.get('style', 'formal')
            style = voice_mapping[current_style]
            
            # URL 인코딩된 텍스트 생성
            encText = urllib.parse.quote(text)
            
            # 데이터 문자열 생성
            data = f"speaker={style['speaker']}&volume=0&speed={style['speed']}&pitch={style['pitch']}&format=mp3&text={encText}"

            headers = {
                "X-NCP-APIGW-API-KEY-ID": self.client_id,
                "X-NCP-APIGW-API-KEY": self.client_secret,
                "Content-Type": "application/x-www-form-urlencoded"
            }

            response = requests.post(
                self.api_url, 
                headers=headers, 
                data=data.encode('utf-8')
            )
            
            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"TTS API Error: {response.status_code}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 