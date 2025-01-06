import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import asyncio
from typing import Optional
import concurrent.futures

class HuggingFaceHandler:
    def __init__(self):
        print("=== HuggingFace 모델 초기화 시작 ===")
        self.model_name = "EleutherAI/polyglot-ko-5.8b"
        
        # CUDA 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 중인 디바이스: {self.device}")
        
        print("1. 토크나이저 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        print("✓ 토크나이저 로딩 완료")
        
        print("2. GPU 메모리 정리 중...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        
        print("3. 모델 로딩 중... (1-3분 소요)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # 반정밀도(FP16) 사용
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # CPU 메모리 사용량 최적화
        )
        self.model.eval()  # 추론 모드로 설정
        
        self.model_loaded = True
        print("=== 초기화 완료! 서비스 준비됨 ===")
            
        self.model_loaded = True
        self.emojis = ['💕', '✨', '🥺', '😊', '💝', '🌸', '💗', '💖']
        
        # 추론 타임아웃 설정 (초)
        self.inference_timeout = 300   # 60초로 수정

    async def get_completion(self, message: str, style: str) -> Optional[str]:
        if not self.model_loaded:
            print("[상태] 모델이 아직 로드되지 않았습니다")
            return None
            
        try:
            # 스타일별 프롬프트 정의
            style_prompts = {
                'formal': f"""아래 문장을 격식체로 바꾸어 주세요. 문장의 원래 의미는 그대로 유지하면서 말투만 변경하세요.
                원문: "{message}"
                변환된 문장:""",
                
                'casual': f"""아래 문장을 친근하고 편안한 말투로 바꾸어 주세요. 문장의 원래 의미는 그대로 유지하면서 말투만 변경하세요.
                원문: "{message}"
                변환된 문장:""",
                
                'polite': f"""아래 문장을 공손하고 예의 바른 말투로 바꾸어 주세요. 문장의 원래 의미는 그대로 유지하면서 말투만 변경하세요.
                원문: "{message}"
                변환된 문장:""",
                
                'cute': f"""아래 문장을 귀엽고 애교있는 말투로 바꾸어 주세요. 문장의 원래 의미는 그대로 유지하면서 말투만 변경하세요.
                원문: "{message}"
                변환된 문장:"""
            }

            prompt = style_prompts.get(style, style_prompts['casual'])  # 기본값은 친근체

            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)  # 명시적으로 device 지정

            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            outputs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=100,     
                        temperature=0.3,        # 더 낮춰서 안정성 확보
                        do_sample=True,
                        top_p=0.85,            # 더 낮춰서 관련성 높은 토큰만 선택
                        repetition_penalty=1.1,  # 약간의 반복 방지만 적용
                        num_beams=3,           # 빔 서치 추가
                        early_stopping=True     # 적절한 시점에 생성 중단
                    )
                ),
                timeout=self.inference_timeout
            )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # cute 스타일일 때만 이모지 추가
            if style == 'cute':
                emoji_count = random.randint(1, 2)
                selected_emojis = ' ' + ''.join(random.sample(self.emojis, emoji_count))
                response = response + selected_emojis

            return response

        except asyncio.TimeoutError:
            print(f"[타임아웃] {self.inference_timeout}초 초과")
            return None
        except Exception as e:
            print(f"[에러] HuggingFace 모델 오류: {e}")
            return None
    def _generate_response(self, message: str, style: str) -> str:
        # 기존의 동기 처리 코드를 여기로 이동
        if style == 'cute':
            prompt = f"""다음 문장을 귀엽고 발랄한 말투로 변환해주세요...."""
        else:
            prompt = f"""다음 문장을 변환해주세요...."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
