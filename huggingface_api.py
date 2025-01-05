import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import asyncio
from typing import Optional

class HuggingFaceHandler:
    def __init__(self):
        print("=== HuggingFace 모델 초기화 시작 ===")
        self.model_name = "EleutherAI/polyglot-ko-5.8b"
        
        print("1. 토크나이저 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        print("✓ 토크나이저 로딩 완료")
        
        print("2. GPU 메모리 정리 중...")
        torch.cuda.empty_cache()
        print("✓ GPU 메모리 정리 완료")
        
        print("3. 모델 로딩 중... (1-3분 소요)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ 모델 로딩 완료")
        
        self.model_loaded = True
        print("=== 초기화 완료! 서비스 준비됨 ===")
            
        self.model_loaded = True
        self.emojis = ['💕', '✨', '🥺', '😊', '💝', '🌸', '💗', '💖']
        
        # 추론 타임아웃 설정 (초)
        self.inference_timeout = 30   # 추론 타임아웃을 60초로 설정

    async def get_completion(self, message: str, style: str) -> Optional[str]:
        if not self.model_loaded:
            print("[상태] 모델이 아직 로드되지 않았습니다")
            return None
            
        try:
            if style == 'cute':
                prompt = f"""다음 문장을 귀엽고 발랄한 말투로 변환해주세요.

                규칙:
                1. "~용", "~얏", "~냥" 같은 귀여운 어미 사용하기
                2. 밝고 긍정적인 톤으로 변환하기
                3. 짧고 간단하게 변환하기
                4. 문장 끝에는 느낌표나 물음표 사용하기

                입력: "{message}"
                출력:"""
            else:
                prompt = f"""다음 문장을 변환해주세요.
                입력: "{message}"
                출력:"""

            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            # 추론 시간만 타임아웃 적용
            print("[처리] 추론 시작...")
            outputs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=32,
                        temperature=0.7,
                        do_sample=True
                    )
                ),
                timeout=self.inference_timeout
            )
            print("[처리] 추론 완료")

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

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