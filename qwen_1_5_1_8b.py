import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random
import asyncio
from typing import Optional

# 기본 프롬프트 정의
base_prompt = """
당신은 문장 변환 전문가입니다.
주어진 문장을 지정된 스타일로 변환해주세요.
변환된 문장만 출력하세요. 다른 설명은 하지 마세요.
코드 블록이나 따옴표 없이 순수한 텍스트만 출력하세요.
"""

# 스타일별 지침 정의
style_instructions = {
    'formal': "격식있고 공식적인 어투로 변환해주세요.",
    'casual': "친근하고 편안한 어투로 변환해주세요.",
    'polite': "매우 공손하고 예의바른 어투로 변환해주세요.",
    'cute': "귀엽고 애교있는 어투로 변환해주세요."
}

class TestHandler:
    def __init__(self):
        print("=== HuggingFace 모델 초기화 시작 ===")
        self.model_name = "Qwen/Qwen1.5-1.8B"
        
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
        self.inference_timeout = 300  # 추론 타임아웃 설정 (초)
        self.emojis = ['💕', '✨', '🥺', '😊', '💝', '🌸', '💗', '💖']
        print("=== 초기화 완료! 서비스 준비됨 ===")

    async def get_completion(self, message: str, style: str) -> Optional[str]:
        if not self.model_loaded:
            print("[상태] 모델이 아직 로드되지 않았습니다")
            return None

        try:
            # 스타일에 해당하는 지침 가져오기
            style_instruction = style_instructions.get(style, "")
            if not style_instruction:
                print(f"[경고] 지원하지 않는 스타일: {style}. 기본 스타일로 변환합니다.")
                style_instruction = "지정된 스타일에 맞게 문장을 변환해주세요."

            # 프롬프트 구성
            prompt = f"""{base_prompt}
                    {style_instruction}

                    입력: "{message}"
                    """

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=30000
            ).to(self.model.device)

            # 추론 시간만 타임아웃 적용
            print("[처리] 추론 시작...")
            outputs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=50,
                        temperature=0.5,
                        top_p=0.9,
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

            # 스타일별 추가 로직 적용
            response = self.apply_style_logic(response, style)
            return response

        except asyncio.TimeoutError:
            print(f"[타임아웃] {self.inference_timeout}초 초과")
            return None
        except Exception as e:
            print(f"[에러] HuggingFace 모델 오류: {e}")
            return None

    def apply_style_logic(self, response: str, style: str) -> str:
        """스타일별 변환 로직 적용"""
        if style == "formal":
            response = response.replace("~야", "~입니다").replace("~네", "~입니다")
            response += " 감사합니다."

        elif style == "casual":
            response = response.replace("~입니다", "~야").replace("~합니다", "~해")
            response += " ㅎㅎ"

        elif style == "polite":
            response = response.replace("~야", "~입니다").replace("~해", "~하시겠어요?")
            response += " 고맙습니다."

        elif style == "cute":
            response = response.replace("~입니다", "~냥").replace("~합니다", "~얌")
            emoji_count = random.randint(1, 2)
            selected_emojis = " " + "".join(random.sample(self.emojis, emoji_count))
            response += selected_emojis

        return response
