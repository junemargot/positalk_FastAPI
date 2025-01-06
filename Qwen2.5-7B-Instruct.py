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
변환된 텍스트만 출력하세요. 다른 설명은 하지 마세요.
코드 블록이나 따옴표 없이 순수한 텍스트만 출력하세요.
"""

# 스타일별 지침 정의
style_instructions = {
    'formal': "격식있고 공식적인 어투로 변환해주세요.",
    'casual': "친근하고 편안한 어투로 변환해주세요.",
    'polite': "매우 공손하고 예의바른 어투로 변환해주세요.",
    'cute': "귀엽고 애교있는 어투로 변환해주세요."
}

class HuggingFaceHandler:
    def __init__(self):
        print("=== HuggingFace 모델 초기화 시작 ===")
        self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        
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

        # 추론 타임아웃 설정 (초)
        self.inference_timeout = 300   # 추론 타임아웃을 300초로 설정

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
                        max_new_tokens=100,      # 필요에 따라 조정
                        temperature=0.5,        # 덜 무작위적인 응답을 위해 낮춤
                        top_p=0.9,               # 샘플링을 제한
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

            # 디버깅을 위해 원본 응답 출력
            print(f"원본 응답: {response}")

            # 불필요한 코드 블록 제거
            response = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
            # 시작과 끝의 따옴표 제거 (양쪽에 있는 따옴표만 제거)
            response = re.sub(r'^["\']+|["\']+$', '', response).strip()

            # 추가적인 불필요한 문자 제거 (예: 백틱, 따옴표 등)
            response = response.replace('`', '').replace('"', '').replace("'", '').strip()


             # 여러 줄이 있는 경우 첫 번째 유효한 줄만 사용
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            if lines:
                response = lines[0]
            else:
                response = ''

            return response

        except asyncio.TimeoutError:
            print(f"[타임아웃] {self.inference_timeout}초 초과")
            return None
        except Exception as e:
            print(f"[에러] HuggingFace 모델 오류: {e}")
            return None
