import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import asyncio
from typing import Optional

# 기본 프롬프트 정의
base_prompt = """
당신은 문장 변환 전문가입니다.
주어진 문장을 지정된 스타일로 변환해주세요.
변환된 문장만 출력하세요. 다른 설명은 하지 마세요.
코드 블록이나 따옴표 없이 순수한 텍스트만 출력하세요.
"""

# 스타일별 지침 정의는 동일하게 유지
style_instructions = {
    'formal': "격식있고 공식적인 어투('-습니다', '-니다'체)로 변환해주세요.",
    'casual': "친근하고 편안한 어투('-야', '-어'체)로 변환해주세요.",
    'polite': "매우 공손하고 예의바른 어투('-요'체)로 변환해주세요.",
    'cute': "귀엽고 애교있는 어투('~요'체)로 변환해주세요."
}

class BllossomHandler:
    def __init__(self):
        print("=== Llama Korean Bllossom 모델 초기화 시작 ===")
        self.model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
        self.model_loaded = False
        self.inference_timeout = 120
        
        try:
            print("1. 토크나이저 로딩 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✓ 토크나이저 로딩 완료")
            
            print("2. GPU 메모리 정리 중...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print(f"✓ 디바이스 설정 완료: {self.device}")
            
            print("3. 모델 로딩 중...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(self.device)
            print("✓ 모델 로딩 완료")
            
            self.model_loaded = True
            print("=== 초기화 완료! 서비스 준비됨 ===")
            
        except Exception as e:
            print(f"[에러] 초기화 실패: {e}")
            self.model_loaded = False

    async def get_completion(self, message: str, style: str) -> Optional[str]:
        if not self.model_loaded:
            print("[상태] 모델이 아직 로드되지 않았습니다")
            return None
            
        try:
            style_instruction = style_instructions.get(style, "")
            if not style_instruction:
                print(f"[경고] 지원하지 않는 스타일: {style}")
                return None

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
            ).to(self.device)
            
            print("[처리] 추론 시작...")
            outputs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=64,  # Llama는 좀 더 긴 컨텍스트를 처리할 수 있음
                        num_beams=5,
                        temperature=0.7,  # Llama에 맞게 약간 조정
                        top_p=0.95,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                        do_sample=True,
                        early_stopping=True
                    )
                ),
                timeout=self.inference_timeout
            )
            print("[처리] 추론 완료")
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # 응답 정제 로직은 동일하게 유지
            response = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
            response = re.sub(r'^["\']+|["\']+$', '', response).strip()
            response = response.replace('`', '').replace('"', '').replace("'", '').strip()
            
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
            print(f"[에러] Llama 모델 오류: {e}")
            return None