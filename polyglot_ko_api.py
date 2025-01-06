import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    'formal': "격식있고 공식적인 어투('-습니다', '-니다'체)로 변환해주세요.",
    'casual': "친근하고 편안한 어투('-야', '-어'체)로 변환해주세요.",
    'polite': "매우 공손하고 예의바른 어투('-요'체)로 변환해주세요.",
    'cute': "귀엽고 애교있는 어투('~요'체)로 변환해주세요."
}

class PolyglotKoHandler:  # 클래스 이름 변경
    def __init__(self):
        print("=== Polyglot-KO 모델 초기화 시작 ===")  # 로그 메시지도 수정
        self.model_name = "EleutherAI/polyglot-ko-5.8b"
        self.model_loaded = False
        self.inference_timeout = 300
        self.emojis = ['💕', '✨', '🥺', '😊', '💝', '🌸', '💗', '💖']
        
        try:
            print("1. 토크나이저 로딩 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            print("✓ 토크나이저 로딩 완료")
            
            print("2. GPU 메모리 정리 중...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"사용 가능한 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
            
            print("3. 모델 로딩 중... (1-3분 소요)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
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
            # 스타일에 해당하는 지침 가져오기
            style_instruction = style_instructions.get(style, "")
            if not style_instruction:
                print(f"[경고] 지원하지 않는 스타일: {style}")
                return None

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
            ).to(self.device)

            # token_type_ids 제거
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            print("[처리] 추론 시작...")
            outputs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.85,
                        repetition_penalty=1.1,
                        num_beams=3,
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

            # cute 스타일일 때 이모지 추가
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