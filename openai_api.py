from openai import AsyncOpenAI
import os

class OpenAIHandler:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.base_prompt = """
        당신은 문장 변환 전문가입니다.
        주어진 문장을 지정된 스타일로 변환해주세요.
        변환된 문장만 출력하세요. 다른 설명은 하지 마세요.
        """
        
        # Transform.js에 정의된 모델 목록
        self.models = {
            'gpt-4o-mini': "gpt-4o-mini"
        }

    async def get_completion(self, message: str, style: str, sub_model: str='gpt-4o-mini'):
        try:
            print(f"OpenAI 요청: message={message}, style={style}, model={sub_model}")  # 디버깅용
            
            # 실제 API에서 사용할 모델명 결정
            model = self.models.get(sub_model, "gpt-4o-mini") # 기본값으로 gpt-4o-mini 사용

            # style이 유효한지 확인하고, 유효하지 않으면 기본값 사용
            valid_styles = ['formal', 'casual', 'polite', 'cute']
            if style not in valid_styles:
                print(f"유효하지 않은 스타일 '{style}'. 기본 스타일 'formal'로 대체합니다.")
                style = 'formal'

            style_instructions = {
                'formal': "격식있고 공식적인 어투로 변환해주세요.",
                'casual': "친근하고 편안한 어투로 변환해주세요.",
                'polite': "매우 공손하고 예의바른 어투로 변환해주세요.",
                'cute': "귀엽고 애교있는 어투로 변환해주세요."
            }

            # 스타일이 딕셔너리에 없는 경우 기본 지시사항 사용
            instruction = style_instructions[style]

            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": f"다음 문장을 {instruction}\n문장: {message}"}
            ]
            
            print(f"사용할 모델: {model}, 스타일: {style}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI API 오류: {str(e)}")
            return Exception(f"GPT-4o-mini 응답 생성 실패: {str(e)}")
