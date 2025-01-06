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
            'gpt-3.5-turbo': "gpt-3.5-turbo",
            'gpt-4o-mini': "gpt-4o-mini"
        }

    async def get_completion(self, message: str, style: str, sub_model: str = 'gpt-3.5-turbo'):
        try:
            print(f"OpenAI 요청: message={message}, style={style}, model={sub_model}")  # 디버깅용
            
            style_instructions = {
                'formal': "격식있고 공식적인 어투로 변환해주세요.",
                'casual': "친근하고 편안한 어투로 변환해주세요.",
                'polite': "매우 공손하고 예의바른 어투로 변환해주세요.",
                'cute': "귀엽고 애교있는 어투로 변환해주세요."
            }

            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": f"다음 문장을 {style_instructions[style]}\n문장: {message}"}
            ]
            
            model = self.models.get(sub_model, 'gpt-3.5-turbo')
            print(f"사용할 모델: {model}")  # 디버깅용
            
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
            print(f"OpenAI API 상세 에러: {str(e)}")  # 디버깅용
            return None
