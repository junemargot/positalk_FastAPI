from openai import OpenAI
import os

class DeepSeekAIHandler:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"  # 올바른 URL로 수정
        )
        self.base_prompt = """
        당신은 문장 변환 전문가입니다.
        주어진 문장을 지정된 스타일로 변환해주세요.
        변환된 문장만 출력하세요. 다른 설명은 하지 마세요.
        """

    def get_completion(self, message, style):
        style_instructions = {
            'formal': "격식있고 공식적인 어투로 변환해주세요.",
            'casual': "친근하고 편안한 어투로 변환해주세요.",
            'polite': "매우 공손하고 예의바른 어투로 변환해주세요.",
            'cute': "귀엽고 애교있는 어투로 변환해주세요."
        }

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # DeepSeek-V3 모델
                messages=[
                    {"role": "system", "content": self.base_prompt},
                    {"role": "user", "content": f"다음 문장을 {style_instructions[style]}\n문장: {message}"}
                ],
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API 상세 오류: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

# 사용 예시
if __name__ == "__main__":
    deepseek_handler = DeepSeekAIHandler()
    message = "오늘 날씨가 정말 좋네요."
    style = "cute"  # formal, casual, polite, cute 중 선택
    result = deepseek_handler.get_completion(message, style)
    print(result)