import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiHandler:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        self.base_prompt = """
        당신은 문장 변환 전문가입니다.
        주어진 문장을 지정된 스타일로 변환해주세요.
        변환된 문장만 출력하세요. 다른 설명은 하지 마세요.
        """

    def get_completion(self, message, style):
        # 스타일에 따른 구체적인 지시사항 추가
        style_instructions = {
            'formal': "격식있고 공식적인 어투로 변환해주세요.",
            'casual': "친근하고 편안한 어투로 변환해주세요.",
            'polite': "매우 공손하고 예의바른 어투로 변환해주세요.",
            'cute': "귀엽고 애교있는 어투로 변환해주세요."
        }

        try:
            response = self.model.generate_content(
                [
                    {"role": "user", "parts": [{"text": self.base_prompt}]},
                    {"role": "user", "parts": [{"text": f"다음 문장을 {style_instructions[style]}\n문장: {message}"}]}
                ],
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=2048,
                )
            )
            
            return response.text
            
        except Exception as e:
            print(f"Gemini API 오류: {str(e)}")
            return None

# 테스트 코드
if __name__ == "__main__":
    handler = GeminiHandler()
    test_message = "오늘 날씨가 정말 좋네요."
    test_style = "formal"
    result = handler.get_completion(test_message, test_style)
    print(f"변환 결과: {result}") 