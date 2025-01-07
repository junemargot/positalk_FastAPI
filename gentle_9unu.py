from transformers import T5TokenizerFast, T5ForConditionalGeneration, pipeline
import torch

formal_pipeline = None

# 모델 초기화
def init_pipeline():
    # 모델 경로 및 device 지정
    gentle_model_path = '9unu/gentle_speech_translation'

    # 모델과 토크나이저 초기화
    gentle_model = T5ForConditionalGeneration.from_pretrained(gentle_model_path)
    tokenizer = T5TokenizerFast.from_pretrained(gentle_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global gentle_pipeline

    # transformers 파이프라인 생성
    gentle_pipeline = pipeline(
        task = "text2text-generation",
        model = gentle_model, 
        tokenizer = tokenizer, 
        device = device, 
        max_length = 60
    )

# 말투 변환
def convert(text):
    num_return_sequences = 1
    max_length = 60
    output = gentle_pipeline(text, num_return_sequences=num_return_sequences, max_length=max_length)
    # 생성된 텍스트만 반환
    return output[0]['generated_text']

if __name__ == "__main__":
    init_pipeline()

    # text 말투 변환
    text = input("입력문장: ")
    out = convert(text)

    print(f"변환결과: {[x['generated_text'] for x in out]}")