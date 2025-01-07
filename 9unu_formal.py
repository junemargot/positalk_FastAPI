from transformers import T5TokenizerFast, T5ForConditionalGeneration, pipeline
import torch

formal_pipeline = None

# 모델 초기화
def init_pipeline():
    # 모델 경로 및 device 지정
    formal_model_path = '9unu/formal_speech_translation'

    # 모델과 토크나이저 초기화
    formal_model = T5ForConditionalGeneration.from_pretrained(formal_model_path)
    tokenizer = T5TokenizerFast.from_pretrained(formal_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global formal_pipeline

    # transformers 파이프라인 생성
    formal_pipeline = pipeline(
        task="text2text-generation",
        model=formal_model,
        tokenizer=tokenizer,
        device=device,
        max_length=60
    )

# 말투 변환
def convert(text):
    num_return_sequences = 1
    max_length = 60
    return formal_pipeline(text, num_return_sequences=num_return_sequences, max_length=max_length)

if __name__ == "__main__":
    init_pipeline()

    # text 말투 변환
    text = input("입력문장: ")
    out = convert(text)

    print(f"변환결과: {[x['generated_text'] for x in out]}")