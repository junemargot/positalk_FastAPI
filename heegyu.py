from transformers import pipeline

styles = ['문어체','구어체','안드로이드','아재','채팅',
    '초등학생','이모티콘','enfp','신사','할아버지','할머니','중학생',
    '왕','나루토','선비','소심한','번역기']

# 스타일 매핑 딕셔너리 추가
style_mapping = {
    'cute': '이모티콘',
    'casual': 'enfp'
}

# 모델 초기화
def init_pipeline():
    global model
    model = pipeline(
        'text2text-generation',
        model='heegyu/kobart-text-style-transfer'
    )

# 말투 변환
def transfer_text_style(text, target_style, **kwargs):
    # style_mapping에서 변환된 스타일을 가져오거나, 없으면 원래 스타일 사용
    converted_style = style_mapping.get(target_style, target_style)
    
    input = f"{converted_style} 말투로 변환:{text}"
    out = model(input, max_length=64, **kwargs)
    return out[0]['generated_text']

if __name__ == "__main__":
    init_pipeline()
    text = "안녕하세요"

    print("원문: ", text)
    for style in styles:
        print("말투: ", style)
        print(transfer_text_style(text, style))
        print("-" * 50)

    print("cute: ", transfer_text_style(text, 'cute'))
    print("casual: ", transfer_text_style(text, 'casual'))