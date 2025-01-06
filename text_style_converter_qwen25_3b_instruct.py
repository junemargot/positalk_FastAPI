from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict

model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_prompt = """
당신은 문장 변환 전문가입니다.
주어진 문장을 지정된 스타일로 변환해주세요.
변환된 문장만 출력하세요. 다른 설명은 하지 마세요.
원문의 의미는 되도록 유지될 수 있도록 해주세요.
"""

# 스타일별 시스템 프롬프트와 지시사항
style_settings: Dict[str, Dict[str, str]] = {
    'formal': {
        'persona': """당신은 격식과 예의를 중시하는 전문가입니다.
비격식적인 표현을 격식있고 공식적인 표현으로 변환하는 전문성을 가지고 있습니다.
항상 '-습니다', '-입니다'와 같은 격식체를 사용하며, 전문적이고 공식적인 어휘를 선택합니다.""",
        'instruction': "다음 문장을 격식있고 공식적인 어투로 변환해주세요.",
        'examples': [
            {"input": "이거 내일까지 해야돼", 
             "output": "해당 업무를 내일까지 완료해야 합니다."},
            {"input": "근데 이게 맞나?", 
             "output": "그러나 이것이 적절한지 검토가 필요합니다."}
        ]
    },
    'casual': {
        'persona': """당신은 편안하고 친근한 말투를 구사하는 20대입니다.
어떤 문장이든 자연스럽고 친근한 일상 대화체로 바꿀 수 있습니다.
'-야', '-어', '-지' 등의 반말을 사용하고, 구어체 표현을 적절히 활용합니다.""",
        'instruction': "다음 문장을 친근하고 편안한 어투로 변환해주세요.",
        'examples': [
            {"input": "회의 자료를 검토하여 주시기 바랍니다", 
             "output": "회의 자료 한번 봐줘"},
            {"input": "금일 업무보고를 진행하도록 하겠습니다", 
             "output": "오늘 업무보고 할게"}
        ]
    },
    'polite': {
        'persona': """당신은 예의 바른 말투를 구사하는 서비스업 종사자입니다.
항상 상대방을 존중하고 공손한 표현을 사용합니다.
'-요', '-세요'와 같은 존댓말을 사용하며, 정중하고 친절한 어휘를 선택합니다.""",
        'instruction': "다음 문장을 매우 공손하고 예의바른 어투로 변환해주세요.",
        'examples': [
            {"input": "이거 좀 봐줘", 
             "output": "이것 좀 봐주시겠어요?"},
            {"input": "여기서 기다려", 
             "output": "이곳에서 잠시만 기다려 주시겠어요?"}
        ]
    },
    'cute': {
        'persona': """당신은 귀엽고 사랑스러운 말투를 구사하는 아이돌입니다.
어떤 문장이든 귀엽고 애교있는 표현으로 바꿀 수 있습니다.
'~요', '~애요', '~냥' 등의 귀여운 어미를 사용하고, 이모티콘을 적절히 활용합니다.""",
        'instruction': "다음 문장을 귀엽고 애교있는 어투로 변환해주세요.",
        'examples': [
            {"input": "안녕하세요", 
             "output": "안녕하세요~! ❤️"},
            {"input": "잠시만 기다려주세요", 
             "output": "잠시만 기다려주실래용~? 😊"}
        ]
    }
}

def init_pipeline(model_path: str = "Qwen/Qwen2.5-3B-Instruct") -> None:
    """모델과 토크나이저 초기화"""
    global model, tokenizer
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

def create_style_prompt(text: str, style: str) -> str:
    """스타일 변환을 위한 프롬프트 생성"""
    if style not in style_settings:
        raise ValueError(f"지원하지 않는 스타일입니다: {style}")
        
    setting = style_settings[style]
    system_prompt = f"""
{setting['persona']}

당신은 문장 변환 전문가입니다.
주어진 문장을 지정된 스타일로 변환해주세요.
변환된 문장만 출력하세요. 다른 설명은 하지 마세요.
원문의 의미를 해치지 않고 최대한 유지하면서 자연스러운 표현으로 변환해주세요.
일관된 어투를 유지하고, 문맥에 맞는 적절한 어휘를 선택하세요.
"""

    # Few-shot 예시 추가
    examples = "\n\n예시:\n"
    for example in setting['examples']:
        examples += f"입력: {example['input']}\n출력: {example['output']}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{setting['instruction']}\n{examples}\n\n문장: {text}"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

@torch.inference_mode()
def generate_response(text: str, max_new_tokens: int = 512) -> str:
    """응답 생성 - 파라미터 최적화"""
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    ).to(device)
    
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,  # 더 안정적인 출력을 위해 낮춤
        top_p=0.85,      # 더 집중된 확률 분포
        top_k=40,        # top-k sampling 추가
        repetition_penalty=1.3,  # 반복 방지 강화
        no_repeat_ngram_size=3,  # n-gram 반복 방지
        num_beams=3      # beam search 적용
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids 
        in zip(inputs.input_ids, outputs)
    ]
    
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def convert_style(text: str, style: str) -> str:
    """스타일 변환 실행"""
    if model is None:
        init_pipeline()
        
    prompt = create_style_prompt(text, style)
    return generate_response(prompt) 

if __name__ == "__main__":
    init_pipeline()
    
    # 테스트
    test_text = """2025년은 AI가 폭발적으로 성장하는 한 해가 될거야.
왜냐하면 AUNT가 본격적으로 프로젝트를 실행한 해니까."""
    
    print("원문:", test_text)
    print("\n각 스타일별 변환 결과:")
    
    for style in style_settings.keys():
        print(f"\n[{style} 스타일]")
        print(convert_style(test_text, style))
        print("-" * 50)