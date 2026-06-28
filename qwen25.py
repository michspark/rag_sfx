from typing import List, Dict, Any, Tuple
import base64
import itertools
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ----- 설정값: 여기만 바꾸면 다른 예제로 검증 가능 -----
START_INDEX: int = 0                         # FoleyBench train 스플릿에서 시작 인덱스
NUM_EXAMPLES: int = 10                       # 몇 개의 예제를 연속으로 볼지
VIDEO_PATH_TEMPLATE: str = "video_{index}.mp4"  # 예제마다 다른 파일명으로 저장
QUESTION: str = "Describe what is happening in this video in detail."
MAX_NEW_TOKENS: int = 128                    # Qwen이 생성할 최대 토큰 수


def fetch_foley_examples(start: int, count: int) -> List[Dict[str, Any]]:
    """FoleyBench에서 start번째부터 count개 예제를 스트리밍으로 가져온다."""
    # streaming=True: 5000개 전부 받지 않고 필요한 행까지만 흘려본다.
    dataset = load_dataset("FoleyBench/foleybench", split="train", streaming=True)
    # islice(dataset, start, start+count): start~(start+count-1) 구간을 잘라낸다.
    # list(...)로 한 번에 모아 List로 반환한다(스트림은 한 번만 훑는다).
    examples: List[Dict[str, Any]] = list(
        itertools.islice(dataset, start, start + count)
    )
    return examples


def save_video_from_example(example: Dict[str, Any], out_path: str) -> None:
    """예제의 base64 video_data를 디코딩해 mp4 파일로 저장한다."""
    # video_data는 mp4 바이너리를 base64로 인코딩한 '문자열'이다.
    video_bytes: bytes = base64.b64decode(example["video_data"])
    with open(out_path, "wb") as file:
        file.write(video_bytes)


def load_qwen() -> Tuple[Any, Any]:
    """Qwen3-VL 모델과 프로세서를 한 번만 로딩해서 반환한다."""
    # 8B 모델을 GPU에 bf16으로 올린다(device_map="auto"가 알아서 배치).
    # 이 로딩은 무겁기 때문에 루프 밖에서 딱 한 번만 호출한다.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model, processor


def describe_video_with_qwen(model: Any, processor: Any, video_path: str, question: str) -> str:
    """이미 로딩된 Qwen3-VL로 영상을 보고 자연어 설명을 생성해 반환한다."""
    # 입력 메시지: 영상 1개 + 질문 1개.
    # fps/max_pixels로 프레임 수와 해상도를 제한해 VRAM과 속도를 조절한다.
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1.0,                # 초당 약 1프레임 샘플링
                    "max_pixels": 360 * 420,   # 프레임 1장당 최대 픽셀 수 제한
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    # 구조화된 메시지를 모델이 먹는 프롬프트 문자열로 변환한다.
    text: str = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # mp4를 실제 프레임 텐서로 디코딩한다(영상은 video_inputs로 나온다).
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 실제로 몇 프레임이 들어갔는지 확인용 출력.
    num_frames: int = video_inputs[0].shape[0] if video_inputs else 0
    print(f"[info] frames fed to model: {num_frames}")

    # 텍스트 생성(추론).
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    # 입력 프롬프트 토큰을 잘라내고 새로 생성된 토큰만 남긴다.
    trimmed: List = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded: List[str] = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0]


def main() -> None:
    # 0) 모델은 루프 전에 딱 한 번만 로딩(반복 로딩 방지).
    model, processor = load_qwen()

    # 1) FoleyBench에서 예제 10개를 한꺼번에 가져온다.
    examples: List[Dict[str, Any]] = fetch_foley_examples(START_INDEX, NUM_EXAMPLES)

    # 2) 예제를 하나씩 순회하며 처리한다.
    #    enumerate의 start=START_INDEX로 실제 데이터셋 인덱스를 같이 보여준다.
    for index, example in enumerate(examples, start=START_INDEX):
        foley_caption: str = example["caption"]

        # 예제마다 다른 파일명으로 mp4 저장(video_0.mp4, video_1.mp4 ...).
        video_path: str = VIDEO_PATH_TEMPLATE.format(index=index)
        save_video_from_example(example, video_path)

        # 같은 모델/프로세서를 재사용해 화면 설명 생성.
        qwen_description: str = describe_video_with_qwen(
            model, processor, video_path, QUESTION
        )

        # 캡션 vs Qwen 설명 비교 출력.
        print(f"\n############## EXAMPLE {index} ##############")
        print("=== FoleyBench sound caption (ground truth, about AUDIO) ===")
        print(foley_caption)
        print("\n=== Qwen3-VL description (what it SEES, no audio) ===")
        print(qwen_description)


if __name__ == "__main__":
    main()
