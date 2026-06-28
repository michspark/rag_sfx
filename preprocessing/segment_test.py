from typing import List, Dict, Any, Tuple, Optional
import base64
import math
import itertools
from datasets import load_dataset
from torchvision import io as tv_io
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ----- 설정값: 여기만 바꾸면 다른 조건으로 검증 가능 -----
START_INDEX: int = 2                         # FoleyBench train 스플릿에서 스캔 시작 인덱스
SCAN_LIMIT: int = 100                         # 10초짜리를 찾기 위해 최대 몇 개까지 훑을지
TARGET_DURATION: float = 10.0                 # 찾고 싶은 영상 길이(초)
DURATION_TOLERANCE: float = 1.0              # 목표 길이 허용 오차(±초)

SEGMENT_SECONDS: float = 2.0                  # 한 구간의 길이(초)
SEGMENT_FPS: float = 5.0                      # 구간 내 초당 샘플링 프레임 수
# → 구간당 프레임 = SEGMENT_SECONDS * SEGMENT_FPS = 2 * 5 = 10프레임

MAX_NEW_TOKENS: int = 128                     # Qwen이 생성할 최대 토큰 수
QUESTION: str = "Describe what is happening in this short clip in detail."
SOURCE_VIDEO_PATH: str = "segment_source.mp4"  # 선택된 원본 영상을 저장할 경로


def load_qwen() -> Tuple[Any, Any]:
    """Qwen3-VL 모델과 프로세서를 한 번만 로딩해서 반환한다."""
    # 8B 모델은 로딩이 무거우므로 루프 밖에서 딱 한 번만 호출한다.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model, processor


def save_video_from_example(example: Dict[str, Any], out_path: str) -> None:
    """예제의 base64 video_data를 디코딩해 mp4 파일로 저장한다."""
    # video_data는 mp4 바이너리를 base64로 인코딩한 '문자열'이다.
    video_bytes: bytes = base64.b64decode(example["video_data"])
    with open(out_path, "wb") as file:
        file.write(video_bytes)


def get_video_duration(path: str) -> float:
    """mp4 파일의 전체 재생 길이(초)를 구한다."""
    # read_video_timestamps: 프레임 픽셀은 디코딩하지 않고 '타임스탬프 목록'만 읽는다(가볍다).
    # pts_unit="sec" → 타임스탬프를 초 단위 float로 받는다. fps는 영상의 원래 프레임레이트.
    timestamps, fps = tv_io.read_video_timestamps(path, pts_unit="sec")
    if not timestamps:
        return 0.0
    # 마지막 프레임 시작 시각 + 한 프레임이 화면에 머무는 시간 ≈ 전체 길이.
    frame_duration: float = (1.0 / fps) if fps else 0.0
    return timestamps[-1] + frame_duration


def find_video_near_duration(
    target: float, tolerance: float, scan_limit: int
) -> Optional[Tuple[Dict[str, Any], float]]:
    """FoleyBench를 훑어 목표 길이(±tolerance)에 맞는 첫 예제를 찾는다."""
    # streaming=True: 전체를 받지 않고 필요한 만큼만 흘려본다.
    dataset = load_dataset("FoleyBench/foleybench", split="train", streaming=True)
    # START_INDEX부터 scan_limit개까지만 후보로 검사한다.
    candidates = itertools.islice(dataset, START_INDEX, START_INDEX + scan_limit)

    for offset, example in enumerate(candidates, start=START_INDEX):
        # 길이를 재려면 일단 파일로 떨군다(매번 같은 임시 경로에 덮어쓴다).
        save_video_from_example(example, SOURCE_VIDEO_PATH)
        duration: float = get_video_duration(SOURCE_VIDEO_PATH)
        print(f"[scan] index={offset} duration={duration:.2f}s")

        # 목표 길이에 충분히 가까우면 그 예제를 채택한다.
        if abs(duration - target) <= tolerance:
            print(f"[scan] -> picked index={offset} ({duration:.2f}s)")
            return example, duration

    # 끝까지 못 찾으면 None.
    return None


def build_segments(duration: float, segment_length: float) -> List[Tuple[float, float]]:
    """전체 길이를 segment_length초 간격의 (start, end) 구간 리스트로 나눈다."""
    # 구간 개수 = 올림(전체길이 / 구간길이). 10초 / 2초 = 5개.
    num_segments: int = math.ceil(duration / segment_length)
    segments: List[Tuple[float, float]] = []
    for segment_index in range(num_segments):
        start: float = segment_index * segment_length
        # 마지막 구간이 영상 끝을 넘지 않도록 end를 duration으로 자른다.
        end: float = min(start + segment_length, duration)
        segments.append((start, end))
    return segments


def describe_segment(
    model: Any,
    processor: Any,
    video_path: str,
    start: float,
    end: float,
    fps: float,
    question: str,
) -> str:
    """영상의 [start, end] 구간만 fps로 샘플링해 Qwen 설명을 생성한다."""
    # video_start / video_end: 영상 전체가 아니라 이 시간 구간만 디코딩하라는 지시(초 단위).
    # fps: 이 구간을 초당 몇 장으로 샘플링할지. 2초 구간 * 5fps = 10프레임.
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "video_start": start,      # 구간 시작(초)
                    "video_end": end,          # 구간 끝(초)
                    "fps": fps,                # 구간 내 초당 프레임 수
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
    # 지정한 구간을 실제 프레임 텐서로 디코딩한다.
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 이 구간에 실제로 몇 프레임이 들어갔는지 확인용 출력.
    num_frames: int = video_inputs[0].shape[0] if video_inputs else 0
    print(f"[info] frames fed for [{start:.1f}s, {end:.1f}s]: {num_frames}")

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
    # 0) 모델은 루프 전에 딱 한 번만 로딩.
    model, processor = load_qwen()

    # 1) 목표 길이(약 10초)에 맞는 FoleyBench 영상을 찾는다.
    found = find_video_near_duration(TARGET_DURATION, DURATION_TOLERANCE, SCAN_LIMIT)
    if found is None:
        print(f"[error] {TARGET_DURATION}s±{DURATION_TOLERANCE}s 영상을 찾지 못했습니다.")
        return
    example, duration = found
    foley_caption: str = example["caption"]
    # 채택된 영상은 이미 SOURCE_VIDEO_PATH에 저장돼 있다(스캔 단계에서 마지막에 저장됨).

    # 2) 전체 길이를 2초 구간들로 나눈다.
    segments: List[Tuple[float, float]] = build_segments(duration, SEGMENT_SECONDS)
    print(f"\n[plan] duration={duration:.2f}s -> {len(segments)} segments of {SEGMENT_SECONDS}s")

    # 3) 참고용으로 FoleyBench 오디오 캡션(정답)을 먼저 보여준다.
    print("\n=== FoleyBench sound caption (ground truth, about AUDIO) ===")
    print(foley_caption)

    # 4) 구간을 하나씩 순회하며 같은 모델로 설명 생성.
    for segment_index, (start, end) in enumerate(segments):
        description: str = describe_segment(
            model, processor, SOURCE_VIDEO_PATH, start, end, SEGMENT_FPS, QUESTION
        )
        print(f"\n############## SEGMENT {segment_index}  [{start:.1f}s ~ {end:.1f}s] ##############")
        print(description)


if __name__ == "__main__":
    main()
