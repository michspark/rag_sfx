# 10초 영상을 2초 구간으로 나눠, 각 구간마다 Qwen3-VL / RAM++ / Places365 출력을
# 한 텍스트로 모으고, 전체 구간을 하나의 텍스트로 concat 해서 파일로 저장한다.
#
# 메모리 전략(RTX 4090 24GB): 세 모델을 동시에 올리지 않고 '한 모델 → 전 구간 처리 → 메모리 비움'
# 순서로 돈다(Qwen 8B + RAM 동시 로딩 시 OOM 위험 때문).

import os
# qwen_vl_utils가 torchcodec(현재 환경에서 libtorchcodec 로딩 실패)을 시도하지 않고
# 바로 torchvision으로 영상을 읽게 강제한다. qwen_vl_utils import 전에 설정해야 적용됨.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchvision")

import gc
import base64
import math
import itertools
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision import io as tv_io
from torchvision import transforms as trn
import torchvision.models as tv_models

# ----- 경로: 스크립트 위치 기준(폴더명이 바뀌어도 안전) -----
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
PLACES_DIR: str = os.path.join(BASE_DIR, "places365")
RAM_CHECKPOINT: str = os.path.join(BASE_DIR, "ram_plus_swin_large_14m.pth")
RESNET_CHECKPOINT: str = os.path.join(PLACES_DIR, "resnet50_places365.pth.tar")
CATEGORY_FILE: str = os.path.join(PLACES_DIR, "categories_places365.txt")
IO_FILE: str = os.path.join(PLACES_DIR, "IO_places365.txt")
SOURCE_VIDEO_PATH: str = os.path.join(BASE_DIR, "segment_source.mp4")
OUTPUT_TEXT_PATH: str = os.path.join(BASE_DIR, "preprocess_output.txt")

# ----- FoleyBench에서 ~10초 영상을 찾기 위한 설정 -----
START_INDEX: int = 2
SCAN_LIMIT: int = 100
TARGET_DURATION: float = 10.0
DURATION_TOLERANCE: float = 1.0

# ----- 구간 분할 설정 -----
SEGMENT_SECONDS: float = 2.0          # 한 구간 길이(초)
SEGMENT_FPS: float = 5.0              # Qwen이 구간을 샘플링할 초당 프레임 수(2초*5 = 10프레임)
MIN_SEGMENT_SECONDS: float = 0.5      # 이보다 짧은 꼬리 구간은 버림(프레임 샘플링 불가 → 에러 방지)

# ----- 모델 설정 -----
DEVICE: str = "cuda"
QWEN_MODEL_ID: str = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_QUESTION: str = "Describe what is happening in this short clip in detail."
QWEN_MAX_NEW_TOKENS: int = 128
QWEN_MAX_PIXELS: int = 360 * 420
RAM_IMAGE_SIZE: int = 384
RAM_VIT: str = "swin_l"
PLACES_NUM_CLASSES: int = 365
PLACES_TOPK: int = 5


# ==========================================================================
# 1) 영상 확보 + 구간 분할 + 프레임 추출
# ==========================================================================
def save_video_from_example(example: Dict[str, Any], out_path: str) -> None:
    """예제의 base64 video_data를 디코딩해 mp4로 저장한다."""
    video_bytes: bytes = base64.b64decode(example["video_data"])
    with open(out_path, "wb") as file:
        file.write(video_bytes)


def get_video_duration(path: str) -> float:
    """mp4의 전체 재생 길이(초)를 타임스탬프만 읽어 가볍게 구한다."""
    timestamps, fps = tv_io.read_video_timestamps(path, pts_unit="sec")
    if not timestamps:
        return 0.0
    frame_duration: float = (1.0 / fps) if fps else 0.0
    return timestamps[-1] + frame_duration


def find_video_near_duration(target: float, tolerance: float, scan_limit: int) -> Optional[Tuple[Dict[str, Any], float]]:
    """FoleyBench를 훑어 목표 길이(±tolerance)에 맞는 첫 예제를 찾아 저장한다."""
    from datasets import load_dataset  # 무거운 import는 필요할 때만
    # streaming: 전체를 받지 않고 필요한 만큼만 흘려본다.
    dataset = load_dataset("FoleyBench/foleybench", split="train", streaming=True)
    candidates = itertools.islice(dataset, START_INDEX, START_INDEX + scan_limit)

    for offset, example in enumerate(candidates, start=START_INDEX):
        # 길이를 재려면 일단 파일로 저장(매번 같은 경로에 덮어씀).
        save_video_from_example(example, SOURCE_VIDEO_PATH)
        duration: float = get_video_duration(SOURCE_VIDEO_PATH)
        print(f"[scan] index={offset} duration={duration:.2f}s")
        if abs(duration - target) <= tolerance:
            print(f"[scan] -> picked index={offset} ({duration:.2f}s)")
            return example, duration
    return None


def build_segments(duration: float, segment_length: float) -> List[Tuple[float, float]]:
    """전체 길이를 segment_length초 간격의 (start, end) 구간 리스트로 나눈다."""
    num_segments: int = math.ceil(duration / segment_length)
    segments: List[Tuple[float, float]] = []
    for segment_index in range(num_segments):
        start: float = segment_index * segment_length
        end: float = min(start + segment_length, duration)  # 마지막 구간이 끝을 안 넘게 자름
        # 0.04초 같은 자투리 구간은 프레임을 못 뽑아 에러나므로 버린다.
        if end - start < MIN_SEGMENT_SECONDS:
            continue
        segments.append((start, end))
    return segments


def extract_frame(video_path: str, time_sec: float) -> Image.Image:
    """영상에서 time_sec 지점 프레임 1장을 PIL(RGB)로 뽑는다."""
    frames, _audio, _info = tv_io.read_video(
        video_path, start_pts=time_sec, end_pts=time_sec + 0.2, pts_unit="sec"
    )
    if frames.shape[0] == 0:
        raise ValueError(f"{video_path} 의 {time_sec}s 지점에서 프레임을 못 찾았습니다.")
    first_frame = frames[0].numpy()  # (H, W, C), RGB, 0~255
    return Image.fromarray(first_frame)


# ==========================================================================
# 2) Qwen3-VL: 각 2초 구간 영상을 설명
# ==========================================================================
def run_qwen_on_segments(video_path: str, segments: List[Tuple[float, float]]) -> List[str]:
    """각 구간을 video_start/video_end로 잘라 Qwen3-VL 설명을 생성한다."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print("[qwen] loading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)

    descriptions: List[str] = []
    for start, end in segments:
        # 메시지: 이 시간 구간만 fps로 샘플링한 영상 + 질문.
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "video_start": start,
                        "video_end": end,
                        "fps": SEGMENT_FPS,
                        "max_pixels": QWEN_MAX_PIXELS,
                    },
                    {"type": "text", "text": QWEN_QUESTION},
                ],
            }
        ]
        text: str = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        # 추론(그래디언트 불필요 → 메모리 절약).
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS)
        # 입력 프롬프트 토큰을 잘라내고 새로 생성된 부분만 디코딩.
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        descriptions.append(decoded[0].strip())
        print(f"[qwen] segment [{start:.1f}s~{end:.1f}s] done")

    # GPU 메모리 비우기(다음 모델을 위해).
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return descriptions


# ==========================================================================
# 3) RAM++: 각 구간 대표 프레임의 태그
# ==========================================================================
def run_ram_on_frames(frames: List[Image.Image]) -> List[str]:
    """각 프레임을 RAM++에 넣어 인식 태그 문자열을 만든다."""
    from ram.models import ram_plus
    from ram import inference_ram, get_transform

    print("[ram] loading model...")
    transform = get_transform(image_size=RAM_IMAGE_SIZE)
    model = ram_plus(pretrained=RAM_CHECKPOINT, image_size=RAM_IMAGE_SIZE, vit=RAM_VIT)
    model.eval()
    model = model.to(DEVICE)

    tags_list: List[str] = []
    for index, frame in enumerate(frames):
        image_tensor = transform(frame).unsqueeze(0).to(DEVICE)  # (1,C,H,W)
        with torch.no_grad():
            english_tags, _chinese = inference_ram(image_tensor, model)
        tags_list.append(english_tags)
        print(f"[ram] frame {index} done")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tags_list


# ==========================================================================
# 4) Places365 (ResNet50): 각 구간 대표 프레임의 장면 카테고리 + 실내/실외
# ==========================================================================
def load_places_labels() -> Tuple[Tuple[str, ...], np.ndarray]:
    """장면 카테고리 이름과 실내/실외 라벨을 로컬 파일에서 읽는다."""
    classes: List[str] = []
    with open(CATEGORY_FILE) as class_file:
        for line in class_file:
            classes.append(line.strip().split(" ")[0][3:])  # "/a/airport 0" → "airport"
    labels_IO: List[int] = []
    with open(IO_FILE) as f:
        for line in f.readlines():
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0=indoor, 1=outdoor
    return tuple(classes), np.array(labels_IO)


def run_places_on_frames(frames: List[Image.Image]) -> List[Dict[str, Any]]:
    """각 프레임의 장면 Top-K 카테고리와 실내/실외 판정을 만든다."""
    print("[places] loading model...")
    classes, labels_IO = load_places_labels()

    # 빈 resnet50(출력 365) 골격 + 체크포인트(module. 접두사 제거) 로드.
    model = tv_models.resnet50(num_classes=PLACES_NUM_CLASSES)
    checkpoint = torch.load(RESNET_CHECKPOINT, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(DEVICE)

    # Places365 입력 변환(224 리사이즈 + ImageNet 정규화).
    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results: List[Dict[str, Any]] = []
    for index, frame in enumerate(frames):
        input_img = transform(frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        # 상위 10개 예측의 IO 라벨 평균으로 실내/실외 투표.
        io_value = float(np.mean(labels_IO[idx[:10]]))
        environment = "indoor" if io_value < 0.5 else "outdoor"
        top = [(float(probs[i]), classes[idx[i]]) for i in range(PLACES_TOPK)]
        results.append({"environment": environment, "top": top})
        print(f"[places] frame {index} done")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ==========================================================================
# 5) 세그먼트별로 묶고 전체를 하나의 텍스트로 concat
# ==========================================================================
def assemble_text(
    duration: float,
    segments: List[Tuple[float, float]],
    qwen_texts: List[str],
    ram_texts: List[str],
    places_results: List[Dict[str, Any]],
) -> str:
    """세 모델 출력을 세그먼트별로 모아 하나의 텍스트로 만든다."""
    lines: List[str] = []
    lines.append(f"VIDEO: {SOURCE_VIDEO_PATH}")
    lines.append(f"DURATION: {duration:.2f}s  |  SEGMENTS: {len(segments)} x {SEGMENT_SECONDS}s")
    lines.append("")

    for i, (start, end) in enumerate(segments):
        lines.append(f"========== SEGMENT {i}  [{start:.1f}s ~ {end:.1f}s] ==========")
        # Qwen 설명
        lines.append(f"[Qwen3-VL] {qwen_texts[i]}")
        # RAM 태그
        lines.append(f"[RAM++ tags] {ram_texts[i]}")
        # Places365 장면
        place = places_results[i]
        top_str = ", ".join(f"{name} ({prob:.3f})" for prob, name in place["top"])
        lines.append(f"[Places365] {place['environment']} | {top_str}")
        lines.append("")  # 세그먼트 사이 빈 줄

    return "\n".join(lines)


def main() -> None:
    # 1) FoleyBench에서 ~10초 영상 찾기
    found = find_video_near_duration(TARGET_DURATION, DURATION_TOLERANCE, SCAN_LIMIT)
    if found is None:
        print(f"[error] {TARGET_DURATION}s±{DURATION_TOLERANCE}s 영상을 찾지 못했습니다.")
        return
    _example, duration = found

    # 2) 구간 분할 + 각 구간 중간 지점 프레임 추출(RAM/Places용)
    segments = build_segments(duration, SEGMENT_SECONDS)
    midpoints = [(start + end) / 2.0 for start, end in segments]
    frames = [extract_frame(SOURCE_VIDEO_PATH, t) for t in midpoints]
    print(f"[plan] duration={duration:.2f}s -> {len(segments)} segments")

    # 3) 모델별 순차 처리(메모리 안전)
    qwen_texts = run_qwen_on_segments(SOURCE_VIDEO_PATH, segments)
    ram_texts = run_ram_on_frames(frames)
    places_results = run_places_on_frames(frames)

    # 4) 세그먼트별로 묶어 하나의 텍스트로 concat
    final_text = assemble_text(duration, segments, qwen_texts, ram_texts, places_results)

    # 5) 파일 저장 + 출력
    with open(OUTPUT_TEXT_PATH, "w") as f:
        f.write(final_text)
    print("\n" + "=" * 60)
    print(final_text)
    print("=" * 60)
    print(f"\n[done] saved to {OUTPUT_TEXT_PATH}")


if __name__ == "__main__":
    main()
