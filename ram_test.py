from typing import Tuple, Any
from PIL import Image
import numpy as np
from torchvision import io as tv_io

from ram.models import ram_plus, ram, tag2text  # noqa: F401  (ram/tag2text는 비교용으로 남겨둠)
from ram import inference_ram, get_transform

# ----- 설정값: 여기만 바꾸면 다른 프레임/모델로 검증 가능 -----
VIDEO_PATH: str = "video_0.mp4"               # 프레임을 뽑아올 영상
FRAME_TIME_SEC: float = 1.0                   # 영상에서 몇 초 지점의 프레임을 볼지

RAM_CHECKPOINT: str = "ram_plus_swin_large_14m.pth"  # ★ 실제 체크포인트(.pth) 경로로 바꿔주세요
IMAGE_SIZE: int = 384                         # RAM이 학습된 입력 해상도(384가 표준)
VIT_BACKBONE: str = "swin_l"                  # ram_plus_swin_large 체크포인트에 맞는 백본


def extract_frame(video_path: str, time_sec: float) -> Image.Image:
    """영상에서 time_sec 지점의 프레임 1장을 PIL 이미지로 뽑는다."""
    # read_video: [start_pts, end_pts] 구간만 디코딩한다(pts_unit="sec" → 초 단위).
    # end를 살짝 뒤(+0.2s)로 줘서 그 구간 안의 첫 프레임을 확실히 잡는다.
    frames, _audio, _info = tv_io.read_video(
        video_path, start_pts=time_sec, end_pts=time_sec + 0.2, pts_unit="sec"
    )
    # frames: (T, H, W, C) uint8 텐서. 구간 안의 첫 프레임만 쓴다.
    if frames.shape[0] == 0:
        raise ValueError(f"{video_path} 의 {time_sec}s 지점에서 프레임을 못 찾았습니다.")
    first_frame: np.ndarray = frames[0].numpy()  # (H, W, C), 값 범위 0~255
    # numpy 배열을 PIL 이미지로 변환한다(RAM의 전처리 transform이 PIL을 받기 때문).
    return Image.fromarray(first_frame)


def load_ram_model(checkpoint: str, image_size: int, vit: str) -> Tuple[Any, Any]:
    """RAM++ 모델과 전처리 transform을 로딩해서 반환한다."""
    # get_transform: RAM 입력 규격(리사이즈 + 정규화)에 맞는 torchvision transform.
    transform = get_transform(image_size=image_size)
    # ram_plus: 체크포인트를 불러와 모델을 만든다. image_size/vit는 체크포인트와 일치해야 한다.
    model = ram_plus(pretrained=checkpoint, image_size=image_size, vit=vit)
    model.eval()          # 추론 모드(드롭아웃 등 끔)
    model = model.to("cuda")
    return model, transform


def recognize_tags(image: Image.Image, model: Any, transform: Any) -> str:
    """이미지 1장을 RAM++에 넣어 인식된 태그 문자열을 반환한다."""
    # transform(image): PIL → 텐서(C,H,W). unsqueeze(0): 배치 차원 추가 → (1,C,H,W).
    image_tensor = transform(image).unsqueeze(0).to("cuda")
    # inference_ram: (영어 태그, 중국어 태그) 튜플을 반환한다. 태그는 " | "로 구분.
    english_tags, _chinese_tags = inference_ram(image_tensor, model)
    return english_tags


def main() -> None:
    # 1) 영상에서 프레임 1장 추출
    frame: Image.Image = extract_frame(VIDEO_PATH, FRAME_TIME_SEC)

    # 2) RAM++ 모델 로딩
    model, transform = load_ram_model(RAM_CHECKPOINT, IMAGE_SIZE, VIT_BACKBONE)

    # 3) 그 프레임에서 태그 인식
    tags: str = recognize_tags(frame, model, transform)

    # 4) 결과 출력
    print(f"\n=== RAM++ recognized tags ({VIDEO_PATH} @ {FRAME_TIME_SEC}s) ===")
    print(tags)


if __name__ == "__main__":
    main()
