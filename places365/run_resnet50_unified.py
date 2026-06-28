# Places365 scene recognition with ResNet50 on a single video frame.
# wideresnet18 전용이던 run_placesCNN_unified.py를 resnet50용으로 옮긴 버전.
# scene attributes는 wideresnet18(512차원) 전용 가중치라 resnet50(2048차원)에선 제외함.

import os
from typing import List, Tuple
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.models as models
from torchvision import transforms as trn
from torchvision import io as tv_io
import cv2
from PIL import Image

# ----- 설정값: 경로/프레임만 바꾸면 다른 입력으로 검증 가능 -----
PLACES_DIR: str = "/home/sangheon/Desktop/SFX/places365"
CHECKPOINT: str = os.path.join(PLACES_DIR, "resnet50_places365.pth.tar")
CATEGORY_FILE: str = os.path.join(PLACES_DIR, "categories_places365.txt")  # 365개 장면 이름
IO_FILE: str = os.path.join(PLACES_DIR, "IO_places365.txt")               # 실내(1)/실외(2) 라벨
VIDEO_PATH: str = "/home/sangheon/Desktop/SFX/video_0.mp4"
FRAME_TIME_SEC: float = 1.0                                               # 몇 초 지점 프레임을 볼지
CAM_OUT: str = os.path.join(PLACES_DIR, "cam_resnet50.jpg")              # CAM 결과 저장 경로
NUM_CLASSES: int = 365

# hook이 채워 넣을 중간 특징 저장소(전역). load_model에서 layer4에 hook을 건다.
features_blobs: List[np.ndarray] = []


def load_labels() -> Tuple[Tuple[str, ...], np.ndarray]:
    """장면 카테고리 이름과 실내/실외 라벨을 로컬 파일에서 읽는다."""
    # categories_places365.txt: "/a/airport_terminal 0" 형식 → 앞 3글자("/a/")를 떼고 이름만.
    classes: List[str] = []
    with open(CATEGORY_FILE) as class_file:
        for line in class_file:
            classes.append(line.strip().split(" ")[0][3:])

    # IO_places365.txt: 마지막 숫자가 1이면 실내, 2면 실외 → 0/1로 바꿔 저장.
    labels_IO: List[int] = []
    with open(IO_FILE) as f:
        for line in f.readlines():
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 = indoor, 1 = outdoor
    return tuple(classes), np.array(labels_IO)


def returnTF() -> trn.Compose:
    """Places365 입력 규격(224x224 리사이즈 + ImageNet 정규화) 변환을 만든다."""
    return trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def hook_feature(module, input, output) -> None:
    """forward 중 layer4의 출력 텐서를 numpy로 떠서 features_blobs에 쌓는다."""
    # output: (1, 2048, 7, 7) → squeeze → (2048, 7, 7). CAM 계산에 쓴다.
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnCAM(feature_conv: np.ndarray, weight_softmax: np.ndarray, class_idx: List[int]) -> List[np.ndarray]:
    """특정 클래스에 대한 Class Activation Map을 256x256으로 만들어 반환한다."""
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape  # resnet50: (2048, 7, 7)
    output_cam: List[np.ndarray] = []
    for idx in class_idx:
        # fc 가중치(해당 클래스 행) · 특징맵 = 위치별 기여도 맵.
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)        # 0 이상으로 시프트
        cam_img = cam / np.max(cam)    # 0~1 정규화
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def load_model() -> torch.nn.Module:
    """torchvision resnet50 골격에 Places365 체크포인트를 얹어 로딩한다."""
    # weights 없이 빈 resnet50을 만들되 출력 클래스 수를 365로 맞춘다.
    model = models.resnet50(num_classes=NUM_CLASSES)

    # 체크포인트 로드. 학습이 DataParallel로 돼 있어 키에 'module.' 접두사가 붙어 있다 → 제거.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # CAM을 만들려면 마지막 conv 블록(layer4)의 출력이 필요 → forward hook 등록.
    model.layer4.register_forward_hook(hook_feature)
    return model


def extract_frame(video_path: str, time_sec: float) -> Image.Image:
    """영상에서 time_sec 지점 프레임 1장을 PIL(RGB) 이미지로 뽑는다."""
    # [time_sec, time_sec+0.2] 구간만 디코딩해서 첫 프레임을 쓴다.
    frames, _audio, _info = tv_io.read_video(
        video_path, start_pts=time_sec, end_pts=time_sec + 0.2, pts_unit="sec"
    )
    if frames.shape[0] == 0:
        raise ValueError(f"{video_path} 의 {time_sec}s 지점에서 프레임을 못 찾았습니다.")
    first_frame = frames[0].numpy()  # (H, W, C), RGB, 0~255
    return Image.fromarray(first_frame)


def main() -> None:
    # 1) 라벨 + 모델 + 변환 준비
    classes, labels_IO = load_labels()
    model = load_model()
    tf = returnTF()

    # 2) fc 가중치(softmax 직전)를 꺼내 CAM용으로 음수는 0으로 클리핑.
    #    params[-1]=fc.bias, params[-2]=fc.weight (shape: 365 x 2048)
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax < 0] = 0

    # 3) 영상 프레임 1장 추출 → 전처리 → 배치 차원 추가.
    frame_rgb = extract_frame(VIDEO_PATH, FRAME_TIME_SEC)
    input_img = tf(frame_rgb).unsqueeze(0)  # (1, 3, 224, 224)

    # 4) forward → softmax → 확률 내림차순 정렬.
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    print(f"RESULT ON {VIDEO_PATH} @ {FRAME_TIME_SEC}s")

    # 5) 실내/실외 판정: 상위 10개 예측의 IO 라벨 평균으로 투표.
    io_image = np.mean(labels_IO[idx[:10]])
    print("--TYPE OF ENVIRONMENT:", "indoor" if io_image < 0.5 else "outdoor")

    # 6) 장면 카테고리 Top-5.
    print("--SCENE CATEGORIES:")
    for i in range(0, 5):
        print("{:.3f} -> {}".format(probs[i], classes[idx[i]]))

    # 7) Top-1 클래스에 대한 CAM 생성 후 원본 프레임에 겹쳐 저장.
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)  # cv2는 BGR을 쓴다
    height, width, _ = bgr.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + bgr * 0.5
    cv2.imwrite(CAM_OUT, result)
    print(f"Class activation map saved as {CAM_OUT}")


if __name__ == "__main__":
    main()
