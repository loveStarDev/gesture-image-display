# Gesture Image Display

MediaPipe를 사용해 웹캠으로 손동작을 실시간 인식하고, 지정한 이미지를 별도 창에 표시하는 프로그램입니다.

---

## 프로젝트 구조

```
gesture-image-display/
├── main.py                 # 메인 프로그램
├── hand_landmarker.task    # MediaPipe 손 인식 모델
├── requirements.txt        # Python 패키지 목록
├── 실행.bat                # 더블클릭 실행 파일 (Windows)
└── images/                 # 손동작별 이미지 폴더
    ├── 1.jpg / 1.jpeg      # 1번 손동작 이미지
    ├── 2.jpg / 2.jpeg      # 2번 손동작 이미지
    ├── 3.jpg / 3.jpeg      # 3번 손동작 이미지
    └── 4.jpg / 4.jpeg      # 4번 손동작 이미지
```

---

## 요구사항

- Python 3.11 (3.12 이상은 mediapipe 미지원)
- 웹캠

---

## 설치

### 1. Python 3.11 설치

winget으로 자동 설치:
```cmd
winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
```

또는 [python.org](https://python.org/downloads) 에서 Python 3.11 다운로드 후 설치
- 설치 시 **"Add python.exe to PATH"** 반드시 체크

### 2. 패키지 설치

```cmd
cd C:\Workspace\repository\gesture-image-display
pip install -r requirements.txt
```

### 3. MediaPipe 모델 다운로드

`hand_landmarker.task` 파일이 없는 경우:
```cmd
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -o hand_landmarker.task
```

---

## 실행

### 방법 1 - 배치 파일 (권장)
`실행.bat` 더블클릭

### 방법 2 - 명령 프롬프트
```cmd
cd C:\Workspace\repository\gesture-image-display
python main.py
```

---

## 화면 구성

| 창 이름 | 설명 |
|---------|------|
| `Gesture Cam` | 웹캠 화면 + 손 랜드마크 + 하단 상태바 |
| `Image` | 인식된 손동작에 해당하는 이미지 (별도 창) |

- 손동작 인식 시 → `Image` 창에 이미지 즉시 표시
- 손을 치우거나 제스처 미감지 시 → `Image` 창 즉시 닫힘
- 하단 상태바: 현재 감지된 제스처명 + 타이머 바 표시
- `Q` 키로 종료

---

## 손동작 목록

| 번호 | 이미지 파일 | 손동작 | 인식 조건 |
|------|------------|--------|-----------|
| 1 | `images/1.*` | 호랑이 어흥 | 손바닥을 카메라로 향하고 4손가락(검지~새끼) 모두 구부리기 |
| 2 | `images/2.*` | V / 브이 | 검지+중지만 펼치기 (손바닥/손등 방향 무관) |
| 3 | `images/3.*` | 엄지 치켜들기 | 엄지손가락만 펼치기 (방향 무관) |
| 4 | `images/4.*` | ㅜ자 / 총 모양 | 엄지+검지만 펼치고 나머지 말기 (방향 무관) |

---

## 이미지 교체

`images/` 폴더에 원하는 이미지를 넣으면 됩니다.

지원 확장자: `.jpeg`, `.jpg`, `.png`, `.bmp`, `.webp`

```
images/1.jpeg  ← 1번 손동작에 표시할 이미지
images/2.png   ← 2번 손동작에 표시할 이미지
...
```

이미지 파일이 없으면 해당 손동작은 이미지 창이 뜨지 않습니다.

---

## 손동작 추가/수정

`main.py` 상단의 `GESTURE_IMAGE_MAP`과 `GESTURE_LABELS`에 항목을 추가하고,
`detect_gesture()` 함수에 감지 조건을 작성하면 됩니다.

```python
GESTURE_IMAGE_MAP = {
    "claw":     "images/1",
    "peace":    "images/2",
    "thumb_up": "images/3",
    "gun":      "images/4",
    "my_gesture": "images/5",  # 새 손동작 추가
}

# detect_gesture() 안에 조건 추가
if not thumb and not index and not middle and ring and pinky:
    return "my_gesture"
```

---

## 기술 스택

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.11 |
| 손 인식 | MediaPipe Hand Landmarker (Tasks API) |
| 영상 처리 | OpenCV |
| 손가락 판별 방식 | 3D 랜드마크 좌표 거리 기반 (tip-mcp vs pip-mcp) |
| 손바닥/손등 판별 | 손목-중지MCP x좌표 관계로 판별 |

### 손가락 펼침 판별 원리

MediaPipe는 손의 21개 랜드마크 3D 좌표를 제공합니다.
각 손가락의 `tip(끝마디)`, `pip(중간마디)`, `mcp(손가락뿌리)` 간 3D 거리를 비교합니다.

```
tip-mcp 거리 > pip-mcp 거리 × 1.2  →  펼침
```

이 방식은 손의 방향(손바닥/손등/가로)에 무관하게 동작합니다.
