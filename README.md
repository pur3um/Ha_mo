# Ha-mo

![dirty_hand2](https://user-images.githubusercontent.com/99007025/225613970-8545cbbf-b46f-49ae-993f-449269a0567d.jpg)

## Description

> **Ha-mo**는 일상 속에서 손을 사용하기 어려울 때, 모션만으로 원하는 기능을 수행하도록 돕는 프로그램입니다.
모드에 따라 최소 4개부터 최대 8개까지의 명령을 카메라에 손을 인식시키는 것만으로 수행할 수 있습니다.


### 1. 환경 설정
> - `requirements.txt`있는 디렉토리 안에서 명령프롬프트 실행
> - 가상환경 생성 및 지정 (파이썬 3.8버전으로)
> ```python
> # 가상환경 설정
> conda create -n test python=3.8
> conda activate test
> 
> # 라이브러리 install
> pip install -r requirements.txt
> ```
### 2. 기술 스택 및 모델 구조

![hamo_process](https://user-images.githubusercontent.com/99007025/225612686-ebd00865-4c20-4587-83b9-70a23ee2f754.PNG)



### 3. Ha-mo UI

[//]: # (![image]&#40;https://user-images.githubusercontent.com/77317312/124409338-1e13cb00-dd83-11eb-8c8d-13e56986a116.png&#41;)
![image](/hamo_ui.png)

### 4. Ha-mo 실행 설명

- method1. 
    - 제공되는 main.exe 파일 실행
- method2. 
    - `PlayData_Final_Project/deployment` 경로에서 IDE 실행
    - `main.py` 실행


### 5. 디렉토리 구조

```bash
├── deployment                            - 배포용 디렉토리
│
├── pre_productVersion
│         │
│         ├── common  -> [model, src에 공통으로 쓰이는 module]
│         │    │
│         │    ├── draw.py                - img 데이터 생성, 데이터 저장
│         │    └── HandTrackingModule.py  - 손 인식 모듈
│         │
│         ├── model   -> [transfer learning을 위한 디렉토리]
│         │    │
│         │    ├── data/                  - 학습용 데이터
│         │    ├── process/               - 각 모델별(VGG, MobilNetV2, LSTM) 생성 파일
│         │    ├── saved_model/           - 학습된 모델 저장 directory
│         │    └── preprocessing.py       - 학습 데이터 생성 file
│         │
│         └── src     -> [DryHand app기능 구현 모듈 디렉토리]
│              │
│              ├── test/                  - 개발을 위한 test directory
│              ├── AutopyClass.py         - window controller
│              └── GestureModule.py       - 모델 input 값 전처리, 추론 함수 정의
│
└── 제출용                                - 프로젝트 주요 파일 정리 디렉토리
```

### 6. Hand Motion Project 기능 설명


### 6-1. [mode 선택 기능]

1. 유튜브 모드 
    
    | 기능 | 제스처 |
    | -- | -- |
    | 10초 전/후 이동 | `<`, `>` |
    | 이전/이후 동영상 이동 | `ㅣ`, `ㅡ` |
    | 광고 스킵 | `Z` |
    | 유튜브 창 띄우기 | `Y` |
    | 일시정지/재생 | `S` |
    | 전체화면 전환/복귀| `W` |

2. 웹 브라우저 모드
    
    | 기능 | 제스처 |
    | -- | -- |
    | 이전/다음 페이지 이동 | `<`, `>` |
    | 크롬창 켜기/끄기 | `S`, `W` |

3. 프레젠테이션 모드
    
    | 기능 | 제스처 |
    | -- | -- |
    | 전/후 슬라이드 이동 | `<`, `>` |
    | 파워포인트 전체창 | `S` |
    | 전체창 나오기 | `W` |

### 6-2. [기타 편의 기능]

1. `Play Button` -> 동작인식 및 동작인식 확인용 디스플레이 시작
2. `Stop Button` -> 동작인식 및 디스플레이 중지
3. `프레임 수 조절` -> 초당 이미지의 수를 조절가능
4. `명암 조절` -> 웹캠에서 받아들이는 이미지의 명암 조절 가능


  
### 7. 참고 자료

- [virtual painter](https://www.youtube.com/watch?v=ZiwZaAVbXQo)
- [Real-Time Hand Gesture 논문](https://www.koreascience.or.kr/article/JAKO201919866854640.pdf)
- [PyQt5](https://wikidocs.net/book/2165)
- [PyAuto](https://pyautogui.readthedocs.io/en/latest/)
- [mediapipe doc](https://google.github.io/mediapipe/)
- [mediapipe](https://www.youtube.com/watch?v=WQeoO7MI0Bs&t=1371s)



























