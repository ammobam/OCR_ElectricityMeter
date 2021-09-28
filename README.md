(🐥진행 중인 프로젝트입니다. 작성 중인 README입니다.)

# OCR Project

## [🐥프로젝트 수행일지 (click)](https://ammobam.github.io/categories/#ocr-project)

## Repository 구조
```
OCR Project
├── 📁OCR_ElectricityMeter_imgprep
|	├── 📁src
|	|	├── 📃select_ROI.ipynb
|	|	├── 📃img_prep.ipynb
|	|	└── 📃(🐥추가예정)
|	└── 📁data/roi (ROI 좌표 파일)
└── 📁OCR_ElectricityMeter_android 
```

## 목차

  * [프로젝트 개요](#프로젝트-개요)
    + [프로젝트 목적](#프로젝트-목적)
    + [프로젝트 목표](#프로젝트-목표)

## 분석 환경 및 도구
- **HW/Server**
	- Window (ver 10 / CPU i7 / RAM 16)
	- Ubuntu (ver 18)
- **Language**
	- Python (ver 3.7)
- **Tools**
	- Github
	- Slack
	- Google Drive
- **IDE**
	- PyCharm
	- AndroidStudio
- **Library**
	- OpenCV (ver 4.5)
	- Tensorflow (🐥)
	- Matplotlib (ver 3.2)
	- Numpy (ver 1.19)


## 프로젝트 개요
### 프로젝트 목적
- 전기계량기와 모뎀 이미지로부터 OCR을 수행하여 제조번호, 바코드 등을 인식하고자 함
- 개인 디바이스로 전기계량기와 모뎀을 촬영하여 이미지를 서버에 전송하면 계량기 타입, 제조번호, 모뎀 바코드를 문자로 읽어 내어 json 형식으로 추출하여 저장함
- 추후 계량기와 모뎀의 1:1 관리하는 서비스 등에 이용될 예정임

### 프로젝트 목표
1. 문자 검출(ROI) 평가 목표 : IoU 0.5 이상
2. 문자 인식 평가 목표 : 1 - NED 0.8 이상
3. End-to-End(검출+인식) 평가 방법
	- 문자 검출, 인식에 대해 순차적으로 평가함


<details>
<summary> 평가지표 배경지식📌(펼치기)</summary>
<div markdown="1">   

- ROI 평가 지표 : IoU
	- Intersection over Union
	- 이미지에서 문자영역 검출(ROI)에 대한 평가 지표
	- 정답과 예측 박스 겹치는지 확인

- 문자인식 평가 지표
	- 1 - NED
		- Normalized Edit Distance
		- 정답과 예측 단어에서 글자의 일치율
		- 정답과 예측 단어 간 편집거리를 측정한 뒤 긴 단어의 길이로 정규화함 (글자기반)
		- 편집거리?
            - 두 문자열의 유사도를 판단하는 기준
            - 어떤 문자열이 다른 문자열과 같아지도록 할 때 삽입, 삭제, 변경을 수행하는 횟수의 최소값
	- WEM
		- Word based Exactly Matching
		- 정답과 예측 단어가 정확히 일치하는지 확인함 (단어기반)
		- 전체 글자(단어)의 일치도를 1, 0으로 평가함

</div>
</details>


### 프로젝트 기간
- 데이터 수집
- 데이터 전처리 (1주)
- 이미지에서 ROI 추출(1주)
- ROI에서 문자 인식(1주)
- 모델 학습(1주)
- 모델 개선을 위한 데이터 가공(1주)
- 웹서비스 구축(1주)

### 업무 분장
- 모든 팀원은 이미지 처리 및 모델링 학습을 공통 목표로 하고 있음
- 모든 팀원이 동일한 과정을 수행함


## 데이터 탐색
- 데이터 : 전기계량기 및 모뎀 이미지 파일
- [전기계량기] 전기계량기 타입, 전력량, 상세종류, 모델명, 제조번호
- [모뎀] 바코드
	- 추출해서 저장할 데이터는 바코드 이미지? 숫자로 변환한 데이터?

## 전체 과정

1. 전기계량기 및 모뎀의 형태 추출
	- 이미지 전처리
		- 회전된 이미지 수평 맞추기
		- 그외 형태 추출을 위한 전처리 작업 수행
	- 형태 ROI 추출 및 좌표 저장

2. 형태 영역에서 데이터 영역 추출
	- 이미지 전처리
		- 데이터 영역 추출을 위한 전처리 작업 수행
	- 데이터 ROI 추출 및 좌표 저장

- ROI 추출 예시 그림 


	<img src="/ROI추출예시.jpg"  width="500" height="447">


3. 데이터 영역에서 문자 인식
	- 이미지 전처리
		- 모델링을 위한 이미지 사이즈 조정 (비율?)
		- 문자 인식을 위한 전처리 작업 수행 (이진화 작업)
	- OCR 모델 학습 및 예측
		- 문자 이미지에서 문자를 인식하는 모델(OCR) 생성
        - 이미지에서 추출해야 할 데이터는 숫자, 영어, 한글, 특수기호(괄호, 하이픈 등)임
        - 숫자는 10가지, 한글, 영어는 특정 단어만 등장함
        - 전기계량기 타입별로 데이터 포맷(길이, 하이픈 등)이 정해져 있음
        - 영어, 한글, 특수문자의 경우 패턴(자주 등장하는 글자)이 정해져 있음

4. 레이블 작성
	- 형태 영역 좌표, 데이터 영역 좌표, 인식한 문자 데이터를 원본 이미지와 비교하여 수정한 것으로 레이블 데이터를 작성함
	- 각 ROI 좌표는 x1, y1, x2, y2 4개의 정수로 저장함
	- json 파일로 작성함
    - 예시
	``` json
    img_label = {
        "meter_img": {"file_name": "계량기이미지파일이름.jpg", "width": 211, "height": 216,
                      "meter_type": "G-Type",
                      "wattage": 1623.05,
                      "meter_detail": "교류단상2선식",
                      "model_name": "NJ12-210-GEN(특수형)",
                      "serial_number": "08-25-0130056-1509"}},{
        "modem_img": {"file_name": "모뎀이미지파일이름.jpg", "width": 200, "height": 230,
                      "modem_barcode": "EP1544124470"},
        "meter_ROI": [{
            "device_ROI": [0,0,50,70],
            "data_ROI": [[meter_type좌표], [wattage좌표], [meter_detail좌표], [model_name좌표], [serial_number좌표]]}],
        "modem_ROI":[0,0,50,70]
    }
	```

5. 문자 인식 모델 평가
	- 1-NED 0.8 이상을 목표로 함

6. 문자 인식 모델 개선을 위한 데이터 가공
	- GAN을 이용한 이미지 데이터 생성

7. 문자 인식 모델 개선
8. 웹 서비스 제작

