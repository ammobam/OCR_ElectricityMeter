# 계량기 영역 도출 - 네 개 사각형 검출
# (1) 이미지 전처리 - 이진화, 경계선 또렷하게. 네 개 사각형 잘 나오게 그림자, 빛 처리
# (2) 네 개 사각형에 대한 영역 검출
# (3) 비율을 적용하여 계량기 영역 도출





# 데이터 영역에서 기울기 자동 변환을 위한 작업
# (1) 이미지 전처리 - 모폴로지. 문자영역이 잘 뭉쳐지는 커널, 파라미터 찾기.

from prep_OCR import OCR_prep
x = OCR_prep('./data/ElectricityMeter', './data/roi')
x.preprocess(0,2)


# (2) 기울기 기준 탐색
# (2)-1. contouring 수행 - cv2.approxPolyDP(contour, epsilon, closed)
# (2)-2. hough 변환을 이용한 직선 검출
# (2)-3. Form OCR을 이용한.......... 회전.
## https://www.pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/
# (3) 이미지 3분할하여 각 영역에서 기준 기울기 찾기
# (4) 이미지 3분할 하여 각 영역에 속한 문자를 기울기 변환
# (5) 각 영역에 OCR 수행하여 성능 확인





# OCR 모델 개선
# (1) 이미지 훈련 데이터 늘리기 - GAN
# (2) 이미지 훈련 데이터 늘리기 - 폰트별 이미지 데이터 생성
# (3) 적절한 전처리




# Android 앱
# (1) [선택] 촬영 - 이미지 서버 전송 - 서버에서 전처리 및 모델링 수행 - 디바이스 전송 - 출력
# (2) 다른방법 : 촬영 - 안드로이드 상에서 이미지 전처리 및 모델링 수행 - 출력
# 예상 이슈 : 속도, 자원의 효율적 이용 필요. 알고리즘 공부.



# 팀원 서포트한 코드
'''
# 객체 탐지 모델링 후 segmentation 복원 및 확인 코드
from select_ROI import Dir2File
from prep_img import GetImg
x = Dir2File('./data/segmentation')  # segmentation 경로 입력
src_names = x.filename()  # 이미지 이름 가져옴
y = Dir2File('./data/segmentation')  # 각 segmentation을 모델 수행하여 나온 결과물 경로 입력
model_outputs = y.filename()


for src_name, model_output in zip(src_names, model_outputs):
    x = GetImg(src_name)
    # 그레이스케일로 받아와서 이미지 사이즈 줄임
    src_resized = x.resize_to_origin(x.printGray(), model_output)

    # 여기서 뭔가 작업
'''