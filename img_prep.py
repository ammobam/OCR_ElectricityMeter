from select_ROI import ROI2csv

# 1. roi csv 를 저장할 디렉토리 이름 입력하기
ROI2csv = ROI2csv('./dd')
# 2. 원본 이미지 디렉토리, 읽어올 파일 인덱스 입력
# 3. 적절한 roi 를 드래그 한 다음 esc 를 눌러 다음 파일 작업
# ROI2csv.roi2csv('./data/ElectricityMeter', 0, 3)
# 4. 추출된 ROI 좌표 x1, y1, x2, y2 확인

import os
directory = './dd'
try:
    if not os.path.exists(directory):
        os.makedirs(directory)
except OSError:
    print('Error: Creating directory. ' + directory)