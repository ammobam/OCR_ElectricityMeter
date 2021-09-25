# # 1. 액정 부분 ROI 좌표를 추출 (완료: ./data/roi)
# # 2. 좌표를 기준으로 일정 데이터 영역을 추출할 수 있는 비율 찾기
# from select_ROI import Dir2File
# r = ROI2Img(roi_path='./data/roi')
# file_path = './data/Electricitymeter'
# dir2file = Dir2File(file_path=file_path)
# src_names = dir2file.filename()
#
# width_sum = 0
# height_sum = 0
# for i, src_name in enumerate(src_names[0:-1]):
#     x1, y1, x2, y2 = r.read_roi(src_name)
#     width = x2 - x1
#     height = y2 - y1
#
#     width_sum += width
#     height_sum += height
#
# print(f"액정 가로/세로 비율 평균:{width_sum/height_sum:.2f}") # 2.77



# 3. 데이터 영역 찾기

from select_ROI import ROI2Img
r = ROI2Img(roi_path='./data/roi')
r.roi2img('./data/Electricitymeter', 0, -1)
# 영역 바깥으로 설정된 사진들
# ./data/Electricitymeter/01232009721_P1107.jpg
# ./data/ElectricityMeter/01232010345_P1132.jpg


# 4. 이미지 전처리 > 침식/팽창 > contouring
# 5. OCR 각 데이터 영역 읽어보기
# 6. 모델 개선을 위해 각 contour 박스를 구간별 회전 수행
