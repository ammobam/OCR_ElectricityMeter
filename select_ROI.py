# 원본 사진에서 마우스로 ROI를 선택하여 좌표를 csv로 저장하는 소스코드

# 패키지 임포트
import os
import cv2
import csv


# 디렉토리 내의 파일이름을 리스트로 가져오는 클래스
class Dir2File:

    # 디렉토리 경로 설정
    def __init__(self, file_path):
        self.file_path = file_path   # 이미지 들어있는 폴더명
        self.file_names = os.listdir(self.file_path)   # 이름 받아오기

    # 파일이름 리스트 만들기
    def Filename(self):
        src_names = []
        for name in self.file_names:
            # 이름에서 확장자 제외
            src_name, _jpg = name.split('.')
            src_names.append(src_name)
        return src_names


# 이미지에서 마우스로 ROI를 추출하고 esc 키를 누르면 좌표가 csv 파일로 저장되는 클래스
class ROI2csv:

    # ROI 좌표 저장할 디렉토리 설정
    def __init__(self, roi_path):
        self.roi_path = roi_path

    # 디렉토리에 있는 n번 이미지부터 m번 이미지까지 ROI 추출 수행
    def roi2csv(self, file_path, n, m):
        dir2File = Dir2File(file_path=file_path)
        file_path = dir2File.file_path
        src_names = dir2File.Filename()

        # 이미지 데이터 불러오기
        for src_name in src_names[n:m]:
            src_file = file_path + '/' + src_name + '.jpg'
            src = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)

            # 이미지 resize하기
            src_height,src_width = src.shape[:2]
            ## 이미지 가로길이가 700이 되도록 설정함
            ratio = 700/src_width
            src_height, src_width = int(src_height*ratio), int(src_width*ratio)
            ## 파라미터 입력 시에는 가로, 세로 순서로 입력
            src = cv2.resize(src, (src_width, src_height))

            while True:
                # 이미지에서 원하는 template 영역을 드래그하면 좌표를 받아오기
                roi = cv2.selectROI(src)

                # roi 확인
                print("roi: ", roi, "(x, y, width, height)")

                # 좌표로 변환
                x1 = roi[0]
                y1 = roi[1]
                x2 = roi[0] + roi[2]
                y2 = roi[1] + roi[3]
                coordinates = [x1, y1, x2, y2]

                # 사각형 그리기
                rect = cv2.rectangle(src, (x1, y1), (x2, y2), (0,0,255), thickness=3)
                cv2.imshow('roi', rect)

                # 3. 좌표를 csv 파일로 저장하기. 파일 이름은 '원본이미지이름.csv'
                roi_file = self.roi_path + '/' + src_name
                with open(roi_file, 'w', newline='') as out:
                    csv.writer(out).writerow(coordinates)

                # roi를 드래그하고 키보드 esc를 누르면 저장됨
                key = cv2.waitKey()
                if key == 27:  # esc 키
                    break

            cv2.destroyAllWindows()

# 실행 예시
# 1. roi csv를 저장할 디렉토리 이름 입력하기
ROI2csv = ROI2csv('./roi')
# 2. 원본 이미지 디렉토리, 읽어올 파일 인덱스 입력
# 3. 적절한 roi를 드래그 한 다음 esc를 눌러 다음 파일 작업
ROI2csv.roi2csv('./img_src_0906', 0, 3)
