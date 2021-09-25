'''
# 사용방법 예시코드

# 1. 이미지 좌표 추출
from select_ROI import ROI2csv
ROI2csv = ROI2csv('data/ElectricityMeter_ROI_screen') # roi 좌표를 저장할 디렉토리 입력하기
ROI2csv.roi2csv('./data/ElectricityMeter', 0, 20) # 읽어올 이미지의 디렉토리, 인덱스 처음, 끝 입력
# 적절한 roi 를 드래그하고 esc 를 눌러 좌표 추출을 반복함


# 2. 좌표를 이미지에 표시
from select_ROI import ROI2Img
r = ROI2Img(roi_path='./data/roi') # csv 디렉토리 입력
r.roi2img('./data/Electricitymeter', 0, 20) # 읽어올 이미지의 디렉토리, 인덱스 처음, 끝 입력

'''



# 패키지 임포트
import os
import cv2
import csv


# 예외처리 클래스
class MyError(Exception):
    def __init__(self, msg='init_error_msg'):
        self.msg = msg

    def __str__(self):
        return self.msg


# 디렉토리 내의 파일이름을 리스트로 가져오는 클래스
class Dir2File:
    # 디렉토리 경로 설정
    def __init__(self, file_path):
        self.file_path = file_path  # 이미지 들어있는 폴더명
        self.file_names = os.listdir(self.file_path)  # 이름 받아오기

    # 파일이름 리스트 만들기
    def filename(self):
        src_names = []
        for name in self.file_names:
            # 이름에서 확장자 제외
            src_name, _jpg = name.split('.')
            src_names.append(src_name)
        return src_names


# 이미지에서 마우스로 ROI 를 추출하고 esc 키를 누르면 좌표가 csv 파일로 저장하는 클래스
class ROI2csv:
    # ROI 좌표 저장할 디렉토리 설정
    def __init__(self, roi_path):
        self.roi_path = roi_path

    # 디렉토리에 있는 n번 이미지부터 m번 이미지까지 ROI 추출 수행
    def roi2csv(self, file_path, n, m):

        try:
            if not os.path.exists(file_path):
                raise MyError(f"{file_path}는 없는 경로입니다. 이미지 경로를 확인하십시오.")
            if not os.path.exists(self.roi_path):
                raise MyError(f"{self.roi_path}는 없는 경로입니다. 좌표를 저장할 경로를 확인하십시오.")

        except MyError as e:
            print(e)

        # 디렉토리가 존재하는 경우 실행
        else:
            dir2file = Dir2File(file_path=file_path)
            file_path = dir2file.file_path
            src_names = dir2file.filename()

            # 이미지 데이터 불러오기
            for i, src_name in enumerate(src_names[n:m]):
                src_file = file_path + '/' + src_name + '.jpg'
                print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인
                src = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)

                # 이미지 resize 하기
                src_height, src_width = src.shape[:2]
                # 이미지 가로길이가 700이 되도록 설정함
                ratio = 700 / src_width
                src_height, src_width = int(src_height * ratio), int(src_width * ratio)
                # 파라미터 입력 시에는 가로, 세로 순서로 입력
                src = cv2.resize(src, (src_width, src_height))

                while True:
                    # 이미지에서 원하는 template 영역을 드래그하면 좌표를 받아오기
                    # roi --> (x, y, width, height)
                    roi = cv2.selectROI("src", src)

                    # 좌표로 변환
                    x1 = roi[0]
                    y1 = roi[1]
                    x2 = roi[0] + roi[2]
                    y2 = roi[1] + roi[3]
                    coordinates = [x1, y1, x2, y2]

                    # roi 좌표 확인
                    print("- roi 좌표: ", coordinates, "[x1, y1, x2, y2]")

                    # 사각형 그리기
                    rect = cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                    cv2.imshow('roi', rect)

                    # 좌표를 csv 파일로 저장하기. 파일 이름은 '원본이미지이름.csv'
                    roi_file = self.roi_path + '/' + src_name
                    with open(roi_file, 'w', newline='') as out:
                        csv.writer(out).writerow(coordinates)

                    # roi 를 드래그하고 키보드 esc 를 누르면 다음 이미지 작업 수행함
                    key = cv2.waitKey()
                    if key == 27:  # esc 키
                        break

                cv2.destroyAllWindows()


# 이미지에 ROI 좌표를 표시하는 클래스
class ROI2Img:
    # ROI 좌표 가져올 디렉토리 설정
    def __init__(self, roi_path):
        self.roi_path = roi_path

    # roi 좌표 읽어오기
    def read_roi(self, src_name):
        # csv 파일 읽어오기
        roi_file = self.roi_path + '/' + src_name
        with open(roi_file, 'r', newline='') as coordinates:
            coo_obj = csv.reader(coordinates, delimiter=',')  # reader object
            # 객체 내부 요소는 파일이 열려있는 상태에서만 꺼낼 수 있으므로 with 구문 안에서 꺼냄
            # 객체의 요소를 리스트로 만들어주고, 이중리스트를 리스트로 풀어냄
            coo = sum([c for c in coo_obj], [])
            # 리스트 안의 문자열을 숫자로 바꿈
            coo = [int(a) for a in coo]
            return coo

    # 디렉토리에 있는 n번 이미지부터 m번 이미지까지 ROI 좌표 출력 수행
    def roi2img(self, file_path, n, m):

        try:
            if not os.path.exists(file_path):
                raise MyError(f"{file_path}는 없는 경로입니다. 이미지 경로를 확인하십시오.")
            if not os.path.exists(self.roi_path):
                raise MyError(f"{self.roi_path}는 없는 경로입니다. 좌표를 저장할 경로를 확인하십시오.")

        except MyError as e:
            print(e)

        # 디렉토리가 존재하는 경우 실행
        else:
            dir2file = Dir2File(file_path=file_path)
            file_path = dir2file.file_path
            src_names = dir2file.filename()

            # 이미지 데이터 불러오기
            for i, src_name in enumerate(src_names[n:m]):
                src_file = file_path + '/' + src_name + '.jpg'
                print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인
                src = cv2.imread(src_file, cv2.IMREAD_COLOR)

                # 이미지 resize 하기
                src_height, src_width = src.shape[:2]
                # 이미지 가로길이가 700이 되도록 설정함
                ratio = 700 / src_width
                src_height, src_width = int(src_height * ratio), int(src_width * ratio)
                # 파라미터 입력 시에는 가로, 세로 순서로 입력
                src = cv2.resize(src, (src_width, src_height))

                while True:
                    # 좌표 읽어오기
                    coo = self.read_roi(src_name)
                    x1, y1, x2, y2 = coo

                    # roi 영역에 사각형 그리기
                    rect = cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                    cv2.imshow('roi', rect)

                    # 계량기 데이터 영역에 사각형 그리기
                    width = x2 - x1
                    height = y2 - y1
                    x1_ = int(x1 - 0.4*width)
                    x2_ = int(x2 + 0.4*width)
                    y1_ = int(y1 - 0.7*height)
                    y2_ = int(y2 + 3*height)
                    rect = cv2.rectangle(src, (x1_, y1_), (x2_, y2_), (0, 0, 255), thickness=2)
                    cv2.imshow('roi', rect)

                    # roi 를 드래그하고 키보드 esc 를 누르면 다음 이미지 작업 수행함
                    key = cv2.waitKey()
                    if key == 27:  # esc 키
                        break

                cv2.destroyAllWindows()
