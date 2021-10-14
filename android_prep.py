# 안드로이드에서 받은 이미지 파일 처리 클래스

import cv2
import os


class prep:

    # 파일경로 설정
    def __init__(self, file_path):
        self.file_path = file_path  # 이미지 들어있는 파일경로
        self.file_names = os.listdir(file_path)  # 이미지 파일 이름
        self.src_name = sum([name.split('.')[:1] for name in self.file_names], [])[-1]  # 이미지 파일이름
        self.src_file = self.file_path + '/' + self.src_name + '.jpg'

    # 이미지 전처리 후 저장 경로 생성
    def create_path(self, file_path):
        try:
            if not os.path.exists(self.file_path):
                os.makedirs(self.file_path)
        except OSError:
            print(f"{self.file_path}는 없는 경로입니다. 해당 이미지의 text_roi 폴더를 생성합니다.")


    # 안드로이드에서 전송한 이미지 파일 ROI 영역 탐지
    # return 딕셔너리 - key: 이미지 이름, value: 8-segment영역 정보 튜플 (x, y, w, h)
    def find_8seg(self):

        src_roi = dict()    # 좌표를 저장할 딕셔너리 생성
        src_file = self.file_path + '/' + self.src_name + '.jpg'
        print(f"- 수행파일:{src_file}")  # 확인

        # 좌표를 저장할 경로 생성
        roi_path = './data/show/' + self.src_name
        self.create_path(roi_path)


        ## 이미지 읽어오기 - GrayScale
        src = cv2.imread(self.src_file, cv2.IMREAD_GRAYSCALE)



        ## Histogram equalization
        src = cv2.equalizeHist(src)
        # get.img_check(src)


        ## Binary
        # parameter
        max_val = 255
        C = -10
        bin = 10    # block_size 홀수
        if (src.shape[1] // bin) % 2 == 0:
            block_size = src.shape[1] // bin + 1
        else:
            block_size = src.shape[1] // bin

        src = cv2.adaptiveThreshold(src, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                           block_size, C)
        # get.img_check(src)
        cv2.imwrite(roi_path + '/' + self.src_name + '_binary' + '.jpg', src)   # 저장


        ## contouring
        contours, hierarchy = cv2.findContours(src, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)
        src_check = cv2.imread(self.src_file, cv2.IMREAD_COLOR)
        src_contour = cv2.drawContours(src_check, contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
        cv2.imwrite(roi_path + '/' + self.src_name + '_contour' + '.jpg', src_contour)   # 저장
        # get.img_check(src_contour)


        # 추려낸 contours 대상으로 사각형 그리기
        data_roi_coo = set()   # 좌표 저장할 set

        for con in contours:
            perimeter = cv2.arcLength(con, True)    # 외곽선 둘레 길이

            # x, y, w, h의 namespace 설정
            x = 0
            y = 0
            w = 0
            h = 0

            for epsilon in range(100, 200):
                epsilon = epsilon / 1000

                # 빠른 객체탐지를 위해 작은 값 무시
                if epsilon * perimeter < 20.0:
                    break

                # 외곽선 근사화하여 좌표 반환
                approx = cv2.approxPolyDP(con, epsilon * perimeter, True)

                # 4개의 코너를 가지는 Edge Contour에 대해 사각형 추출
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(con)        # 좌표를 감싸는 최소면적 사각형 정보 반환


                    src_w = src.shape[1]
                    src_h = src.shape[0]

                    # 8-segment ROI 특정
                    if (w < 0.2 * src_w) or (w > 0.6 * src_w) or (h > 0.2 * src_h) or (w / h > 4) or (
                            w / h < 1.7):  # 노이즈 무시
                        break

                    # ROI 좌표를 튜플로 저장함
                    data_roi_coo.add((x, y, w, h))

                    # 사각형 그리기
                    rect = cv2.rectangle(src_check, (x, y), (x+w, y+h), (255, 0, 255), thickness=2)
                    cv2.imwrite(roi_path + '/' + self.src_name + '_rect' + '.jpg', rect)  # 저장
                    # cv2.imshow('roi', rect)
                    # key = cv2.waitKey()
                    # if key == 27:
                    #     break

        src_roi[self.src_name] = data_roi_coo

        return src_roi


    # ROI를 탐지하지 못한 경우의 처리
    def no_box(self):
        # 이미지 전처리 결과 딕셔너리 가져오기
        src_roi = self.find_8seg()

        print(f"-----------------------------ROI of {self.src_name}-----------------------------")

        if len(src_roi[self.src_name]) == 0:
            print("ROI를 탐지하지 못했습니다. 다시 촬영해주십시오.")
        elif len(src_roi[self.src_name]) == 2:
            print("ROI가 2개 탐지하였습니다. 첫번째 ROI를 기준으로 OCR 수행합니다.")
        else:
            print(src_roi[self.src_name])


