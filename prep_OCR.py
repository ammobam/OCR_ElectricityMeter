'''
# 예시코드

# OCR을 위한 이미지 전처리 수행 및 데이터 ROI 파일 저장
from prep_OCR import OCR_prep
x = OCR_prep('./data/ElectricityMeter', './data/roi')
x.preprocess(0,10)


# 전체 이미지 중 8-segment ROI를 탐지
from prep_OCR import OCR_prep
x = OCR_prep('./data/ElectricityMeter', './data/roi')
src_roi = x.preprocess_8seg(n, m)  # n-m번 이미지까지
# 전체 이미지 중 8-segment ROI을 탐지한 경우의 수
# 정말 잘 탐지했는지 하나하나 확인한 결과, 이미지 당  1개 또는 2개 영역을 추출함. ROI 는 반드시 포함되어 있음.
print("8segment ROI를 탐지한 이미지 개수:", sum([1 for x in src_roi.values() if len(x) != 0]), "전체: 789개")

'''

# OCR 수행을 위한 전처리

import cv2
import os

import numpy as np

from prep_img import GetImg
from select_ROI import Dir2File, ROI2Img, CreatePath #, MyError
import pytesseract



class OCR_prep:

    def __init__(self, file_path, roi_path):
        self.file_path = file_path
        self.roi_path = roi_path

    # work in ROI, not over the entire area
    # img width: 700
    def work_in_ROI_700(self, roi_filename, src):
        # Get ROI coordinates
        r = ROI2Img(self.roi_path)
        x1, y1, x2, y2 = r.read_roi(roi_filename)

        width = x2 - x1
        height = y2 - y1

        x1_ = int(x1 - 0.4 * width)
        x2_ = int(x2 + 0.4 * width)
        y1_ = int(y1 - 0.7 * height)
        y2_ = int(y2 + 3 * height)

        # return x1_, y1_, x2_, y2_
        return src[y1_:y2_, x1_:x2_]  # height, width


    def work_in_ROI_origin(self, roi_filename, src):
        # Get ROI
        r = ROI2Img(self.roi_path)

        # get ROI coordinates
        x1, y1, x2, y2 = r.read_roi(roi_filename)
        width = x2 - x1
        height = y2 - y1

        # calculate ROI coordinates in origin image
        x1_ = int(x1 - 0.4 * width)
        x2_ = int(x2 + 0.4 * width)
        y1_ = int(y1 - 0.7 * height)
        y2_ = int(y2 + 3 * height)

        # return x1_, y1_, x2_, y2_
        return src[y1_:y2_, x1_:x2_]  # height, width

        # Get original img size
        src_height, src_width = src.shape[:2]
        # print("------------", src_height, src_width, "-------------")

        # calculate ratio
        # tan = src_height / src_width  # 좌표변환 기준
        ratio = int(src_width / 700)    # 700 -> 원본 이미지 사이즈 변환 비율
        # ratio_h = int(700 * src_height / src_width) ############################


        x1_ = x1 * ratio
        y1_ = y1 * ratio
        x2_ = x1_ + (roi_width * ratio)
        y2_ = y1_ + (roi_height * ratio)
        # x2_ = x2 * ratio + roi_width * ratio
        # y2_ = y2 * ratio + roi_width * ratio

        # x1_ = int((x1 - 0.4 * roi_width) * ratio)
        # x2_ = int((x2 + 0.4 * roi_width) * ratio)
        # y1_ = int((y1 - 0.7 * roi_height) * ratio)
        # y2_ = int((y2 + 3 * roi_height) * ratio)

        # cropping ROI
        return src[y1_:y2_, x1_:x2_]  # height, width
    
    
    # 적절한 전처리 과정을 찾는 메소드
    # 이미지 가로 사이즈 700 기준으로 수행함
    def preprocess_700(self, n, m):
        # Get Img
        dir2file = Dir2File(file_path=self.file_path)
        file_path = dir2file.file_path
        src_names = dir2file.filename()
        r = ROI2Img(self.roi_path)

        for i, src_name in enumerate(src_names[n:m]):
            src_file = file_path + '/' + src_name + '.jpg'
            print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인

            ## GrayScaling
            x = GetImg(src_name)
            # image check
            src = x.resize_check(x.printGray())

            ## Binary
            # parameter
            # max_val = src.max()
            max_val = 255
            # C = 20
            C = 16 ## G-Type 영역 글자가 잘 보이도록 적절히 조절
            # block_size는 홀수가 되도록 설정함
            bin = 10
            # bin = 5
            if (src.shape[1] // bin) % 2 == 0:
                block_size = src.shape[1] // bin + 1
            else:
                block_size = src.shape[1] // bin
            # src_binary = cv2.adaptiveThreshold(src, max_val, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size,
            #                                    C)
            src_binary = cv2.adaptiveThreshold(src, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size,
                                               C)
            # x.resize_check(src_binary)
            # cv2.imwrite('./src_binary.jpg', src_binary) # 이미지 저장


            ## morphology operation
            # MORPH_DILATE
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            src_dilate = cv2.morphologyEx(src_binary, cv2.MORPH_DILATE, kernel)

            # MORPH_ERODE
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
            src_erode = cv2.morphologyEx(src_binary, cv2.MORPH_ERODE, kernel)

            src_gradient = src_dilate - src_erode
            # cv2.imwrite('./src_gradient.jpg', src_gradient) # 이미지 저장


            # MORPH_GRADIENT : dilate - erode. Get outline.
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # src_gradient = cv2.morphologyEx(src_binary, cv2.MORPH_GRADIENT, kernel)

            # MORPH_CLOSE : dilate -> erode. Lump pixels.
            # c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
            src_close = cv2.morphologyEx(src_gradient, cv2.MORPH_CLOSE, c_kernel)
            # cv2.imwrite('./src_close.jpg', src_close) # 이미지 저장


            # image check
            src_close = x.resize_check(src_close)


            ## contouring
            contours, hierarchy = cv2.findContours(src_close, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)
            # draw contour
            src = x.resize_check(x.printRGB())
            src_contour = cv2.drawContours(src, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)

            x.resize_check(src_contour)
            # cv2.imwrite('./src_contour.jpg', src_contour) # 이미지 저장


            # 데이터 영역 ROI와 함께 확인
            x1, y1, x2, y2 = r.read_roi(src_name)
            width = x2 - x1
            height = y2 - y1
            x1_ = int(x1 - 0.4 * width)
            x2_ = int(x2 + 0.4 * width)
            y1_ = int(y1 - 0.7 * height)
            y2_ = int(y2 + 3 * height)
            rect = cv2.rectangle(src_contour, (x1_, y1_), (x2_, y2_), (255, 0, 0), thickness=2)
            x.resize_check(rect)
            # cv2.imwrite('./rect.jpg', rect) # 이미지 저장


    # OCR 전처리 수행
    # return 딕셔너리 - key:이미지 이름, value:데이터 영역 정보 (x, y, w, h)
    # 각 데이터영역에 대한 ROI 이미지 파일 저장할 경우, 1번만 수행할 것
    # 원본 이미지 사이즈에 대해 수행함. 파라미터 값, ROI 좌표를 동적으로 변환함
    def preprocess(self, n, m):

        # Get Img
        dir2file = Dir2File(file_path=self.file_path)
        file_path = dir2file.file_path
        src_names = dir2file.filename()

        src_roi = dict()    # 좌표를 저장할 딕셔너리 생성

        for i, src_name in enumerate(src_names[n:m]):
            src_file = file_path + '/' + src_name + '.jpg'
            print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인

            # 텍스르 영역 이미지를 저장할 경로 생성
            roi_path = './data/text_roi/' + src_name
            rp = CreatePath()
            rp.create_path(roi_path)


            ## GrayScaling
            get = GetImg(src_name)
            src = get.printGray()     # 원본 이미지 가져오기

            ## Binary
            # parameter
            # max_val = src.max()
            max_val = 255
            C = 10      # G-Type 영역 글자가 잘 보이도록 적절히 조절
            bin = 10    # block_size는 홀수가 되도록 설정함

            if (src.shape[1] // bin) % 2 == 0:
                block_size = src.shape[1] // bin + 1
            else:
                block_size = src.shape[1] // bin

            src_binary = cv2.adaptiveThreshold(src, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                               block_size, C)
            # get.img_check(src_binary)




            ## morphology operation
            # 원본이미지에서 적절한 커널사이즈가 설정되도록 변경
            src_width, _ = src_binary.shape

            # # MORPH_DILATE - ellipse
            # # k_box_3 = int(3 * src_width / 700)
            # k_box_x = int(7 * src_width / 700)
            # k_box_y = int(2 * src_width / 700)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_box_x, k_box_y))
            # src_dilate = cv2.morphologyEx(src_binary, cv2.MORPH_DILATE, kernel)
            # # get.img_check(src_dilate)

            # MORPH_ERODE - ellipse
            k_box_x = int(4 * src_width / 700)
            k_box_y = int(1 * src_width / 700)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_box_x, k_box_y))
            src_erode = cv2.morphologyEx(src_binary, cv2.MORPH_ERODE, kernel)

            # src_gradient = src_dilate - src_erode
            # get.img_check(src_gradient)

            # # MORPH_CLOSE
            # k_box_x = int(7 * src_width / 700)
            # k_box_y = int(1 * src_width / 700)
            # c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box_x, k_box_y))
            # src_close = cv2.morphologyEx(src_gradient, cv2.MORPH_CLOSE, c_kernel)
            # get.img_check(src_close)

            # MORPH_DILATE - rect
            # k_box_3 = int(3 * src_width / 700)
            k_box_x = int(1 * src_width / 700)
            k_box_y = int(2 * src_width / 700)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box_x, k_box_y))
            src_dilate_2 = cv2.morphologyEx(src_erode, cv2.MORPH_DILATE, kernel)
            # get.img_check(src_dilate_2)

            # MORPH_ERODE - rect
            k_box_x = int(5 * src_width / 700)
            k_box_y = int(2 * src_width / 700)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box_x, k_box_y))
            src_erode_2 = cv2.morphologyEx(src_dilate_2, cv2.MORPH_ERODE, kernel)
            # get.img_check(src_erode_2)

            src_close = src_erode_2

            # get.img_check(src)



            ## contouring
            contours, hierarchy = cv2.findContours(src_close, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)
            # print(len(contours)) # 이미지 당 200개 이상 있음
            # print("원래 외곽선:", contours[:2])

            # contours 중 면적이 넓은 외곽선 70개 그려서 확인
            # cv2.contourArea : contours가 감싼 면적
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:70]
            # contours = sorted(contours, key=cv2.contourArea, reverse=False)[30:]

            # 왜 sorted를 실행하면 아직 선언하지도 않은 x namespace 에러가 나지 ?
            # --> cv2.contourArea가 비동기 처리 방식일 수 있음
            # 해결방법 1: 메소드 실행 흐름을 끊어서 contours를 저장하고 불러내서 사용
            # 해결방법 2: cv2.contourArea 를 사용하지 않고 직접 연산하는 방식 이용

            #print(contours)
            # print("정렬한 외곽선:", contours[:2])

            # # contour 확인
            # src = get.printRGB()
            # src_contour = cv2.drawContours(src, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
            # get.img_check(src_contour)


            # 추려낸 contours 대상으로 사각형 그려서 text 영역별로 crop 수행
            idx = 0
            data_roi_coo = set()   # 좌표 저장할 set. epsilon 범위별로 잡히는 좌표가 중복되는 경우가 있음. 중복 저장을 피하기 위해 set에 저장.
            for con in contours:
                idx += 1
                # print("-----------\n", idx, con)

                perimeter = cv2.arcLength(con, True)    # 외곽선 둘레 길이를 반환


                # x, y, w, h의 namespace 설정
                x = 0
                y = 0
                w = 0
                h = 0

                for epsilon in range(0, 200):

                    epsilon = epsilon / 1000
                    # 외곽선 근사화하여 좌표 반환
                    approx = cv2.approxPolyDP(con, epsilon * perimeter, True)

                    # 다각형 그려서 확인
                    # poly = cv2.polylines(src, [approx], True, (0, 0, 255), thickness=1)
                    # cv2.imshow('poly', poly)
                    # cv2.waitKey(0)


                    # 4개의 코너를 가지는 Edge Contour에 대해 사각형 추출
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(con)      # 좌표를 감싸는 최소면적 사각형 정보 반환
                        # text_roi = src_binary[y:y+h, x:x+w]     # crop

                        src_w = src.shape[1]
                        src_h = src.shape[0]
                        # 빠른 객체탐지를 위해 작은 값 무시
                        if (w < 0.2 * src_w) or (w > 0.6 * src_w) or (h > 0.2 * src_h) or (w/h > 4) or (w/h < 1.7):  # 노이즈 무시
                            # or (w/h > 3) or (w/h < 2)
                            break

                        # 사각형 그리기
                        # 이미지 출력을 수행하면 메모리상에 있던 데이터가 반환돼서 이미지 저장할 데이터가 사라짐
                        # 저장 시에는 반드시 주석처리 할 것
                        rect = cv2.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
                        cv2.imshow('roi', rect)
                        key = cv2.waitKey()
                        if key == 27:
                            break

                        # 각 텍스트 영역 이미지 저장
                        # 이미지 당 10초 내외로 걸림
                        # cv2.imwrite(roi_path + '/' + str(idx) + '.jpg', text_roi)

                    # 면적이 0인 좌표는 저장하지 않음
                    if w * h != 0:
                        data_roi_coo.add((x, y, w, h))   # ROI 좌표를 튜플로 저장함
                # print("data_roi_coo:", data_roi_coo)

            src_roi[src_name] = data_roi_coo
        return src_roi



    # 8-segment 영역(액정) 탐지
    # return 딕셔너리 - key: 이미지 이름, value: 8-segment영역 정보 튜플 (x, y, w, h)
    def preprocess_8seg(self, n, m):

        # Get Img
        dir2file = Dir2File(file_path=self.file_path)
        file_path = dir2file.file_path
        src_names = dir2file.filename()

        src_roi = dict()    # 좌표를 저장할 딕셔너리 생성
        for i, src_name in enumerate(src_names[n:m]):
            src_file = file_path + '/' + src_name + '.jpg'
            print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인


            # 좌표를 저장할 경로 생성
            roi_path = './data/text_roi/' + src_name
            rp = CreatePath()
            rp.create_path(roi_path)


            ## 이미지 읽어오기
            get = GetImg(src_name)
            src = get.printGray()

            # src = get.printLAB()  # LAB 컬러채널
            # src = src[:,:,0]      # l, a, b에서 컬러채널 l 분리

            # median = cv2.medianBlur(src, 5)    # 필터 적용
            # src = 255 - median  # 필터 반전
            # get.img_check(src)


            ## Histogram equalization
            src = cv2.equalizeHist(src)
            # get.img_check(src)


            ## Binary
            # parameter
            max_val = 255
            C = -10      # 8-seg 영역이 잘 보이도록 전처리
            bin = 10    # block_size는 홀수가 되도록 설정함

            if (src.shape[1] // bin) % 2 == 0:
                block_size = src.shape[1] // bin + 1
            else:
                block_size = src.shape[1] // bin


            # 외곽선 검출의 경우 검은 바탕에서 흰 영역 찾기가 더 수월하여,
            # 액정영역이 흰영역으로 표시되도록 흑백 반전된 cv2.THRESH_BINARY_INV를 이용함
            src = cv2.adaptiveThreshold(src, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                               block_size, C)
            # get.img_check(src)




            # morphology operation
            # 원본이미지에서 적절한 커널사이즈가 설정되도록 변경
            src_width, _ = src.shape

            # # MORPH_DILATE - rect
            # k_box_x = int(1 * src_width / 700)
            # k_box_y = int(2 * src_width / 700)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box_x, k_box_y))
            # src = cv2.morphologyEx(src, cv2.MORPH_DILATE, kernel)
            # # get.img_check(src)
            #
            # # MORPH_ERODE - ellipse
            # k_box_x = int(5 * src_width / 700)
            # k_box_y = int(2 * src_width / 700)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_box_x, k_box_y))
            # src = cv2.morphologyEx(src, cv2.MORPH_ERODE, kernel)
            # # get.img_check(src)




            ## contouring
            # contours, hierarchy = cv2.findContours(src_close, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)
            contours, hierarchy = cv2.findContours(src, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)


            # contour 확인
            src_check = get.printRGB()
            # src_contour = cv2.drawContours(src_check, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
            # get.img_check(src_contour)


            # 추려낸 contours 대상으로 사각형 그려서 text 영역별로 crop 수행
            data_roi_coo = set()   # 좌표 저장할 set. epsilon 범위별로 잡히는 좌표가 중복되는 경우가 있음. 중복 저장을 피하기 위해 set에 저장.
            for con in contours:

                perimeter = cv2.arcLength(con, True)    # 외곽선 둘레 길이를 반환

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

                    # 다각형 그려서 확인
                    # poly = cv2.polylines(src, [approx], True, (0, 0, 255), thickness=1)
                    # cv2.imshow('poly', poly)
                    # key = cv2.waitKey()
                    # if key == 27:
                    #     break


                    # 4개의 코너를 가지는 Edge Contour에 대해 사각형 추출
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(con)        # 좌표를 감싸는 최소면적 사각형 정보 반환
                        # text_roi = src_binary[y:y+h, x:x+w]     # crop

                        src_w = src.shape[1]
                        src_h = src.shape[0]

                        if (w < 0.2 * src_w) or (w > 0.6 * src_w) or (h > 0.2 * src_h) or (w / h > 4) or (
                                w / h < 1.7):  # 노이즈 무시
                            break

                        # 사각형 그리기
                        # 이미지 출력을 수행하면 메모리상에 있던 데이터가 반환돼서 이미지 저장할 데이터가 사라짐
                        # 저장 시에는 반드시 주석처리 할 것
                        rect = cv2.rectangle(src_check, (x, y), (x+w, y+h), (255, 0, 255), thickness=2)
                        cv2.imshow('roi', rect)
                        key = cv2.waitKey()
                        if key == 27:
                            break

                    # 면적이 0인 좌표는 저장하지 않음
                    if w * h != 0:
                        data_roi_coo.add((x, y, w, h))   # ROI 좌표를 튜플로 저장함

            src_roi[src_name] = data_roi_coo

        return src_roi





    # 액정영역 찾기
    def find_8seg(self, n, m):

        # Get Img
        dir2file = Dir2File(file_path=self.file_path)
        file_path = dir2file.file_path
        src_names = dir2file.filename()


        # 이미지 전처리
        roi_coo = self.preprocess(n, m)  # 딕셔너리, 키:src_name, 값:데이터영역 (x, y, w, h)

        for i, src_name in enumerate(src_names[n:m]):
            src_file = file_path + '/' + src_name + '.jpg'
            print("-----------------------------------------")
            print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인

            print(f"{src_name}에서 추출한 영역 개수:", len(roi_coo[src_name]))
            print("추출 영역 데이터 확인(x, y, w, h):", roi_coo[src_name])

            # 이미지 가져오기
            get = GetImg(src_name)
            src = get.printGray()

            # 각 추출 영역에 대해 OCR 수행
            for roi in roi_coo[src_name]:
                x, y, w, h = roi

                # 사각형 그리기
                # 이미지 출력을 수행하면 메모리상에 있던 데이터가 반환돼서 이미지 저장할 데이터가 사라짐
                # 저장 시에는 반드시 주석처리 할 것
                rect = cv2.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
                print(len(rect))

                cv2.imshow('roi', rect)
                key = cv2.waitKey()
                if key == 27:
                    break





    def run_OCR(self, n, m):
        pass

        # # 전처리 속도보다 전처리 완료된 이미지를 불러오는 게 더 빠름
        # # 만약 src_name에 해당하는 text_roi 폴더가 있으면 파일 불러오고
        # # 없으면 해당 이미지 1장에 대해 preprocess 수행되도록 함
        # # text_roi = self.preprocess(n, m)
        #
        # # Get Img
        # dir2file = Dir2File(file_path=self.file_path)
        # file_path = dir2file.file_path
        # src_names = dir2file.filename()
        #
        # # pytesseract PATH 설정
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        #
        # # 이미지 전처리
        # roi_data = self.preprocess(n, m)  # 딕셔너리, 키:src_name, 값:데이터영역 (x, y, w, h)
        #
        # for i, src_name in enumerate(src_names[n:m]):
        #     src_file = file_path + '/' + src_name + '.jpg'
        #     print("-----------------------------------------")
        #     print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인
        #
        #     print(f"{src_name}에서 추출한 영역 개수:", len(roi_data[src_name]))
        #     print("추출 영역 데이터 확인(x, y, w, h):", roi_data[src_name])
        #
        #     # 이미지 가져오기
        #     get = GetImg(src_name)
        #     src = get.printGray()
        #
        #     # 각 추출 영역에 대해 OCR 수행
        #     for roi in roi_data[src_name]:
        #         x, y, w, h = roi
        #
        #         # 원본 이미지에 대해 데이터 ROI 영역 슬라이싱
        #         src_roi = src[y:y+h, x:x+w]
        #
        #         # 해당 영역에서 pytesseract 수행
        #         # config parameters
        #         # -l : 사용할 언어 설정
        #         # --oem 1 : LSTM OCR 엔진 사용 설정
        #         config = ('-l kor+eng --oem 1 --psm 3')
        #         text = pytesseract.image_to_string(src_roi, config=config)
        #         # text = pytesseract.image_to_string(src_roi, lang='kor')
        #
        #         # 확인
        #         # if len(text) != 1:
        #         #     print(text, len(text), type(text))
        #         print(text)











# Perspective transform
class Perspec_T:
    pass
    # Polygon
    # transform to square
