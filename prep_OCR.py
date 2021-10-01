'''
# 예시코드

# OCR을 위한 이미지 전처리 수행 및 데이터 ROI 파일 저장
from prep_OCR import OCR_prep
x = OCR_prep('./data/ElectricityMeter', './data/roi')
x.preprocess(0,10)





'''

# OCR 수행을 위한 전처리

import cv2
import os
from prep_img import GetImg
from select_ROI import Dir2File, ROI2Img, CreatePath
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
    # 1번만 수행할 것
    # 각 데이터영역에 대한 ROI 이미지 파일 저장함.
    # 원본 이미지 사이즈에 대해 수행함. 파라미터 값, ROI 좌표를 동적으로 변환함
    def preprocess(self, n, m):

        # Get Img
        dir2file = Dir2File(file_path=self.file_path)
        file_path = dir2file.file_path
        src_names = dir2file.filename()

        for i, src_name in enumerate(src_names[n:m]):
            src_file = file_path + '/' + src_name + '.jpg'
            print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인

            # 텍스르 영역 이미지를 저장할 경로 생성
            roi_path = './data/text_roi/' + src_name
            rp = CreatePath()
            rp.create_path(roi_path)


            ## GrayScaling
            x = GetImg(src_name)
            # image check
            # src = x.resize_check(x.printGray())   # 여기서 모든 이미지의 가로픽셀이 700으로 줄어들음. 이미지를 줄이지 말고 좌표에 원래 비율을 적용하자.
            src = x.printGray()     # 원본 이미지 가져오기

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
            # x.img_check(src_binary)




            ## morphology operation
            # 원본이미지에서 적절한 커널사이즈가 설정되도록 변경
            src_width, _ = src_binary.shape

            # # MORPH_DILATE - ellipse
            # # k_box_3 = int(3 * src_width / 700)
            # k_box_x = int(7 * src_width / 700)
            # k_box_y = int(2 * src_width / 700)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_box_x, k_box_y))
            # src_dilate = cv2.morphologyEx(src_binary, cv2.MORPH_DILATE, kernel)
            # # x.img_check(src_dilate)

            # MORPH_ERODE - ellipse
            k_box_x = int(4 * src_width / 700)
            k_box_y = int(1 * src_width / 700)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_box_x, k_box_y))
            src_erode = cv2.morphologyEx(src_binary, cv2.MORPH_ERODE, kernel)

            # src_gradient = src_dilate - src_erode
            # x.img_check(src_gradient)

            # # MORPH_CLOSE
            # k_box_x = int(7 * src_width / 700)
            # k_box_y = int(1 * src_width / 700)
            # c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box_x, k_box_y))
            # src_close = cv2.morphologyEx(src_gradient, cv2.MORPH_CLOSE, c_kernel)
            # x.img_check(src_close)

            # MORPH_DILATE - rect
            # k_box_3 = int(3 * src_width / 700)
            k_box_x = int(1 * src_width / 700)
            k_box_y = int(2 * src_width / 700)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box_x, k_box_y))
            src_dilate_2 = cv2.morphologyEx(src_erode, cv2.MORPH_DILATE, kernel)
            # x.img_check(src_dilate_2)

            # MORPH_ERODE - rect
            k_box_x = int(5 * src_width / 700)
            k_box_y = int(2 * src_width / 700)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_box_x, k_box_y))
            src_erode_2 = cv2.morphologyEx(src_dilate_2, cv2.MORPH_ERODE, kernel)
            # x.img_check(src_erode_2)

            src_close = src_erode_2



            ## contouring
            contours, hierarchy = cv2.findContours(src_close, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)
            # print(len(contours)) # 이미지 당 200개 이상 있음

            # contours 중 면적이 넓은 외곽선 70개 그려서 확인
            # 가장 면적이 넓은 외곽선은 대부분 배경에 그려지므로 제외함
            # cv2.contourArea : contours가 감싼 면적
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:70]
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)[2:100]
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:]

            # contour 확인
            # src = x.printRGB()
            # src_contour = cv2.drawContours(src, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
            # x.img_check(src_contour)

            # 추려낸 contours 대상으로 사각형 그려서 text 영역별로 crop 수행
            idx = 0
            for con in contours:
                idx += 1
                # print("-----------\n", idx, con)

                perimeter = cv2.arcLength(con, True)    # 외곽선 둘레 길이를 반환

                for epsilon in range(0, 200):
                    epsilon = epsilon/1000              # epsilon 을 0.001 단위로 늘려가며 적용해야 "G-Type"부분이 잡힘
                    approx = cv2.approxPolyDP(con, epsilon * perimeter, True) # 외곽선 근사화하여 좌표 반환

                    # # 다각형 그려서 확인
                    # poly = cv2.polylines(src, [approx], True, (0, 0, 255), thickness=3)
                    # cv2.imshow('poly', poly)
                    # key = cv2.waitKey()
                    # if key == 27:  # esc 키
                    #     break

                    # 4개의 코너를 가지는 Edge Contour에 대해 사각형 추출
                    if len(approx) == 4:

                        x, y, w, h = cv2.boundingRect(con)      # 좌표를 감싸는 최소면적 사각형 정보 반환
                        text_roi = src_binary[y:y+h, x:x+w]     # crop

                        # 사각형 그리기
                        # 이미지 출력을 수행하면 메모리상에 있던 데이터가 반환돼서 이미지 저장할 데이터가 사라짐
                        # 저장 시에는 반드시 주석처리 할 것
                        # rect = cv2.rectangle(src, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
                        # cv2.imshow('roi', rect)
                        # key = cv2.waitKey()
                        # if key == 27:  # esc 키
                        #     break

                        # 각 텍스트 영역 이미지 저장
                        # 이미지 당 약 10여초 걸림. 객체가 매우 많은 경우는 20초.
                        cv2.imwrite(roi_path + '/' + str(idx) + '.jpg', text_roi)



    def run_OCR(self, n, m):
        pass

        # 전처리 속도보다 전처리 완료된 이미지를 불러오는 게 더 빠르겠지
        # 만약 src_name에 해당하는 text_roi 폴더가 있으면 파일 불러오고
        # 없으면 preprocess 수행되도록 해야겠다
        # text_roi = self.preprocess(n, m)

        # # pytesseract PATH 설정
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        #
        # for ..........
        #
        #     # 해당 영역에서 pytesseract 수행
        #     # config parameters
        #     # -l : 사용할 언어 설정
        #     # --oem 1 : LSTM OCR 엔진 사용 설정
        #     config = ('-l kor+eng --oem 1 --psm 3')
        #     text = pytesseract.image_to_string(src_roi, config=config)
        #     # text = pytesseract.image_to_string(src_roi, lang='kor')
        #
        #     # 확인
        #     print(text)








# Perspective transform
class Perspec_T:
    pass
    # Polygon
    # transform to square
