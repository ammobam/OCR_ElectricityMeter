'''
# 예시코드
# OCR을 위한 이미지 전처리 수행
from prep_OCR import OCR_prep
x = OCR_prep('./data/ElectricityMeter', './data/roi')
x.preprocess(0,10)
'''

# OCR 수행을 위한 전처리

import cv2
from prep_img import GetImg
from select_ROI import Dir2File, ROI2Img


class OCR_prep:

    def __init__(self, file_path, roi_path):
        self.file_path = file_path
        self.roi_path = roi_path

    # work in ROI, not over the entire area
    def work_in_ROI(self, roi_filename, src):
        # Get ROI
        r = ROI2Img(self.roi_path)
        # get ROI coordinates
        x1, y1, x2, y2 = r.read_roi(roi_filename)
        width = x2 - x1
        height = y2 - y1
        x1_ = int(x1 - 0.4 * width)
        x2_ = int(x2 + 0.4 * width)
        y1_ = int(y1 - 0.7 * height)
        y2_ = int(y2 + 3 * height)

        # cropping ROI
        return src[y1_:y2_, x1_:x2_]  # height, width


    def preprocess(self, n, m):
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
            # src_close = x.resize_check(src_close)


            ## contouring
            contours, hierarchy = cv2.findContours(src_close, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)
            # draw contour
            src = x.resize_check(x.printRGB())
            src_contour = cv2.drawContours(src, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)

            # x.resize_check(src_contour)
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






# Perspective transform
class Perspec_T:
    pass
    # Polygon
    # transform to square
