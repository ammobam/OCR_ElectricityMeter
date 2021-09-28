# OCR 수행을 위한 전처리

import cv2
from prep_img import GetImg
from select_ROI import Dir2File, ROI2Img


class OCR_prep:

    def __init__(self, file_path):
        self.file_path = file_path


    def preprocess(self, n, m):
        # Get Img
        dir2file = Dir2File(file_path=self.file_path)
        file_path = dir2file.file_path
        src_names = dir2file.filename()

        for i, src_name in enumerate(src_names[n:m]):
            src_file = file_path + '/' + src_name + '.jpg'
            print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인


            # GrayScaling
            x = GetImg(src_name)
            src = x.printGray()
            # src = x.printRGB()
            # src = x.printYCrCb()
            # src = x.printHSV()

            # image check
            x.resize_check(src)


            # Binary
            # parameter
            max_val = src.max()
            C = 10
            # block_size는 짝수가 되도록 설정함
            if (src.shape[1] // 10) % 2 == 0:
                block_size = src.shape[1] // 10 + 1
            else:
                block_size = src.shape[1] // 10
            src_binary = cv2.adaptiveThreshold(src, max_val, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)


            # morphology operation
            # MORPH_GRADIENT : dilate - erode. Get outline.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            src_gradient = cv2.morphologyEx(src_binary, cv2.MORPH_GRADIENT, kernel)
            # MORPH_CLOSE : dilate -> erode. Lump pixels.
            c_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            src_close = cv2.morphologyEx(src_gradient, cv2.MORPH_CLOSE, c_kernel)


            # image check
            x.resize_check(src_close)


            # contouring in ROI


    # work in ROI, not over the entire area
    def work_in_ROI(self, roi_path, n, m):
        # Get ROI
        r = ROI2Img(roi_path)
        x = Dir2File(self.file_path)
        src_names = x.filename()

        for i, src_name in enumerate(src_names[n:m]):
            src_file = self.file_path + '/' + src_name + '.jpg'
            print(f"- {(i + 1) / (m - n) * 100:.1f}%.....{i + 1}번째_수행파일:{src_file}")  # 확인

            # get ROI coordinates
            x1, y1, x2, y2 = r.read_roi(src_name)

            # get image
            x = GetImg(src_name)
            src = x.printHSV()
            src = x.resize_check(src)
            print(src.shape)


# Perspective transform
class Perspec_T:
    pass
    # Polygon
    # transform to square