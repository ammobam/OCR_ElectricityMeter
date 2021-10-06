



# 8-segment 영역(액정) 탐지
# return 딕셔너리 - key: 이미지 이름, value: 8-segment영역 정보 튜플 (x, y, w, h)
import cv2

from prep_img import GetImg
from select_ROI import Dir2File, CreatePath



class prep:

    def __init__(self, file_path):
        self.file_path = file_path  # 파일 읽어올 경로 입력
        # self.roi_path = roi_path    # 전처리 파일 저장할 경로 입력

    # 안드로이드에서 전송한 이미지 파일 ROI 영역 탐지 함수
    def find_8seg(self):

        # 이미지 불러오기
        dir2file = Dir2File(file_path=self.file_path)
        file_path = dir2file.file_path
        src_name = dir2file.filename()

        src_roi = dict()    # 좌표를 저장할 딕셔너리 생성
        src_file = file_path + '/' + src_name + '.jpg'
        print(f"- 수행파일:{src_file}")  # 확인

        # 좌표를 저장할 경로 생성
        roi_path = './data/show/' + src_name
        rp = CreatePath()
        rp.create_path(roi_path)


        ## 이미지 읽어오기 - GrayScale
        get = GetImg(src_name)
        src = get.printGray()


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
        cv2.imwrite(roi_path + '/' + src_name + '_binary' + '.jpg', src)   # 저장


        ## contouring
        contours, hierarchy = cv2.findContours(src, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_TC89_L1)
        src_check = get.printRGB()
        src_contour = cv2.drawContours(src_check, contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
        cv2.imwrite(roi_path + '/' + src_name + '_contour' + '.jpg', src_contour)   # 저장
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
                    cv2.imwrite(roi_path + '/' + src_name + '_rect' + '.jpg', rect)  # 저장
                    # cv2.imshow('roi', rect)
                    # key = cv2.waitKey()
                    # if key == 27:
                    #     break

        src_roi[src_name] = data_roi_coo

        return src_roi