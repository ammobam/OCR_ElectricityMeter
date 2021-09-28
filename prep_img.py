import numpy as np
import cv2
import matplotlib.pyplot as plt

# 이미지 이름을 입력하면 배열로 가져오는 클래스
class GetImg:
    def __init__(self, src_name):
        self.src_file = './data/ElectricityMeter/' + src_name + '.jpg'

    def resize_check(self, src):
        # 이미지 resize 하기
        src_height, src_width = src.shape[:2]
        # 이미지 가로길이가 700이 되도록 설정함
        ratio = 700 / src_width
        src_height, src_width = int(src_height * ratio), int(src_width * ratio)
        src_resize = cv2.resize(src, (src_width, src_height))
        # 이미지 확인
        while True:
            cv2.imshow("src_check", src_resize)
            key = cv2.waitKey()
            if key == 27:  # esc 키
                break
        cv2.destroyAllWindows()
        return src_resize

    def printGray(self):
        src = cv2.imread(self.src_file, cv2.IMREAD_GRAYSCALE)
        print(src)
        return src

    def printRGB(self):
        src = cv2.imread(self.src_file, cv2.IMREAD_COLOR)
        # src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        return src

    def printYCrCb(self):
        src = cv2.imread(self.src_file, cv2.IMREAD_COLOR)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
        return src

    def printHSV(self):
        src = cv2.imread(self.src_file, cv2.IMREAD_COLOR)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        return src

# 이미지를 컬러채널로 나눠 히스토그램 그리는 클래스
class GetHist:
    # cv2.split(src) 과 동일함
    def sep_color_channel(self, src):
        h = src[:,:,0]
        s = src[:,:,1]
        v = src[:,:,2]
        return h, s, v

    def get_histogram(self, src):
        # 총 256개의 픽셀 값에서 histSize 만큼 나눠 주게 되면 hist 구간의 개수가 됨
        histSize = 32
        nbins = 256 // histSize

        # x축을 nbins 만큼의 구간으로 만들되 각 tick를 [0, 255] 까지 범위 안에서 만듦
        binX = np.arange(histSize) * nbins
        histColor = ['b', 'g', 'r']
        labels = ['H', 'S', 'V']
        for i in range(3):
            # 에러 : (-215:Assertion failed) rsz == dims*2 || (rsz == 0 && images.depth(0) == CV_8U) in function 'cv::calcHist'
            # 파라미터는 모두 리스트 형태로 준다
            hist = cv2.calcHist(images=[src], channels=[i], mask=None, histSize=[histSize], ranges=[0, 256])
            plt.plot(binX, hist, color=histColor[i], label=labels[i]);plt.legend();plt.show()



# 이미지에서 중심영역만 가져오는 클래스
class FindCenter:

    # 전역변수
    def __init__(self, src):
        self.src = src

    # 이미지에서 center 영역 가져오기
    def find_center(self)->np.array:
        x_center_left = int(self.src.shape[0]*2/5)
        x_center_right = int(self.src.shape[0]*3/5)
        y_center_left = int(self.src.shape[1]*2/5)
        y_center_right = int(self.src.shape[1]*3/5)

        center = self.src[x_center_left:x_center_right, y_center_left:y_center_right]
        return center

# 이미지 영역의 색상 기술통계량 확인 클래스
class AreaColor:
    def __init__(self, area:np.array):
        self.area = area

    # 평균
    def mean(self) -> int:
        area_mean = int(self.area.mean())
        return area_mean
    # 중앙값
    def med(self) -> int:
        area_med = int(np.median(self.area))
        return area_med
    # 최소값
    def min(self) -> int:
        area_min = int(self.area.min())
        return area_min
    # 최대값
    def max(self) -> int:
        area_max = int(self.area.max())
        return area_max

    # 특정 영역에서의 자주 등장한 색상값 가져오기
    def freq(self,nth:int)->int:
        mode_val, mode_counts = np.unique(self.area, return_counts=True)

        # 등장횟수(mode_counts) 기준으로 정렬해서 색상값(mode_val) 가져오기
        mode_dict = {}
        for val, count in zip(mode_val, mode_counts):
            # 빈도수를 key, 그때의 색상값을 val로 하는 딕셔너리
            mode_dict[count] = val
        # 딕셔너리를 내림차순으로 정렬해서 등장빈도 n번째 값 가져오기
        mode_key = sorted(mode_dict, reverse=True)[nth]
        return mode_dict[mode_key]



