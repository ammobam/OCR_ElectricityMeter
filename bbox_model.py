# Region Proposal
# selective search
# hierarchical grouping algorithm 이용

import cv2
import numpy as np
from matplotlib import pyplot as plt
import prep_img

src_file = './img_src_0906/img_000001.jpg'
src_file = './img_src/img_prac.jpg'

# 1. 이미지 컬러로 읽어오기
# 1_1. RGB
src = cv2.imread(src_file, cv2.IMREAD_COLOR)
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
src.shape  # (4032, 3024, 3)
#r, g, b = cv2.split(src)
plt.imshow(src);plt.axis('off');plt.show()
#plt.imshow(r);plt.axis('off');plt.show()
#plt.imshow(g);plt.axis('off');plt.show()
#plt.imshow(b);plt.axis('off');plt.show()


# 액정 부분의 평균 값 알아보기
x_center = src.shape[0]//2
y_center = src.shape[1]//2
src_mean = src[x_center-100:x_center+100, y_center-200:y_center+200].mean() # 72.42
src[x_center-100:x_center+100, y_center-200:y_center+200] = src_mean
plt.matshow(src)

# 액정 부분의 중앙값 알아보기
src_med = np.median(src[x_center-100:x_center+100, y_center-100:y_center+100]) # 72.42
src[x_center-100:x_center+100, y_center-100:y_center+100] = (src_med, src_med, 0)
plt.matshow(src)


####################################


# 1_2. HSV
# 좀더 세밀한 컬러영역 탐지를 위해 hsv 이용
src = cv2.imread(src_file, cv2.IMREAD_COLOR)
src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
#src.shape  # (4032, 3024, 3)

h, src_s, v = cv2.split(src)
plt.imshow(h);plt.axis('off');plt.show()
plt.imshow(src_s);plt.axis('off');plt.show()    # S로 읽을 때 액정 영역 탐지 잘 함
#plt.imshow(v);plt.axis('off');plt.show()


## 전력량 영역 검출
# src를 hsv로 읽고 s 이미지에서 객체 검출

# 이미지에서 중앙영역 가져오기
x_center_left = int(src_s.shape[0]*2/5)
x_center_right = int(src_s.shape[0]*3/5)
y_center_left = int(src_s.shape[1]*2/5)
y_center_right = int(src_s.shape[1]*3/5)

center = src_s[x_center_left:x_center_right, y_center_left:y_center_right]

# center 영역의 기술통계량 확인
center_mean = int(center.mean())  # 12
center_med = int(np.median(center))  # 11
center_min = int(center.min())  # 0
center_max = int(center.max())  # 44
#center_mode, _ = mode(center, axis=None) # 24
#int(center_mode)

# 빈도수
'''
ar = np.array([[0,0,0,0,0,1], [0,0,1,0,2,0]])
np.unique(ar, return_index=True, return_counts=True)
# n차원 배열이라도 1차원으로 폈을 때의 인덱스로 반환함 
# 3개의 배열 리턴함
# 1 : 값 리턴. 순서는 값이 작은 순.
# 2 : 각 값의 인덱스 리턴 (return_index=True)
# 3 : 각 값의 빈도수 리턴 (return_counts=True)
# 각 값의 빈도수 순으로 정렬해서 써야겠다
'''
center_mode_val, center_mode_counts = np.unique(center, return_counts=True)
center_mode_val, center_mode_counts
# center_mode_counts 순으로 정렬해서 center_mode_val 받아서 써야겠다
mode_dict = {}
for val, count in zip(center_mode_val, center_mode_counts):
    # 빈도수를 key, 그때의 색상값을 val로 하는 딕셔너리
    mode_dict[count] = val
# 딕셔너리를 내림차순으로 정렬해서 등장빈도 n번째 값 가져오기
mode_key = sorted(mode_dict, reverse=True)[0]
mode_dict[mode_key]



# (1) 최소값에서 오차범위 적당히 주기
# (2) 최소값의 주변 영역의 평균값 가져오기
# (3) 최소, 최대에 따라 평활화 하고, 70-90 사이 값 확인하기

center_smooth = (center-center_min)/(center_max-center_min)*255
int(center_smooth.max())
plt.imshow(center_smooth);plt.axis('off');plt.show()

center_smooth_mean = int(center_smooth.mean())  # 72
center_smooth_med = int(np.median(center_smooth))  # 63
center_smooth_min = int(center_smooth.min())  # 0
center_smooth_max = int(center_smooth.max())  # 255
center_smooth_mode_val, center_smooth_mode_counts = np.unique(center_smooth, return_counts=True)
# center_mode_counts 순으로 정렬해서 center_mode_val 받아서 써야겠다
mode_dict = {}
for val, count in zip(center_smooth_mode_val, center_smooth_mode_counts):
    # 빈도수를 key, 그때의 색상값을 val로 하는 딕셔너리
    mode_dict[count] = val
# 딕셔너리를 내림차순으로 정렬해서 등장빈도 n번째 값 가져오기
mode_key = sorted(mode_dict, reverse=True)[1]
mode_dict[mode_key]



# 확인
# 중앙영역 보정한 이미지의 중앙값으로 대체
src_s[x_center_left:x_center_right, y_center_left:y_center_right]=center_smooth_med
plt.imshow(src_s);plt.axis('off');plt.show()
plt.imshow(center_smooth);plt.axis('off');plt.show()

# 중앙영역 보정한 이미지의 평균값으로 대체
src_s[x_center_left:x_center_right, y_center_left:y_center_right]=center_smooth_mean
plt.imshow(src_s);plt.axis('off');plt.show()

# 중앙영역 보정한 이미지의 빈도값으로 대체
src_s[x_center_left:x_center_right, y_center_left:y_center_right]=mode_dict[mode_key]
plt.imshow(src_s);plt.axis('off');plt.show()




src_smooth = (src_s-center_min)/(center_max-center_min)*255
src_smooth[x_center_left:x_center_right, y_center_left:y_center_right]=120
plt.imshow(src_smooth);plt.axis('off');plt.show()




####################
# 영상 히스토그램 구하기
# https://gaussian37.github.io/vision-opencv-histogram/
# calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist

src_name = 'img_prac'
#src_file = './img_src/' + src_name + '.jpg'
src_file = 'img_src_0906/img_000001.jpg'
src = cv2.imread(src_file, cv2.IMREAD_COLOR)
src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
src.shape
h = src[:,:,0]
s = src[:,:,1]
v = src[:,:,2]

# 총 256개의 픽셀 값에서 histSize 만큼 나눠 주게 되면 hist 구간의 개수가 됨
histSize = 32
nbins = 256 // histSize
# x축을 nbins 만큼의 구간으로 만들되 각 tick를 [0, 255] 까지 범위 안에서 만듦
binX = np.arange(histSize) * nbins
histColor = ['b', 'g', 'r']
labels = ['H', 'S', 'V']
for i in range(3):
    # (-215:Assertion failed) rsz == dims*2 || (rsz == 0 && images.depth(0) == CV_8U) in function 'cv::calcHist'
    # 파라미터는 모두 리스트 형태로 준다
    hist = cv2.calcHist(images=[src], channels=[i], mask=None, histSize=[histSize], ranges=[0, 256])
    plt.plot(binX, hist, color=histColor[i], label=labels[i]);plt.legend();plt.show()


####################
# 액정 영역의 mean 값 +- 10 정도 차이 나는 영역 검출하기
img = np.where((s > center_smooth_mean-10) & (src_s < center_smooth_mean+10), 255, src_s)
plt.imshow(img);plt.axis('off');plt.show()










# 액정 영역과 계량기 영역의 비율 구해서 적용하기

# 2. 관심영역이 배경과 구분되고, 균일한 컬러를 갖도록 전처리

# 3. hierarchical grouping algorithm 이용하여 selective search 수행

# 4. ROI 추출 결과 확인

# 5. 모델 개선을 위한 NMS 수행
