from prep_img import GetImg, GetHist, AreaColor, FindCenter

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 이미지 가져오기
src_name = '01232004639_P1138'
src_name = '01232007156_P160'
x = GetImg(src_name)
# src = x.printYCrCb()
# src = x.printRGB()
src = x.printHSV()


# 이미지 채널 분리
y, cr, cb = cv2.split(src)
y.shape
cr.shape
cb.shape

src.shape
src[0].shape
src.shape
src[:,:,0].shape



# 채널 Y에 대해서만 히스토그램 평활화 수행
y_eq = cv2.equalizeHist(y)
# 컬러 채널 합치기
src_eq = cv2.merge([y_eq, cr, cb])
cv2.imshow("src_eq", src_eq)
# 히스토그램 확인
g = GetHist()
g.get_histogram(src)    # 원본
g.get_histogram(src_eq) # 컬러채널 H 평활화



# 전체 이미지의 색상값 조사
c = AreaColor(src_eq)
print(c.mean(), c.med(), c.min(), c.max())
# 중심 영역의 색상값 조사
center = FindCenter(src).find_center()
c = AreaColor(center)
print(c.mean(), c.med(), c.min(), c.max())



# 색상 범위 지정하여 영역 분할
# inRange() lowerb, upperb에 이미지와 동일 사이즈의 행렬을 적용하면 각 픽셀마다 다른 값 적용 가능함
# lowerb < src < upperb
# lowerb, upperb 는 각 색상채널의 튜플로
lower_hue = 90
upper_hue = 120
lowerb = (lower_hue, 0, 50)
upperb = (upper_hue, 255, 200)

mask = cv2.inRange(src, lowerb, upperb)
cv2.imshow("sep_color", mask);cv2.imshow("src", src)

test = np.zeros(200*200*3).reshape(200, 200, 3)
test.shape
test[0,:,:] = 200
test[:,1,:] = 100
test[:,:,2] = 5
test.shape
test
cv2.imshow("test", test)
