# 이미지에서 계량기 ROI 추출하기
# 1. 계량기의 형태가 도드라지도록 이미지를 전처리하고
# 2. ROI 추출하고`
# 3. IoU 평가

# %% ------------------------------------------
# 패키지
import cv2
from matplotlib import pyplot as plt
import numpy as np

# %% ------------------------------------------
# 이미지 불러오기 - 흑백
# 파일이름이 한글이면 못 읽어옴. 영문으로 바꿀 것.
#src_name = 'img_practice'
# src_name = 'img_prac'
# src_file = './img_src/' + src_name + '.jpg'
src_file = './img_src/' +'01232007156_P160.jpg'
src = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
print(type(src))

# 전체 파일 불러오고 확인하기
for src_name in src_names:
    src_file = './img_src_0906/' + src_name + '.jpg'
    src = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
    print(src_name, ":", src.shape)



# 확인
# cv2.imshow()는 창을 동시에 여러 개 볼 수 있음
print(src.shape)
plt.imshow(src, cmap='gray');plt.axis('off');plt.show()
# cv2.imshow("src", src)



# %% ------------------------------------------
# 1.이미지 전처리
# 1-1.명암
dst = src * 2

# 확인
cv2.imshow("contrast", dst)
#plt.imshow(dst, cmap='gray')

# 저장
dst_process = src_name + '_contrast'
dst_file = './img_prep/' + dst_process + '.jpg'
cv2.imwrite(dst_file, dst)


# 1-2.binary(이진화)
# 적절한 임계값 고르기
print("이미지 배열", src)
print("최소값", src.min())
print("최대값", src.max())
print("평균값", src.mean())
print("중앙값:", np.median(src))

# binary 수행
# cv2.threshold(src, thresh, maxval, type[, dst]) -> retval, dst
# 반환하는 튜플에서 1번째 데이터가 이미지 변환 배열임


# 공통 파라미터 설정
maxval = src.max()


# thresh : median
thresh = np.median(src)
binary_type = cv2.THRESH_BINARY
dst = cv2.threshold(src, thresh, maxval, binary_type)

plt.imshow(dst[1], cmap='gray'); plt.show()
#cv2.imshow("thresh: median", dst[1])
#plt.matshow(dst[1])

# thresh : mean
thresh = src.mean()
binary_type = cv2.THRESH_BINARY
dst = cv2.threshold(src, thresh, maxval, binary_type)
plt.imshow(dst[1], cmap='gray'); plt.show()
#cv2.imshow("thresh: mean", dst[1])

# mean + THRESH_OTSU
thresh = np.mean(src)
binary_type = cv2.THRESH_OTSU
dst = cv2.threshold(src, thresh, maxval, binary_type)
plt.imshow(dst[1], cmap='gray'); plt.show()
#cv2.imshow("thresh: mean + THRESH_OTSU", dst[1])

# median + THRESH_OTSU(선택)
thresh = np.median(src)
binary_type = cv2.THRESH_OTSU
dst = cv2.threshold(src, thresh, maxval, binary_type)
plt.imshow(dst[1], cmap='gray'); plt.show()
#cv2.imshow("thresh: median + THRESH_OTSU", dst[1])

# TODO : 형태 ROI에서 적합한 처리 정하기 (이진화 -> 침식)
# 이미지 파일로 저장
# 작업내용으로 파일이름 정함
dst_process = src_name + '_binary_med_OTSU'
dst_file = './img_prep/' + dst_process + '.jpg'
# 저장
cv2.imwrite(dst_file, dst[1])


# 1-3.적응형 이진화
# 이건 형태 ROI보다는 데이터 ROI에 적절해보임

# 파라미터 설정
maxval = src.max()
adaptiveMethod = [cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C), #cv2.BORDER_ISOLATED]
thresholdType = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]
C = -10 #########
C = 10

# block_size는 짝수가 되도록 설정함
if (src.shape[1]//10) %2==0:
    block_size = src.shape[1]//10 + 1
else :
    block_size = src.shape[1] // 10

# adaptiveMethod : MEAN # 0
dst = cv2.adaptiveThreshold(src, maxval, adaptiveMethod[0], thresholdType[0], block_size, C)
#plt.imshow(dst, cmap='gray'); plt.show()
cv2.imshow("adaptiveMethod : MEAN", dst)


# adaptiveMethod : GAUSSIAN # 1
# cv2.BORDER_REPLICATE 와 동일한 값
dst = cv2.adaptiveThreshold(src, maxval, adaptiveMethod[1], thresholdType[0], block_size, C)
cv2.imshow("adaptiveMethod : GAUSSIAN", dst)

# 저장
dst_process = src_name + '_adaptiveMethod_GAUSSIAN'
dst_file = './img_prep/' + dst_process + '.jpg'
cv2.imwrite(dst_file, dst)




# adaptiveMethod : BORDER_ISOLATED
# Unknown/unsupported adaptive threshold method
# 16은 안 됨
## 이 사진에서는 thresh:mean + OTSU가 적절
## 실제 데이터 셋에서는...


# %% ------------------------------------------
# 2. 침식
# binary 수행한 파일 불러오자
# 나중에 전처리 선택하고 나서는 필요 없을 과정
src = cv2.imread(dst_file, cv2.IMREAD_GRAYSCALE)
plt.imshow(src, cmap='gray')
#cv2.imshow("src", src)

# 구조요소 행렬 생성
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))


# 침식 (선택)
dst = cv2.erode(src, kernel)
plt.imshow(dst, cmap='gray')
#cv2.imshow("dst", dst)

# 저장
dst_process = src_name + '_adaptiveMethod_GAUSSIAN_erode'
dst_file = './img_prep/' + dst_process + '.jpg'
cv2.imwrite(dst_file, dst)


# 팽창
dst = cv2.dilate(src, kernel)
plt.imshow(dst, cmap='gray')
#cv2.imshow("dst", dst)

# 저장
dst_process = src_name + '_adaptiveMethod_GAUSSIAN_dilate'
dst_file = './img_prep/' + dst_process + '.jpg'
cv2.imwrite(dst_file, dst)




# 3. 레이블링

_, src_label = cv2.connectedComponents(src)
retval, labels, stats, centroids = cv2.connectedComponentsWithStats(src)
labels
stats




centroids

# 확인
np.unique(src)
plt.matshow(src)
plt.matshow(src_label)
plt.matshow(labels)




# %% ------------------------------------------
# 3. 엣지 검출


