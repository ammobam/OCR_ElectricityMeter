from prep_OCR import OCR_prep


# 전처리 수행
# 딕셔너리 반환 - key:이미지 이름, value:데이터 영역 정보 (x, y, w, h)
x = OCR_prep('./data/ElectricityMeter', './data/roi')
src_roi = x.preprocess_8seg(0, 3)

# 전체 이미지 중 8-segment 영역을 탐지한 경우의 수
# 정말 잘 탐지했는지 하나하나 확인해보자 - 이미지당  1개 또는 2개 영역을 추출함. ROI는 반드시 포함함.
# print("8segment ROI를 탐지한 이미지 개수:", sum([1 for x in src_roi.values() if len(x) != 0]), "전체: 789개")    # 639/789개
print(src_roi.items())
print("이게뭐지", [x for x in src_roi.values() if len(x) != 0])


# OCR 수행
# x.run_OCR(30, 32)

# dict_a = {"소스1":{(1,2,3,4), (2,3,4,5)}, "소스2":{(5,5,5,5), (6,6,6,6)}}
# list_a = ["소스1", "소스2"]
# for key in dict_a.keys():
#     print(key)
#     print(type(key))
#     print(dict_a[key])
# for key in list_a:
#     print(key)
#     print(type(key))
#     print(dict_a[key])
