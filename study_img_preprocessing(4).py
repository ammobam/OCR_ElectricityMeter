from prep_OCR import OCR_prep


# 전처리 수행
# 딕셔너리 반환 - key:이미지 이름, value:데이터 영역 정보 (x, y, w, h)
x = OCR_prep('./data/ElectricityMeter', './data/roi')
x.preprocess(0, 30)

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
