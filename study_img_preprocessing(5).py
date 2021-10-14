from prep_OCR import OCR_prep


# 전처리 수행
# 딕셔너리 반환 - key:이미지 이름, value:데이터 영역 정보 (x, y, w, h)
x = OCR_prep('./data/ElectricityMeter', './data/roi')
# src_roi = x.preprocess_8seg(25, 30)
src_roi = x.preprocess_8seg(0,-1)
# print("8segment ROI를 탐지한 이미지 개수:", sum([1 for x in src_roi.values() if len(x) != 0]), ", 전체: 789개")
# print("8segment ROI를 2개 탐지한 이미지 개수:", sum([1 for x in src_roi.values() if len(x) == 2]), ", 전체: 789개")
# print("8segment ROI를 탐지 개수:", [len(x) for x in src_roi.values() if len(x) != 0], ", 전체: 789개")


# x.no_box(0, 10)

# x.run_OCR(134, 136)


# x.find_4in8seg(0,3)
