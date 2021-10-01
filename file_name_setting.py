# 이미지 파일명 영어로 변환하기

# 기존 파일 이름 가져오기
import os
file_path = './img_src_0906'
file_names = os.listdir(file_path)

# 파일이름 다시 붙이기
i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = 'img_{0:0>6}.jpg'.format(str(i))
    #print(dst, type(dst))
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1


# 파일이름에서 확장자 제외하고 이름만 리스트에 저장하기
# 문자열.split(sep='구분자', maxsplit=분할횟수)

# 파일 이름 가져오기
file_path = './img_src_0906'
file_names = os.listdir(file_path)
# 확장자 제외한 이름만 저장할 리스트
src_names = []
for name in file_names:
    src_name, _jpg = name.split('.')
    #print(src_name)
    src_names.append(src_name)


#######################
from prep_img import File

f = File('./img_src_0906')
f.names
f.file_rename()