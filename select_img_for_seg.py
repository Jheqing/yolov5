import os

import cv2
from tqdm import tqdm
import shutil

path="./dataset/feces_dataset/"
selectPath="./dataset/selectpath/"
filesPath = os.listdir(path)

def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:

        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False

mkdir(selectPath)

files=tqdm(filesPath)
for file in files:
    absolute_file_path = os.path.join(path, file)
    if '.txt' in absolute_file_path.lower():
        file = open(absolute_file_path)
        lines=file.readlines()
        file.close()
        if len(lines)>0:
            isSelected=False
            for line in lines:
                line.replace("\n","")
                curLine = line.strip().split(" ")
                c1,x1,y1,w1,h1=curLine
                c, x, y, w, h =int(c1),float(x1),float(y1),float(w1),float(h1)

                # if c not in [4,5]:
                if c  in [11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,31,32,33,34,35,36,37]:
                    isSelected = True
                    break
            if isSelected:
                scrtxt = os.path.join(path, file.name.replace(path, ""))
                destxt = os.path.join(selectPath, file.name.replace(path, ""))
                # jpg
                filejpg = open(str(absolute_file_path).replace("txt", "jpg"))
                scrjpg = os.path.join(path, filejpg.name.replace(path, ""))
                filejpg.close()
                desjpg = os.path.join(selectPath, filejpg.name.replace(path, ""))
                shutil.copyfile(scrjpg, desjpg)
                shutil.copyfile(scrtxt, destxt)