import os
from tqdm import tqdm
import shutil
import cv2
from collections import defaultdict

labels = ['1_unknow'
    , '2_hongxibao'
    , '3_baixibao'
    , '401_zhenjun'
    , '5_nongqiu'
    , '6_zhifangdi'
    , '7_shangpi'
    , '8_dianfenkeli'
    , '9_caosuangaijiejing'
    , '10_jushixibao'
    , '11_linzhibaozi'
    , '12_ganxichongluan'
    , '13_huichongluan'
    , '14_gouchongluan'
    , '15_naochongluan'
    , '16_bianchongluan'
    , '17_taochongluan'
    , '18_jiangpianchongluan'
    , '19_xuexichongluan'
    , '20_dichong'
    , '21_xiakeleidengjiejing'
    , '22_feixichongluan'
    , '23_amiba'
    , '24_lanshijiadibianmaochong'
    , '25_renyanangyuanchong'
    , '26_yinbaozichong'
    , '27_fenleiyuanxianchong'
    , '28_xueye'
    , '29_zhiwuzuzhi'
    , '30_jiedizuzhi'
    , '31_jixianwei'
    , '2301_amiba'
    , '1702_taochongluan'
    , '1901_xuexichongluan'
    , '2302_amiba'
    , '2303_amiba'
    , '1703_taochongluan'
    , '2002_dichong'
    , '2304_amiba'
    , '2601_baozichong'
    , '1706_taochongluan'
    , '602_zhifangdi'
    , '603_zhifangdi'
    , '1704_taochongluan'
    , '1705_taochongluan'
    , '1902_xuexichongluan'
    , '1903_xuexichongluan'
    , '2001_dichong'
    , '2101_jiejing'
    , '2305_amiba'
    , '2401_lanshijiadibianmaochong'
    , '2402_lanshijiadibianmaochong'
    , '2602_baozichong'
    , '2603_baozichong'
    , '2604_baozichong'
    , '2605_baozichong'
    , '2701_xianchong'
    , '3201_xianmaochong'
    , '3500_nianyesi'
    , '3700_xijun'
    , '402_zhenjun'
    , '2608_baozichong'
          ]

dict = defaultdict(int)


def calcObj():
    path = 'D:\\fence_dataset\\train\\dataset\\feces_dataset\\txt\\'

    for i in labels:
        dict[i] = 0

    filesPath = os.listdir(path)

    files = tqdm(filesPath)
    for file1 in files:
        absolute_file_path = os.path.join(path, file1)

        if '.txt' in absolute_file_path.lower():
            file = open(absolute_file_path)
            lines = file.readlines()

            if len(lines) > 0:
                for line in lines:
                    line.replace("\n", "")
                    curLine = line.strip().split(" ")
                    c1, x1, y1, w1, h1 = curLine
                    # c1, x1, y1, w1, h1 ,_= curLine
                    c, x, y, w, h = int(c1), float(x1), float(y1), float(w1), float(h1)
                    dict[labels[c]] = dict[labels[c]] + 1

    for k, v in dict.items():
        print(k)
    for k, v in dict.items():
        print(v)


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


def selectlabelImage(src, label, des):
    filesPath = os.listdir(label)
    mkdir(des)
    nameSet = set()
    files = tqdm(filesPath)
    for file in files:
        absolute_file_path = os.path.join(src, file)
        if '.txt' in absolute_file_path.lower():
            name = str(file).split(".")[0]
            nameSet.add(name)

    for name in nameSet:
        scrtxt = os.path.join(src, name + ".txt")
        destxt = os.path.join(des, name + ".txt")
        scrjpg = os.path.join(src, name + ".jpg")
        desjpg = os.path.join(des, name + ".jpg")
        if os.path.exists(scrjpg):
            # shutil.copyfile(scrjpg, desjpg)
            # shutil.copyfile(scrtxt, destxt)
            shutil.move(scrjpg, desjpg)
        else:
            print(scrjpg)
        if os.path.exists(scrtxt):
            shutil.move(scrtxt, destxt)
        else:
            print(scrtxt)


if __name__ == '__main__':
    calcObj()
    # selectlabelImage('dataset/feces_dataset/', 'dataset/label/', 'dataset/select/')
