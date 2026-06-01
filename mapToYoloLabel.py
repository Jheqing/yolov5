import os
from tqdm import tqdm
import shutil
import cv2
from jointImg import  jointImg

# excrement
labels= [  '1_unknow'
         , '2_hongxibao'
         , '3_baixibao'
         , '4_zhenjun'
         , '5_nongqiu'
         , '6_zhifangdi'
         ,  '7_shangpi'
         ,   '8_dianfenkeli'
         ,  '9_caosuangaijiejing'
         ,   '10_jushixibao'
         ,   '11_linzhibaozi'
         ,   '12_ganxichongluan'
         ,   '13_huichongluan'
         ,   '14_gouchongluan'
         ,  '15_naochongluan'
         ,   '16_bianchongluan'
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
    , '2301_amiba' ]

#urine
# labels = ['100_unknown'
#     , '201_hongxibao'
#     , '202_hongxibao'
#     , '203_hongxibao'
#     , '301_baixibao'
#     , '302_baixibao'
#     , '400_zhenjun'
#     , '701_shangpi'
#     , '702_shangpi'
#     , '703_shangpi'
#     , '704_shangpi'
#     , '705_shangpi'
#     , '3301_guanxing'
#     , '3302_guanxing'
#     , '3303_guanxing'
#     , '3304_guanxing'
#     , '3305_guanxing'
#     , '3306_guanxing'
#     , '3307_guanxing'
#     , '3308_guanxing'
#     , '3401_jiejing'
#     , '3402_jiejing'
#     , '3403_jiejing'
#     , '3404_jiejing'
#     , '3405_jiejing'
#     , '3406_jiejing'
#     , '3407_jiejing'
#     , '3408_jiejing'
#     , '3500_nianyesi'
#     , '3600_jingzi'
#     , '3700_xijun'
#           , '204_hongxibao'
#           , '206_hongxibao'
#           , '207_hongxibao'
#           , '208_hongxibao'
#           , '209_hongxibao'
#           , '210_hongxibao'
#           , '211_hongxibao'
#           , '303_baixibao'
#           , '401_zhenjun'
#           , '402_zhenjun'
#           , '3409_jiejing'
#           , '3410_jiejing'
#           , '3701_xijun'
#           ]

srcPath = 'D:/Handletiling/src/'

srcW = 2448
srcH = 2040
path = srcPath



filesPath = os.listdir(path)

files = tqdm(filesPath)
for file1 in files:
    absolute_file_path = os.path.join(path, file1)
    if '.txt' in absolute_file_path.lower():
        file = open(absolute_file_path)
        lines = file.readlines()

        if len(lines) > 0:
            annotation = ""
            absolute_jpg_path=absolute_file_path.replace(".txt",".jpg")
            # print(absolute_jpg_path)

            if "3500_nianyesi" not in labels:
                img=cv2.imread(absolute_jpg_path)
                srcW = img.shape[1]
                srcH = img.shape[0]
            for line in lines:
                line.replace("\n", "")
                curLine = line.strip().split(" ")
                c1, prob1, x1, y1, w1, h1 = curLine
                c1=labels.index(c1)
                obj_id, prob, x, y, w, h = int(c1), float(prob1), float(x1), float(y1), float(w1), float(h1)

                annotation = annotation + "%s %s %s %s %s %s\n" % (obj_id, (x + w )* 0.5 / srcW, (y + h ) * 0.5/ srcH, (w-x) / srcW, (h-y) / srcH,prob1)
                # annotation = annotation + "%s %s %s %s %s\n" % (obj_id, (x + w )* 0.5 / srcW, (y + h ) * 0.5/ srcH, (w-x) / srcW, (h-y) / srcH)

                with open(absolute_file_path, 'w') as outfile:
                    outfile.write(annotation)

print("end")
jointImg(True)