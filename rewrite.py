import os
from tqdm import tqdm
import shutil
def rewrite_img2(path):
    from PIL import Image
    from PIL import ImageOps
    import PIL
    files = os.listdir(path)
    files = tqdm(files)
    for i, file in enumerate(files):
       if ("jpg"  in file ):
           image = Image.open(path+ "/"+file)
           try:
               # 尝试修复图像
               fixed_image = ImageOps.exif_transpose(image)
               fixed_image.save(path + "/" + file.replace(".jpg", ".jpg"))
           except PIL.UnidentifiedImageError:
               print("无法修复 JPEG 数据：",file)

if __name__ == '__main__':
    rewrite_img2("./dataset/niao_dataset/")