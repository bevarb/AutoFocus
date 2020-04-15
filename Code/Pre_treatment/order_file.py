import os
import random
def order_file(path, flag):
    '''将文件中的图片按顺序排列'''
    listdir = os.listdir(path)
    listdir = sorted(listdir, key=lambda x: int(x.split('.')[-2]))
    for i in range(len(listdir)):
        os.rename(path + "\\" + listdir[i], path + "\\" + str(i) + flag)

def disorder_file(path):
    '''将文件中的所有图片打乱'''
    listdir = os.listdir(path)
    random.shuffle(listdir)
    for i in range(len(listdir)):
        os.rename(path + "\\" + listdir[i], path + "\\" + str(i) + ".tif")

root = 'D:\Project_Data\First\Box_data'
for dir in os.listdir(root):
    path = root + "\\" + dir
    order_file(path, ".tif")
    # 用来打乱所有图片的顺序，由于可能会存在两个名字一样的图片，所以先将图片换个名字
    # order_file(path, ".tifff")
    # disorder_file(path)