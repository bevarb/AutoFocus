import os
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as transforms
import random
class split_set():
    def __init__(self, root, train_size):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        '''
        self.train_data = []
        self.valid_data = []
        classes = [os.path.join(root, img) for img in os.listdir(root)]  # 这里要求能够输出所有图片的路径，二维矩阵
        imgs = []
        for cla in classes:
            temp = [os.path.join(cla, img) for img in os.listdir(cla)]
            random.shuffle(temp)
            imgs.append(temp)
        imgs_num = []
        for classes in imgs:
            imgs_num.append(len(classes))
        for i in range(len(imgs_num)):
            # num = imgs_num[i]
            num = 0
            if imgs_num[i] >= 300:
                num = 300
            else:
                num = imgs_num[i]
            self.train_data.extend(imgs[i][: int(train_size * num)])
            self.valid_data.extend(imgs[i][int(train_size * num):num])

    def get_data(self):
        random.shuffle(self.train_data)
        random.shuffle(self.valid_data)
        return self.train_data, self.valid_data

class read_test():
    '''读取测试集数据，返回数据路径'''
    def __init__(self, root):
        classes = [os.path.join(root, cla) for cla in os.listdir(root)]  # 这里要求能够输出所有图片的路径，二维矩阵
        classes = sorted(classes, key=lambda x: int(x.split('_')[-1]))
        imgs = []
        self.test_data = []
        for cla in classes:
            temp = [os.path.join(cla, img) for img in os.listdir(cla)]
            temp = sorted(temp, key=lambda x: int(x.split('.')[-2].split('\\')[-1]))
            imgs.append(temp)
        for i in range(len(imgs)):
            self.test_data.extend(imgs[i])
    def get_data(self):
        return self.test_data


class data_set(data.Dataset):
    '''继承Dataset的数据加载类，读取图片、数据增强、返回图片及标签'''
    def __init__(self, data, transforms=None, flag=None, T=None, step=None):
        self.flag = flag
        self.imgs = data
        self.T, self.step = T, step
        self.transforms = transforms


    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        label = [2837.0,
                 2837.6,
                 2838.0,
                 2838.6,
                 2839.0,
                 2839.6,
                 2840.0,
                 2840.6,
                 2841.0,
                 2841.6,
                 2842.0,
                 2842.6,
                 2843.0,
                 2843.6,
                 2844.0]
        if self.flag == "video":
           # print(index)
            data = Image.fromarray(self.imgs[index])
            self.data = self.transforms(data)
            i = index % self.T
           # print(i)
            self.label = i * self.step

        else:
            img_path = self.imgs[index]
            # print(self.imgs[index].split('.'))
            index_tolabel = int(self.imgs[index].split('.')[-2].split("\\")[-2].split("_")[-1])
            self.label = label[index_tolabel] - 2837
            data = Image.open(img_path)
            self.data = self.transforms(data)
        return self.data, self.label

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)


