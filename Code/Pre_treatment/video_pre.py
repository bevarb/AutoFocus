import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Data.dataset import read_test
from Data.dataset import data_set
from Model.cnn_regression_model import model_test
from visualize.plot_fig import vis_data
import pandas as pd
from torchvision import transforms as T
import cv2 as cv
from PIL import Image

path1 = "D:\m21top81_step300nmsubtractbng.avi"
videoCapture = cv.VideoCapture(path1)
fps = videoCapture.get(cv.CAP_PROP_FPS)
size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv.CAP_PROP_FRAME_COUNT)
print(fps, size, fNUMS)
img = []
for i in range(int(fNUMS)):
    rat, frame = videoCapture.read()
    img.append(frame)
new_img = []
for i in range(int(fNUMS) - 1):
    img_0 = img[i]
    img_1 = img[i+1]
    sub = cv.subtract(img_0, img_1)
    cv.imshow("sub", sub)
    cv.waitKey(400)