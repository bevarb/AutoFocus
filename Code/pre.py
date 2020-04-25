import cv2
import os
root = "D:\Project_Data\First\Raw_data\\2020.3.3_150nmAu_100X_10us_change focus\\test_1\Pos0"
imgs = [os.path.join(root, img) for img in os.listdir(root)]
imgs = sorted(imgs, key=lambda x: int(x.split('_')[-3]))
for i in range(1, len(imgs)):
    img0 = cv2.imread(imgs[0])
    print(img0)
    img1 = cv2.imread(imgs[i])
    print(img1)
    sub = img1 -img0
    print(sub)
    cv2.imshow('sub', sub)
    cv2.waitKey(100)
# img = cv2.imread(root)
# print(img.shape)
# print(img.dtype)
