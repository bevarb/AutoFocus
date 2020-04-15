import cv2
root = "D:\Program_Data\First\Raw_data\\2020.3.3_150nmAu_100X_10us_change focus\\test_1\Pos0\\img_000000000_Default_000.tif"
img = cv2.imread(root)
print(img.shape)
print(img.dtype)
