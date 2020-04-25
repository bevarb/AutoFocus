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
# 1-模型
model = model_test()
model.load_state_dict(torch.load('../checkpoints/model_regression_30%.pt'))
model.eval()

# 2-输入数据

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

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),  # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_set = data_set(img, transforms=transform, flag="video", T=35, step=0.3)

test_loader = DataLoader(test_set, batch_size=64,
                         # shuffle=True,
                          )

Target = []
right_num = 0


def test(test_loader):
    test_correct = 0
    flag = 0
    predict = 0
    label = 0
    for data, target in test_loader:
        test_output = model(data)
        test_output = test_output.view(target.shape[0])
        test_predict = torch.round(test_output * 10) / 10
        test_correct += torch.sum(test_predict == target.data)
        print("predict:", test_predict)
        print('target:', target.data)
        test_correct += torch.sum(torch.abs((test_predict - target.data)) < 0.2)
        if flag != 0:
            predict = torch.cat([predict, test_predict], dim=0)
            label = torch.cat([label, target.data], dim=0)
        else:
            predict = test_predict
            label = target.data
        flag += 1

    #     a = test_predict.detach().numpy().tolist()
    #     b = target.data.detach().numpy().tolist()
    #    # print(b.dtype)
    #     predict.extend(a)
    #     target.extend(b)
    test_acc = int(test_correct) / len(test_loader.dataset)
    print('测试集准确率为:%.2f' % test_acc)
    return predict, label

predict_data, target_data = test(test_loader)
A = [predict_data.detach().numpy(), target_data.detach().numpy()]
df = pd.DataFrame(A, index=["predict", "target"], columns=None)
df = df.T
print(df)
writer = pd.ExcelWriter('test.xlsx')		# 写入Excel文件
df.to_excel(writer, 'page_1', float_format='%.2f')
writer.save()
writer.close()