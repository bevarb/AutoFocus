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

path = "Code\Data\Second\m2top6_step200nm_subtractbng.avi"
path1 = "D:\m2top6_step200nm_subtractbng.avi"
videoCapture = cv.VideoCapture(path1)
fps = videoCapture.get(cv.CAP_PROP_FPS)
size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv.CAP_PROP_FRAME_COUNT)
print(fps, size, fNUMS)



transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),  # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
rat, frame = videoCapture.read()
Target = []
right_num = 0
model = model_test()
model.load_state_dict(torch.load('../checkpoints/model_min_loss.pt'))
model.eval()
while rat:
        input = transform(frame)
        output = model(input)
        print(output)
        rat, frame = videoCapture.read()



# read_test = read_test('D:\Project_Data\First\Test_data')
# test_data = read_test.get_data()
# test_set = data_set(test_data, transforms=transform)
# test_loader = DataLoader(test_set, batch_size=test_set.__len__())

# def test(test_loader):
#     test_correct = 0
#     test_predict = 0
#     target_data = 0
#     for data, target in test_loader:
#         test_output = model(data)
#         test_output = test_output.view(target.shape[0])
#         test_predict = torch.round(test_output * 10) / 10
#         test_correct += torch.sum(test_predict == target.data)
#         target_data = target.data
#         print("predict:", test_predict)
#         print('target:', target.data)
#         test_correct += torch.sum(torch.abs((test_predict - target.data)) < 0.2)
#     test_acc = int(test_correct) / len(test_loader.dataset)
#     print('测试集准确率为:%.2f' % test_acc)
#     return test_predict, target_data
#
# predict_data, target_data = test(test_loader)
# A = [predict_data.detach().numpy(), target_data.detach().numpy()]
# df = pd.DataFrame(A, index=["predict", "target"], columns=None)
# df = df.T
# print(df)
# writer = pd.ExcelWriter('test.xlsx')		# 写入Excel文件
# df.to_excel(writer, 'page_1', float_format='%.2f')
# writer.save()
# writer.close()