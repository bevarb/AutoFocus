import argparse
import time
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Data.dataset import split_set
from Data.dataset import data_set
from torch.autograd import Variable
from Model.CNN_classify_model import model_classify
from tensorboardX import SummaryWriter
# ------------------------------------------------------------------------------
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.02, help="adam: learning rate")
#parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
#parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space 潜在空间的维数")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
#parser.add_argument("--channels", type=int, default=3, help="number of image channels")
#parser.add_argument("--sample_interval", type=int, default=1000, help="number of image channels")
opt = parser.parse_args()
#print(opt)
transform = transforms.Compose([
  #  transforms.Resize(opt.img_size),
    transforms.RandomHorizontalFlip(),  # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# step2: 数据
split_set = split_set('D:\Project_Data\First\Box_data', train_size=0.85)
train_data, valid_data = split_set.get_data()
train_set = data_set(train_data,  transforms=transform)
valid_set = data_set(valid_data, transforms=transform)
train_loader = DataLoader(train_set, batch_size=opt.batch_size,
                                  shuffle=True,
                                  )
valid_loader = DataLoader(valid_set, batch_size=opt.batch_size,
                                shuffle=True,
                                )

# step3: 创建模型
model = model_classify()
dummy_input = torch.rand(64, 3, 128, 128)
with SummaryWriter(comment='CNN1') as w:
    w.add_graph(model, (dummy_input,))


if train_on_gpu:  # move tensors to GPU if CUDA is available
    model.cuda()
# ------------------------------------------------------------------------------
# specify loss function
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=opt.lr)
valid_loss_min = np.Inf  # track change in validation loss
writer = SummaryWriter()
valid_acc_max = 0
for epoch in range(1, opt.n_epochs + 1):
    # keep track of training and validation loss

    train_loss = 0.0
    valid_loss = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    train_correct = 0
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        train_target = Variable(target)
        optimizer.zero_grad()  # clear the gradients of all optimized variables
        #print(data.shape)

        train_output = model(data)  # forward pass: compute predicted outputs by passing inputs to the model

        loss = criterion(train_output, train_target)  # calculate the batch loss
        # print(output)
        # print(target)

        loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters

        optimizer.step()  # perform a single optimization step (parameter update)

        train_loss += loss.item() * data.size(0)  # update training loss
        _, train_predict = torch.max(train_output.data, 1)
        train_correct += torch.sum(train_predict == train_target.data)

    ######################
    # validate the model #
    ######################
    model.eval()
    valid_correct = 0
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        data = Variable(data)
        target = Variable(target)
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        valid_output = model(data)  # forward pass: compute predicted outputs by passing inputs to the model
        # calculate the batch loss

        loss = criterion(valid_output, target)

        # update average validation loss
        valid_loss += loss.item() * data.size(0)
        _, predict = torch.max(valid_output.data, 1)
        # print(target)
        # print(predict)
        valid_correct += torch.sum(predict == target.data)
    train_acc = int(train_correct) / len(train_loader.dataset)
    valid_acc = int(valid_correct) / len(valid_loader.dataset)
    print('训练准确率为:%.2f ,验证准确率为:%.2f' % (train_acc, valid_acc))

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)
    writer.add_scalars('scalar/acc_loss', {'train_acc': train_acc,
                                     "valid_acc": valid_acc,
                                      'train_loss': train_loss,
                                      'valid_loss': valid_loss}, epoch)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min :
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        name = time.strftime('checkpoints\\model_min_loss' + '%m%d_%H:%M:%S.pt')
        torch.save(model.state_dict(), name)
        valid_loss_min = valid_loss
    elif valid_acc >= valid_acc_max:
        print('Validation accuration increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_acc_max,
            valid_acc))
        name = time.strftime('checkpoints\\model_max_acc' + '%m%d_%H:%M:%S.pt')
        torch.save(model.state_dict(), name)
        valid_acc_max = valid_acc
    # for i, (name, param) in enumerate(model.named_parameters()):
    #     if 'bn' not in name:
    #         writer.add_histogram(name, param, 0)
writer.close()