# define the CNN architecture
import torch.nn as nn
import math
class model_classify(nn.Module):
    def __init__(self):
        super(model_classify, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.drop_blocks = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128 * 8 * 8, 500),
            nn.ReLU(),
            nn.Linear(500, 8)
        )
        self._initialize_weights()

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        out = self.conv_blocks(x)
        # print('out.shape', out.shape)
        out = out.view(-1, 128 * 8 * 8)
        # print('out.shape', out.shape)
        y = self.drop_blocks(out)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()