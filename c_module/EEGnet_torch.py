import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor


class EEGNet(nn.Module):
    def __init__(self, classes_num, num_channels=64, time_points=640, F1=8, F2=16, D=2, drop_out=0.25):
        super(EEGNet, self).__init__()
        self.C = num_channels
        self.T = time_points
        self.F1 = F1  # num of features in layer 1
        self.F2 = F2  # num of features in layer 2
        self.D = D
        self.drop_out = drop_out

        # block 1: Convolution
        self.block_1 = nn.Sequential(
            # Padding to make sure the output shape unchanged,  (left, right, up, bottom)
            nn.ZeroPad2d((floor((self.T - 1) / 4), ceil((self.T - 1) / 4), 0, 0)),
            nn.Conv2d(  # input shape (1, C, T)
                in_channels=1,
                out_channels=self.F1,
                kernel_size=(1, self.T // 2),
                bias=False
            ),  # output shape (F1, C, T)
            nn.BatchNorm2d(self.F1)  # output shape (F1, C, T)
        )

        # block 2: Depthwise Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(  # input shape (F1, C, T)
                in_channels=self.F1,
                out_channels=self.D * self.F1,
                kernel_size=(self.C, 1),
                groups=self.F1,  # not FC
                bias=False
            ),  # output shape (D * F1, 1, T)
            nn.BatchNorm2d(self.D * self.F1),  # output shape (D * F1, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (D * F1, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (D * F1, 1, T//4)
        )

        # block 3: Separable Convolution
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((floor((self.T - 1) / 16), ceil((self.T - 1) / 16), 0, 0)),
            nn.Conv2d(  # input shape (D * F1, 1, T//4)
                in_channels=self.D * self.F1,
                out_channels=self.F2,
                kernel_size=(1, self.T // 8),
                groups=self.F2,  # not FC
                bias=False
            ),  # output shape (F2, 1, T//4)
            nn.Conv2d(  # input shape (F2, 1, T//4)
                in_channels=self.F2,
                out_channels=self.F2,
                kernel_size=(1, 1),
                bias=False
            ),  # output shape (F2, 1, T//4)
            nn.BatchNorm2d(self.F2),  # output shape (F2, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (F2, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear(self.F2 * (self.T // 32), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        # return F.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    input = torch.randn(32, 1, 3, 1152)

    model = EEGNet(2, 3, 1152)

    out = model(input)

    print(model)
    # print(out.shape)
