import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


##########################################################################################################
#[1] 注意力机制

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=True),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 适应三维数据的卷积层
        self.conv = nn.Conv3d(2, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算通道平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算通道最大值
        x_combined = torch.cat([avg_out, max_out], dim=1)  # 在通道维度合并
        x_out = self.conv(x_combined)
        return self.sigmoid(x_out)





# 多尺度+CBAM注意力
class MSpeFEcbam(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels // 4
          
        self.spectral1 = nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=self.c)
        self.spectral2 = nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 7), padding=(0, 0, 3), groups=self.c)
        self.spectral3 = nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 11), padding=(0, 0,5), groups=self.c)
        self.spectral4 = nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 15), padding=(0, 0, 7), groups=self.c)
        
        self.ca = ChannelAttention(self.c * 4)
        self.sa = SpatialAttention()

        self.bn = nn.BatchNorm3d(self.c * 4)
        self.relu = nn.ReLU6()

    def forward(self, x):
        # print(self.c)
        x1 = self.spectral1(x[:, 0:self.c, :])
        
        x2 = self.spectral2(x[:, self.c:2*self.c, :])
        x3 = self.spectral3(x[:, 2*self.c:3*self.c, :])
        x4 = self.spectral4(x[:, 3*self.c:, :])
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)

        mspe = torch.cat((x1, x2, x3, x4), dim=1)
        # print(mspe.shape)
        mspe = self.ca(mspe) * mspe 
        # print(mspe.shape) 
        mspe = self.sa(mspe) * mspe  
        # print(mspe.shape)
        mspe = self.bn(mspe)
        return self.relu(mspe)