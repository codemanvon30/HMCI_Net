import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from add_models import *
import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DepthwiseConvBlock(nn.Module): 
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.ReLU6(),
        )
    def forward(self, x):
        return x + self.net(x)

class SpectralAttention(nn.Module):
    def __init__(self, heads, patch_size, dropout):
        super().__init__()
        self.heads = heads
        self.patch_size = patch_size
        self.scale = patch_size ** -1
        self.conv_project = nn.Sequential(
            nn.Conv3d(1, 3 * heads, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            Rearrange('b h x y s -> b s (h x y)'),
            nn.Dropout(dropout)
        )
        self.reduce_k = nn.Conv2d(self.heads, self.heads, kernel_size=(3, 1), padding=(1, 0), stride=(4, 1), groups=self.heads, bias=False)
        self.reduce_v = nn.Conv2d(self.heads, self.heads, kernel_size=(3, 1), padding=(1, 0), stride=(4, 1), groups=self.heads, bias=False)
        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels=heads, out_channels=1, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            nn.Dropout(dropout),
            Rearrange('b c x y s -> b c s x y'),
            nn.LayerNorm((patch_size, patch_size)),
            Rearrange('b c s x y -> b c x y s')
        )
    def forward(self, x):
        qkv = self.conv_project(x).chunk(3, dim=-1)
        q, k, v = map(lambda a: rearrange(a, 'b s (h d) -> b h s d', h=self.heads), qkv)
        k = self.reduce_k(k)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        v = self.reduce_v(v)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b c s (x y) -> b c x y s', x=self.patch_size, y=self.patch_size)
        out = self.conv_out(out)
        return out

class SpectralTransformerBlock(nn.Module):
    def __init__(self, heads, patch_size, dropout):
        super().__init__()
        self.attention = SpectralAttention(heads, patch_size, dropout)
        self.ffn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            nn.ReLU6(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x
    
class MultiScaleSpectralFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels
        self.spectral1 = nn.Sequential(
            nn.Conv3d(channels, self.c, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral11 = nn.Sequential(
            nn.Conv3d(channels, self.c, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral111 = nn.Sequential(
            nn.Conv3d(channels, self.c, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        
        self.spectral2 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 5), padding=(0, 0, 2)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral22 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral222 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 5, 1), padding=(0, 0, 2)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        
        self.spectral3 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 7), padding=(0, 0, 3)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral33 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral333 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        
        self.spectral4 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 11), padding=(0, 0, 5)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral44 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(11, 1, 1), padding=(5, 0, 0)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral444 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 11, 1), padding=(0, 5, 0)),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
     
    def forward(self, x):
        x1 = self.spectral1(x)
        x11 = self.spectral11(x1)
        x111 = self.spectral111(x11)
        
        x2 = self.spectral2(x)
        x22 = self.spectral22(x2 + x1)
        x222 = self.spectral22(x22 + x11)
        
        x3 = self.spectral3(x)
        x33 = self.spectral33(x3 + x2)
        x333 = self.spectral333(x33 + x22)
        
        x4 = self.spectral4(x)
        x44 = self.spectral44(x4 + x3)
        x444 = self.spectral44(x44 + x33)
        
        mspe = torch.cat((x111, x222, x333, x444), dim=1)
        return mspe


class CrossAttentionConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        attn_map = self.attention(x2)
        return x1 * attn_map

class CrossAttentionTransformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        attn_map = self.attention(x2)
        return x1 * attn_map

class DualBranchCrossTransformer(nn.Module):
    def __init__(self, channels, patch_size, heads, dropout, fc_dim, band_reduce):
        super().__init__()
        self.cross_conv = CrossAttentionConv(channels)
        self.cross_transformer = CrossAttentionTransformer(channels)
        self.conv_branch = DepthwiseConvBlock(channels)
        
        self.transformer_branch = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=(1, 1, 7), padding=(0, 0, 3), stride=(1, 1, 1)),
            SpectralTransformerBlock(heads, patch_size, dropout)
        )
        self.conv_out = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, band_reduce), padding=(1, 1, 0), groups=channels),
            nn.BatchNorm3d(channels), 
            nn.ReLU6()
        )
        self.transformer_out = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(3, 3, band_reduce), padding=(1, 1, 0)),
            nn.BatchNorm3d(channels),
            nn.ReLU6()
        )
        self.out = nn.Sequential(
            nn.Conv3d(4 * channels, fc_dim, kernel_size=1),
            nn.BatchNorm3d(fc_dim),
            nn.ReLU6()
        )
    def forward(self, x):
        x_conv = self.conv_branch(x)
        x_transformer = self.transformer_branch(x)
        
        cross_conv = self.conv_out(self.cross_conv(x_conv, x_transformer))
        cross_transformer = self.transformer_out(self.cross_transformer(x_transformer, x_conv))
        
        conv_out = self.conv_out(x_conv)
        transformer_out = self.transformer_out(x_transformer)
        
        out = self.out(torch.cat((conv_out, cross_conv, transformer_out, cross_transformer), dim=1))
        return out

    
class DualBranchTransformer(nn.Module):
    def __init__(self, channels, patch_size, heads, dropout, fc_dim, band_reduce):
        super().__init__()
        self.conv_branch = DepthwiseConvBlock(channels)
        
        self.transformer_branch = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=(1, 1, 7), padding=(0, 0, 3), stride=(1, 1, 1)),
            SpectralTransformerBlock(heads, patch_size, dropout)
        )
        self.conv_out = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, band_reduce), padding=(1, 1, 0), groups=channels),
            nn.BatchNorm3d(channels), 
            nn.ReLU6()
        )
        self.transformer_out = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(3, 3, band_reduce), padding=(1, 1, 0)),
            nn.BatchNorm3d(channels),
            nn.ReLU6()
        )
        self.out = nn.Sequential(
            nn.Conv3d(2 * channels, fc_dim, kernel_size=1),
            nn.BatchNorm3d(fc_dim),
            nn.ReLU6()
        )
    def forward(self, x):
        x_conv = self.conv_branch(x)
        x_transformer = self.transformer_branch(x)
        conv_out = self.conv_out(x_conv)
        transformer_out = self.transformer_out(x_transformer)
        out = self.out(torch.cat((conv_out, transformer_out), dim=1))
        return out

class SpectralFeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels // 4
        self.spectral1 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral2 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 7), padding=(0, 0, 3), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral3 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 11), padding=(0, 0, 5), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
        self.spectral4 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 15), padding=(0, 0, 7), groups=self.c),
            nn.BatchNorm3d(self.c),
            nn.ReLU6()
        )
     
    def forward(self, x):
        x1 = self.spectral1(x[:, 0:self.c, :])
        x2 = self.spectral2(x[:, self.c:2*self.c, :])
        x3 = self.spectral3(x[:, 2*self.c:3*self.c, :])
        x4 = self.spectral4(x[:, 3*self.c:, :])
        mspe = torch.cat((x1, x2, x3, x4), dim=1)
        return mspe

class DualBranchTransformerNetwork(nn.Module):
    def __init__(self, channels=16, patch_size=9, bands=270, num_class=9, fc_dim=16, heads=2, dropout=0.1):
        super().__init__()
        self.band_reduce = (bands - 7) // 2 + 1
        self.stem = nn.Conv3d(1, channels, kernel_size=(1, 1, 7), padding=0, stride=(1, 1, 2))
        
        self.spectral_feature_extractor = MultiScaleSpectralFeatureExtractor(channels)
        self.dual_branch_transformer = DualBranchCrossTransformer(channels * 4, patch_size, heads, dropout, fc_dim, self.band_reduce)
        
        self.view_process = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 1, 3), padding=0, stride=(1, 1, 1)),
            nn.BatchNorm3d(channels),  
            nn.ReLU6(),  
            nn.AdaptiveAvgPool3d((9, 9, 1))
        )                               
       
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(fc_dim * 5, num_class)
        )
        
    def forward(self, x):
        band_counts = [30, 40, 50, 60]
        b, _, _, _, _ = x.shape
        
        m1, m2, m3, m4 = random_spectral_selection(x, band_counts)
        views = [m1, m2, m3, m4]
        del m1, m2, m3, m4
        m_view = []
        
        x = self.stem(x)
        x = self.spectral_feature_extractor(x)
        feature_main = self.dual_branch_transformer(x)
        del x
        
        for m in views:
            feature = self.view_process(m)
            m_view.append(feature)
        del feature
        
        combined_feature = torch.cat([feature_main, m_view[0] * 0.5, m_view[1] * 0.5, m_view[2] * 0.5, m_view[3] * 0.5], dim=1)
        return self.fc(combined_feature)

def random_spectral_selection(input_tensor, band_counts):
    total_bands = input_tensor.shape[-1]
    outputs = []
    for count in band_counts:
        if count > total_bands:
            raise ValueError("单个视图要求的光谱带数超过了实际带数")

        selected_indices = random.sample(range(total_bands), count)
        selected_indices.sort()
        outputs.append(input_tensor[..., selected_indices])
    return outputs

if __name__ == '__main__':
    model = DualBranchTransformerNetwork(bands=270, num_class=9)
    device = torch.device("cuda:1")
    model = model.to(device)
    model.eval()
    input = torch.randn(4, 1, 9, 9, 270).to(device)
    y = model(input)
    print(y.shape)