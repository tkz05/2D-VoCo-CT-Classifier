from __future__ import division, print_function
import torch
import torch.nn as nn

# 定義基本的卷積區塊
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

# 下採樣模組
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 上採樣模組
class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 定義 Encoder（下採樣部分）
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.in_conv = ConvBlock(params['in_chns'], params['feature_chns'][0], params['dropout'][0])
        self.down1 = DownBlock(params['feature_chns'][0], params['feature_chns'][1], params['dropout'][1])
        self.down2 = DownBlock(params['feature_chns'][1], params['feature_chns'][2], params['dropout'][2])
        self.down3 = DownBlock(params['feature_chns'][2], params['feature_chns'][3], params['dropout'][3])
        self.down4 = DownBlock(params['feature_chns'][3], params['feature_chns'][4], params['dropout'][4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

# 定義 Decoder（上採樣部分）
class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.up1 = UpBlock(params['feature_chns'][4], params['feature_chns'][3], params['feature_chns'][3], dropout_p=0.0)
        self.up2 = UpBlock(params['feature_chns'][3], params['feature_chns'][2], params['feature_chns'][2], dropout_p=0.0)
        self.up3 = UpBlock(params['feature_chns'][2], params['feature_chns'][1], params['feature_chns'][1], dropout_p=0.0)
        self.up4 = UpBlock(params['feature_chns'][1], params['feature_chns'][0], params['feature_chns'][0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(params['feature_chns'][0], params['class_num'], kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.up4(x7, x0)
        return self.out_conv(x8)

# 定義 UNet
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()
        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'bilinear': False,
            'acti_func': 'relu'
        }
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
