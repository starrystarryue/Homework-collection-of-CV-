import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        # 2. define full-connected layer to classify
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
         
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
          
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
          
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        out = self.fc_layers(x)
        return out


class ResBlock(nn.Module):
    ''' residual block'''
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        # 1. define double convolution
             # convolution
             # batch normalization
             # activate function
             # ......
        self.conv1=nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        if in_channel!=out_channel or stride!=1:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.

        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        identity=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if hasattr(self,'shortcut'):
            identity=self.shortcut(x)
        out+=identity
        out=F.relu(out)
        return out


class ResNet(nn.Module):
    '''residual network'''
    def __init__(self):
        super().__init__()

        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify
        self.conv=nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn=nn.BatchNorm2d(16)
        self.layer1=nn.Sequential(*[ResBlock(16,16,1) for _ in range(7)])
        self.layer2 = nn.Sequential(
             ResBlock(16, 32, 2),
            *[ResBlock(32, 32, 1) for _ in range(6)]
        )
        self.layer3 = nn.Sequential(
            ResBlock(32, 64, 2),
            *[ResBlock(64, 64, 1) for _ in range(6)]
        )
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.drop= nn.Dropout(0.2) 
        self.fc = nn.Linear(64, 10)
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avpool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        out = self.fc(x)
        return out
    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        # 1. define convolution
             # 1x1 convolution
             # batch normalization
             # activate function
             # 3x3 convolution
             # ......
             # 1x1 convolution
             # ......
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        
        hidden_channel = int(out_channel / bottle_neck)
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=stride, padding=1, groups=group, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
   
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self, x):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        identity = x
        if hasattr(self, 'shortcut'):
            identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        out = F.relu(out)
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify
        bottleneck = 4  # bottleneck=out_channel/hidden_channel
        group = 8       # 分组卷积group数
        num_blocks = [3, 3, 3]  # 每个stage的block数
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self._make_stage(64, 64, num_blocks[0], bottleneck, group, stride=1),
            self._make_stage(64, 128, num_blocks[1], bottleneck, group, stride=2),
            self._make_stage(128, 256, num_blocks[2], bottleneck, group, stride=2)
        )
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(256, 10)

    def _make_stage(self, in_channel, out_channel, num_blocks, bottleneck, group, stride):
        layers = []
        layers.append(ResNextBlock(in_channel, out_channel, bottleneck, group, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNextBlock(out_channel, out_channel, bottleneck, group, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.features(x)

        x=self.avpool(x)
        x = x.view(x.size(0), -1)
        x=self.drop(x)
        out = self.fc(x)
        return out

