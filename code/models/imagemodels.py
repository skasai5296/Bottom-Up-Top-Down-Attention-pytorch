import torch
import torch.nn as nn

# Bottleneck layer.
class Bottleneck(nn.Module):
    def __init__(self, C, repeat=1):
        super(Bottleneck, self).__init__()
        self.bottle_c = int(C//2)
        self.repeat = repeat
        self.model = nn.ModuleList()
        for i in range(self.repeat):
            self.model.append(name='bottle{}'.format(i+1), nn.Sequential(
                    nn.Conv2d(C, self.bottle_c, 1, stride=1, padding=1),
                    nn.BatchNorm2d(self.bottle_c),
                    nn.LeakyReLU(),
                    nn.Conv2d(self.bottle_c, C, 3, stride=1, padding=1),
                    nn.BatchNorm2d(C),
                    nn.LeakyReLU()
                    )

    def forward(self, x):
        for layer in self.model:
            x += layer(x)
        return x

# Downsampling layer.
class Down(nn.Module):
    def __init__(self, C):
        super(Down, self).__init__()
        self.down_c = C*2
        self.model = nn.Sequential(
                nn.Conv2d(C, self.down_c, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.bottle_c),
                nn.LeakyReLU()
                )

    def forward(self, x):
        return self.model(x)


class YOLOv3(nn.Module):
    def __init__(self, in_c, imsize=256, resnumbers=[1, 2, 8, 8, 4]):
        super(YOLO, self).__init__()
        # input : (bs, 3, H=256, W=256)
        self.model = nn.Sequential()
        self.current_c = 32
        self.model.add_module(name='conv1', nn.Conv2d(in_c, self.current_c, 3, stride=1, padding=1))
        self.repeat = resnumbers
        self.current_im = imsize
        for i in range(len(self.repeat)):
            self.model.add_module(name='down{}'.format(i+1), Down(self.current_c))
            self.current_c *= 2
            self.model.add_module(name='bottle{}'.format(i+1), Bottleneck(self.current_c, repeat=self.repeat[i]))
        self.model.add_module(nn.AvgPool2d())

    def forward(self, x):
        return None

"""
Region Proposal Network
inputs convolutional feature maps and outputs scores and coordinates for k anchors

input : (B, C, H, W)
output : {"scores" : (B, 2k, H, W), "bbs" : (B, 4k, H, W)}
"""
class RPN(nn.Module):
    """
    cin : input channels
    """
    def __init__(self, C, k, ascales, aratios):
        super(RPN, self).__init__()

        self.cin = C
        self.anum = k
        self.ascales = ascales
        self.aratios = aratios

        # strided window
        self.sw = nn.Conv2d(C, 512, 3, 1, 1)
        self.act = nn.ReLU()

        # classification layer
        self.cls_num = len(self.ascales) * len(self.aratios) * 2  # 2k
        self.cls = nn.Conv2d(512, self.cls_num, 1, 1, 0)

        # bounding box layer
        self.bb_num = len(self.ascales) * len(self.aratios) * 4  # 4k
        self.reg = nn.Conv2d(512, self.bb_num, 1, 1, 0)

        self.model = []

    def forward(self, x):

        
        return None
