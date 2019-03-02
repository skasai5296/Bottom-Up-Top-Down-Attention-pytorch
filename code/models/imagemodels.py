import torch
import torch.nn as nn

class Faster_r_cnn(nn.Module):
    def __init__(self):
        super(Faster_r_cnn, self).__init__()
        self.model = []

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
