import torch
import torch.nn as nn
import time

# Bottleneck layer.
class Bottleneck(nn.Module):
    def __init__(self, C, repeat=1):
        super(Bottleneck, self).__init__()
        self.bottle_c = int(C//2)
        self.repeat = repeat
        self.model = nn.ModuleList()
        for i in range(self.repeat):
            self.model.append(nn.Sequential(
                    nn.Conv2d(C, self.bottle_c, 1, stride=1, padding=0),
                    nn.BatchNorm2d(self.bottle_c),
                    nn.LeakyReLU(),
                    nn.Conv2d(self.bottle_c, C, 3, stride=1, padding=1),
                    nn.BatchNorm2d(C),
                    nn.LeakyReLU())
                    )

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

# Downsampling layer.
class Down(nn.Module):
    def __init__(self, C):
        super(Down, self).__init__()
        self.down_c = C*2
        self.model = nn.Sequential(
                nn.Conv2d(C, self.down_c, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.down_c),
                nn.LeakyReLU()
                )

    def forward(self, x):
        return self.model(x)


"""
input : (bs, 3, imsize, imsize)
output : x1y1x2y2, conf, cls
"""
class YOLOv3(nn.Module):
    def __init__(self, in_c=3, classnum=80, S=7, B=3, imsize=256, resnumbers=[1, 2, 8, 8, 4]):
        super(YOLOv3, self).__init__()
        self.model = nn.ModuleList()
        self.current_c = 32
        self.S = S
        self.B = B
        self.classnum = classnum
        self.out_c = B*(4+1+classnum)
        self.repeat = resnumbers
        self.current_im = imsize

        self.model.append(nn.Conv2d(in_c, self.current_c, 3, stride=1, padding=1))
        for i in range(len(self.repeat)):
            self.model.append(Down(self.current_c))
            self.current_c *= 2
            self.current_im /= 2
            self.current_im = int(self.current_im)
            self.model.append(Bottleneck(self.current_c, repeat=self.repeat[i]))
        self.model.append(nn.AvgPool2d(self.current_im))
        self.model.append(nn.Linear(self.current_c, self.S*self.S*self.out_c))

    def forward(self, x):
        self.bs = x.size(0)
        for layer in self.model[:-1]:
            x = layer(x)
        x = x.squeeze()
        x = self.model[-1](x)
        # out : (bs, B, S, S, 5+classnum)
        out = x.view(self.bs, self.B, self.classnum+5, self.S, self.S).permute(0, 1, 3, 4, 2)
        # out_x, out_y, out_w, out_h : (bs, B, S, S, 1)
        # out_conf : (bs, B, S, S, 1)
        # out_cls : (bs, B, S, S, classnum)
        out_x, out_y, out_w, out_h, out_conf, out_cls = out.split([1, 1, 1, 1, 1, self.classnum], dim=-1)
        # sigmoid except for w and h
        out_x = torch.sigmoid(out_x)
        out_y = torch.sigmoid(out_y)
        out_conf = torch.sigmoid(out_conf)
        out_cls = torch.sigmoid(out_cls)
        return out_x, out_y, out_w, out_h, out_conf, out_cls


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


'''
for debugging.
'''
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv3()
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs...'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    before = time.time()
    for i in range(100):
        input = torch.empty((64, 3, 256, 256)).to(device)
        model = model.to(device)
        out = model(input)
    print('time per loop: {}s'.format((time.time()-before)/100))



