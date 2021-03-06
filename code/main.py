import argparse
import sys, os, time

from models.imagemodels import YOLOv3
from models.langmodels import Captioning, Dictionary
from utils import utils
from dataset import COCODataset

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader



def main(args):
    print(args)

    traintrans = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
        ])

    traindset = COCODataset(args.root_dir, args.ann_dir, mode="train", transform=traintrans)
    trainloader = DataLoader(traindset, batch_size=args.batch_size, shuffle=True, num_workers=args.ncpu, collate_fn=utils.collate_fn, drop_last=True)

    """
    create dictionary from args.captionfile
    """
    vocab = Dictionary()
    utils.create_vocfile(traindset, args.captionfile)
    before = time.time()
    vocab.create_vocab(args.captionfile)
    print('took {}s for vocab'.format(time.time()-before), flush=True)

    """
    models for training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo = YOLOv3()
    capmodel = Captioning(vocab_size=len(vocab))

    if args.multi_gpu:
        print('Using {} GPUs...'.format(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            capmodel = nn.DataParallel(capmodel)
            yolo = nn.DataParallel(yolo)
    if args.use_gpu:
        capmodel = capmodel.to(device)
        yolo = yolo.to(device)

    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()
    celoss = nn.CrossEntropyLoss()

    anchors = torch.tensor([[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]], dtype=torch.float).to(device)
    scaled_anchors = utils.scale_anchors(args.image_size, args.grid_size, anchors)

    print('begin training')

    for ep in range(args.epochs):
        for it, sample in enumerate(trainloader):
            # sample contains 'image', 'captions', 'bboxinfo'
            im_ten = sample['image'].to(device)
            info = sample['bboxinfo']
            """
            for objects in info:
                bounding_boxes = []
                classes = []
                for bbinfo in objects:
                    bounding_boxes.append(bbinfo['bbox'])
                    classes.append(bbinfo['obj_id'])
                bbs = torch.tensor(bounding_boxes)
                cls = torch.tensor(classes)
                print(bbs.size())
                print(cls.size())
            im_ten = im_ten.to(device)
            """

            x, y, w, h, pred_conf, pred_cls = yolo(im_ten)
            pred_x, pred_y, pred_w, pred_h = utils.offset_boxes(x, y, w, h, device, args.image_size, args.grid_size, scaled_anchors)
            break
        break

    print('done training')

    """ # size check of captioning model
    c = Captioning(2, 3, 4, 5, 6)
    i = torch.randint(2, (7, 11), dtype=torch.long)
    s = torch.randn((10, 4))
    out = c(s, i)
    print(out.size())
    """




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-s', '--image_size', type=int, default=256)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-gs', '--grid_size', type=int, default=7)
    parser.add_argument('-r', '--root_dir', type=str, default='../../../hdd/dsets/coco/')
    parser.add_argument('-a', '--ann_dir', type=str, default='annotations/')
    parser.add_argument('-c', '--captionfile', type=str, default='../data/caption.txt')
    parser.add_argument('-m', '--multi_gpu', type=bool, default=True)
    parser.add_argument('-g', '--use_gpu', type=bool, default=True)
    parser.add_argument('-n', '--ncpu', type=int, default=8)

    args = parser.parse_args()
    main(args)
