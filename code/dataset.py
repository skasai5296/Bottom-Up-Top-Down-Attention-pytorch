import sys, os, time
import json
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision


class Dictionary():
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word


class VisualGenomeDataset(Dataset):
    def __init__(self, img_path, img_data_file, bbox_file, transform=None):
        self.rootdir = img_path
        before = time.time()
        for filename in os.walk(self.rootdir)[2]:
            self.img_paths.append(filename)
        with open(img_data_file, "r") as f:
            self.imdata_list = json.load(f)
        elapsed = time.time() - before
        print("done loading image data JSON file, took {}s".format(elapsed))
        with open(bbox_file, "r") as f:
            self.bbox_list = json.load(f)
        elapsed = time.time() - before
        print("done loading bounding box + object data JSON file, took {}s".format(elapsed))

    """
    return dict for mapping image filenames to ids
    """
    def get_pth2id(self):
        self.pth2id = {}
        self.id2pth = {}
        for obj in self.img_data_file:
            id = obj["image_id"]
            url = obj["url"]
            _, filename = os.path.split(url)
            self.pth2id[filename] = id
            self.id2pth[id] = filename
        return self.pth2id

    """
    return dict for mapping image ids to bounding box information and object information
    dict has values of {"xywh" : np.array of ints, "obj_name" : string, "attr_name" : list of strings}
    """
    def get_id2regions(self):
        self.id2regions = {}
        for obj in self.region_list:
            id = obj["image_id"]
            regions = obj["regions"]
            self.xywhs = []
            self.objs = []
            self.attrs = []
            for bbox in regions:
                self.xywhs.append(np.array([bbox["x"], bbox["y"], bbox["width"], bbox["height"]]))
                self.objs.append(bbox["name"])
                self.attrs.append(bbox["attributes"])
            self.id2regions[id] = {"xywh" : self.xywhs, "obj_name" : self.objs, "attr_name" : self.attrs}
        return self.id2regions

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        im_relpth = self.img_paths[idx]
        im_abspth = os.path.join(self.rootdir, im_relpth)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform[image]
        im_id = self.pth2id[im_relpth]
        im_reginfo = self.id2regions[im_id]
        return {"image" : image, "reg" : im_reginfo}








