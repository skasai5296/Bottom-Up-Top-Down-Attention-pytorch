import sys, os, time
import json
import pickle
from skimage import io
from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision

from pycocotools.coco import COCO




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


"""
coco dataset.
root_dir :  root directory of dataset
ann_dir :   relative directory of annotations
mode :      train, val, or test
transform : contain transformations for images (torchvision.Transforms)

returns {'image' : PIL Image, 'caption' : string, 'bboxinfo' : [{'image_id' : integer index of image, 'obj_id' : integer index of object, 'bbox' : list containing xywh of bounding box}, ...] }
"""
class COCODataset(Dataset):
    def __init__(self, root_dir="../../dsets/coco/", ann_dir="annotations/", mode="train", transform=None):
        mod = mode + "2014"
        self.imgdir = os.path.join(root_dir, mod)
        anndir = os.path.join(root_dir, ann_dir, "captions_{}.json".format(mod))
        self.api = COCO(anndir)
        self.imgids = self.api.getImgIds()
        self.transform = transform
        bboxann = os.path.join(root_dir, ann_dir, "instances_{}.json".format(mod))
        self.bboxes = []
        self.idx2obj = {}
        self.idx2supercats = {}
        with open(bboxann, "r") as f:
            self.ann = json.load(f)
        for ann in self.ann['annotations']:
            info = {'image_id' : ann['image_id'], 'obj_id' : ann['category_id'], 'bbox' : ann['bbox']}
            self.bboxes.append(info)
        for cat in self.ann['categories']:
            self.idx2obj[cat['id']] = cat['name']
            self.idx2supercats[cat['id']] = cat['supercategory']

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        info = []
        for i in self.bboxes:
            if i['image_id'] == imgid:
                info.append(i)
        img = self.api.loadImgs(imgid)[0]
        impath = os.path.join(self.imgdir, os.path.basename(img["coco_url"]))
        image = Image.open(impath)
        annid = self.api.getAnnIds(imgid)
        anns = self.api.loadAnns(annid)[0]["caption"]
        if self.transform is not None:
            image = self.transform(image)
        sample = {'image' : image, 'caption' : anns, 'bboxinfo' : info}
        return sample


if __name__ == '__main__':
    c = COCODataset()
    print(c[0])



