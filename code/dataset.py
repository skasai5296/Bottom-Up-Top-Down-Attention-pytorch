import sys, os
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
