import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms


from ..models.langmodels import Captioning
from ..dataset import COCODataset

class Solver():
    def create_vocab(loader):
        for i in loader:
            i

    def detect(image):



    def train(args):
        capmodel = Captioning(args.vocab_size, args.embedding_dim, args.image_ft_dim, args.lstm_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for ep in range(args.num_epoch):
            for i, it in enumerate(dataloader):

