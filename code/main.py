import argparse
from models.langmodels import Captioning

import torch


def main(args):
    print(args)


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
    args = parser.parse_args()
    main(args)
