import torch
import torch.nn as nn

from model import UNet, CombinedLoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pass

if __name__ == '__main__':
    main()