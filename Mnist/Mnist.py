import torch
import torch.nn as nn
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self, transform=None):
        super(Net, self).__init__()
        model = resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size)
