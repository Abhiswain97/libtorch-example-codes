import torch
import torch.nn as nn
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self, transform=None):
        super(Net, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size
        )
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)


net = Net()

print(net)
