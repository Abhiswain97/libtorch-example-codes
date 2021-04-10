from torchvision.models import resnet18
import torch.nn as nn
import torch

# 224 is the least input size, depends on the dataset you use
resnet18 = resnet18(pretrained=True)
resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7))
resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
example_input = torch.randn(1, 1, 224, 224)

script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet18.pt')
