from torchvision.models import resnet50
import torch

model = resnet50(pretrained=True)
# 224 is the least input size, depends on the dataset you use
example_input = torch.randn(1, 3, 224, 224)

script_module = torch.jit.trace(model, example_input)
script_module.save('resnet50.pt')
