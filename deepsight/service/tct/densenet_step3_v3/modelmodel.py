import torch
import torch.nn as nn
#from configs.config import *

from torchvision import models

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.model = models.alexnet(pretrained=True)
        #self.path = ABSOLUTE_PATH+'alexnet/alexnet-owt-4df8aa71.pth'
        self.path = '/opt/test/tct/alexnet/alexnet-owt-4df8aa71.pth'
        self.model = models.alexnet(pretrained=False)
        self.model.load_state_dict(torch.load(self.path))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        y = self.classifier(x)
        return y, x
