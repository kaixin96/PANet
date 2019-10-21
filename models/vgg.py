"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.features = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
        )

        self._init_weights()

    def forward(self, x):
        return self.features(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = dic[keys[i]]

            self.load_state_dict(new_dic)
