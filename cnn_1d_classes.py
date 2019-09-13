# ############################################################################ #
# cnn_1d_classes.py                                                            #
# Author: Glenn Dawson                                                         #
# --------------------                                                         #
# Classes and other building blocks for cnn_1d.py                              #
# ############################################################################ #

import numpy as np
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        if n_classes == 2:
            self.output_size = 1
        else:
            self.output_size = n_classes
        
        self.n_filters = 32
        self.inflow = nn.Sequential(nn.Conv1d(in_channels=1,
                                              out_channels=self.n_filters,
                                              kernel_size=5,
                                              stride=1,
                                              padding=2,
                                              dilation=1,
                                              groups=1,
                                              bias=True),
                                    nn.BatchNorm1d(self.n_filters),
                                    nn.ReLU())
        self.block1 = nn.Sequential(nn.Conv1d(self.n_filters, self.n_filters, 5,
                                              padding=2),
                                    nn.BatchNorm1d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv1d(self.n_filters, self.n_filters, 5,
                                              padding=2),
                                    nn.BatchNorm1d(self.n_filters))
        self.skip1 = nn.Sequential(nn.ReLU(),
                                   nn.Conv1d(self.n_filters, self.n_filters, 3,
                                             stride=2, padding=1),
                                   nn.BatchNorm1d(self.n_filters),
                                   nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters))
        self.skip2 = nn.Sequential(nn.ReLU(),
                                   nn.Conv1d(self.n_filters, self.n_filters, 3,
                                             stride=2, padding=1),
                                   nn.BatchNorm1d(self.n_filters),
                                   nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters))
        self.skip3 = nn.Sequential(nn.ReLU(),
                                   nn.Conv1d(self.n_filters, self.n_filters, 3,
                                             stride=2, padding=1),
                                   nn.BatchNorm1d(self.n_filters),
                                   nn.ReLU())
        self.block4 = nn.Sequential(nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters))
        self.skip4 = nn.Sequential(nn.ReLU(),
                                   nn.Conv1d(self.n_filters, self.n_filters, 3,
                                             stride=2, padding=1),
                                   nn.BatchNorm1d(self.n_filters),
                                   nn.ReLU())
        self.block5 = nn.Sequential(nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv1d(self.n_filters, self.n_filters, 3,
                                              padding=1),
                                    nn.BatchNorm1d(self.n_filters))
        self.skip5 = nn.Sequential(nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv1d(self.n_filters, 2, 1, padding=0),
                                   nn.BatchNorm1d(2),
                                   nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(),
                                nn.Linear(44, self.output_size))


    def forward(self, x):
        x = self.inflow(x)
        x = self.skip1(self.block1(x) + x)
        x = self.skip2(self.block2(x) + x)
        x = self.skip3(self.block3(x) + x)
        x = self.skip4(self.block4(x) + x)
        x = self.skip5(self.block5(x) + x)
        x = self.conv1(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=5, zero_init_residual=False):
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv1d(self.inplanes,
                                                 planes * block.expansion,
                                                 stride),
                                       nn.BatchNorm1d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return self.relu(out)


class ClockDrawingDataset(td.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = np.asarray(labels)

        labels = np.unique(self.labels)
        for i in range(len(labels)):
            self.labels[self.labels == labels[i]] = i

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        #features = self.transform(self.data[index, :])
        features = self.data[index, :].reshape(1, -1)
        label = self.labels[index]

        return features, label