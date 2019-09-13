# ############################################################################ #
# utils.py                                                                     #
# Author: Glenn Dawson                                                         #
# --------------------                                                         #
# Classes and other building blocks for cnn_2d.py                              #
# ############################################################################ #

import numpy as np
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms as transforms

# ########################### #
# List of categorical columns #
# ########################### #
cat_cols = ['comm_DwgTotStrokes',
            'copy_DwgTotStrokes_A',
            'ClockFace1TotStrokes',
            'ClockFace1TotStrokes_A',
            'CFNonCFTerminator',
            'CFNonCFNoNoiseTerminator',
            '7.comm_HH1TotStrokes',
            '7.copy_HH1TotStrokes',
            '12.comm_MH1TotStrokes',
            '12.copy_MH1TotStrokes',
            'Digit1Strokes',
            'Digit1Strokes_A',
            'Digit2Strokes',
            'Digit2Strokes_A',
            'Digit3Strokes',
            'Digit3Strokes_A',
            'Digit4Strokes',
            'Digit4Strokes_A',
            'Digit5Strokes',
            'Digit5Strokes_A',
            'Digit6Strokes',
            'Digit6Strokes_A',
            'Digit7Strokes',
            'Digit7Strokes_A',
            'Digit8Strokes',
            'Digit8Strokes_A',
            'Digit9Strokes',
            'Digit9Strokes_A',
            'Digit10Strokes',
            'Digit10Strokes_A',
            'Digit11Strokes',
            'Digit11Strokes_A',
            'Digit12Strokes',
            'Digit12Strokes_A',
            'Digit1Outside',
            'Digit1Outside_A',
            'Digit2Outside',
            'Digit2Outside_A',
            'Digit3Outside',
            'Digit3Outside_A',
            'Digit4Outside',
            'Digit4Outside_A',
            'Digit5Outside',
            'Digit5Outside_A',
            'Digit6Outside',
            'Digit6Outside_A',
            'Digit7Outside',
            'Digit7Outside_A',
            'Digit8Outside',
            'Digit8Outside_A',
            'Digit9Outside',
            'Digit9Outside_A',
            'Digit10Outside',
            'Digit10Outside_A',
            'Digit11Outside',
            'Digit11Outside_A',
            'Digit12Outside',
            'Digit12Outside_A',
            'CenterDotStrokeNum',
            'CenterDotStrokeNum_A',
            'PreFirstHandNoNoiseInitiator',
            'PreFirstHandNoNoiseInitiator_A',
            'PreSecondHandNoNoiseInitiator',
            'PreSecondHandNoNoiseInitiator_A',
            'InterDigitIntervalCount',
            'InterDigitIntervalCount_A',
            'AnchorLatencyIntervalCount',
            'AnchorLatencyIntervalCount_A']

# ######################### #
# 2D Convolutional NN Utils #
# ######################### #

class ClockDrawingDataset2D(td.Dataset):
    def __init__(self, data, labels):
        data = np.asarray(data)
        d1 = data[:, :175]
        d2 = data[:, 175:]
        self.data = np.zeros((data.shape[0], 2, 175))
        for i in range(data.shape[0]):
            self.data[i, :] = np.vstack((d1[i, :],
                                         d2[i, :]))
        self.labels = np.asarray(labels)

        labels = np.unique(self.labels)
        for i in range(len(labels)):
            self.labels[self.labels == labels[i]] = i

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # features = self.transform(self.data[index, :])
        features = self.data[index, :].reshape(1, 2, -1)
        label = self.labels[index]

        return features, label


class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        if n_classes == 2:
            self.output_size = 1
        else:
            self.output_size = n_classes

        self.n_filters = 1
        in_scale = 3

        self.inflow_ = nn.Sequential(nn.Conv2d(in_channels=1,
                                               out_channels=(in_scale *
                                                             self.n_filters),
                                               kernel_size=(3, 5),
                                               stride=1,
                                               padding=(1, 2),
                                               dilation=1,
                                               groups=1,
                                               bias=True),
                                     nn.BatchNorm2d(in_scale * self.n_filters),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=(in_scale *
                                                            self.n_filters),
                                               out_channels=self.n_filters,
                                               kernel_size=(3, 5),
                                               stride=(1, 2),
                                               padding=(1, 2),
                                               dilation=1,
                                               groups=1,
                                              bias=True),
                                     nn.BatchNorm2d(self.n_filters),
                                     nn.ReLU()
                                     )
        self.inflow = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=(in_scale *
                                                            self.n_filters),
                                              kernel_size=(3, 5),
                                              stride=1,
                                              padding=(1, 2),
                                              dilation=1,
                                              groups=1,
                                              bias=True),
                                    nn.BatchNorm2d(in_scale * self.n_filters),
                                    nn.ReLU())
        self.block1 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 5), padding=(1, 2)),
                                    nn.BatchNorm2d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 5), padding=(1, 2)),
                                    nn.BatchNorm2d(self.n_filters))
        self.skip1 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(self.n_filters, self.n_filters,
                                             (3, 5), stride=(1, 2),
                                             padding=(1, 2)),
                                   nn.BatchNorm2d(self.n_filters),
                                   nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters))
        self.skip2 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(self.n_filters, self.n_filters,
                                             (3, 3), stride=(1, 2),
                                             padding=(1, 1)),
                                   nn.BatchNorm2d(self.n_filters),
                                   nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters))
        self.skip3 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(self.n_filters, self.n_filters,
                                             (3, 3), stride=(1, 2),
                                             padding=(1, 1)),
                                   nn.BatchNorm2d(self.n_filters),
                                   nn.ReLU())
        self.block4 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters))
        self.skip4 = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(self.n_filters, self.n_filters,
                                             (3, 3), stride=(1, 2),
                                             padding=(1, 1)),
                                   nn.BatchNorm2d(self.n_filters),
                                   nn.ReLU())
        self.block5 = nn.Sequential(nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(self.n_filters, self.n_filters,
                                              (3, 3), padding=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters))
        self.skip5 = nn.Sequential(nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(in_scale * self.n_filters,
                                             1, 1,
                                             padding=0),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU())
        n_hidden = 5
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(350, n_hidden),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(n_hidden, self.output_size))

    def forward(self, x):
        x = self.inflow(x)
        #x = self.skip1(self.block1(x) + x)
        #x = self.skip2(self.block2(x) + x)
        # x = self.skip3(self.block3(x) + x)
        # x = self.skip4(self.block4(x) + x)
        # x = self.skip5(self.block5(x) + x)
        x = self.conv1(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# ######### #
# MLP Utils #
# ######### #

class ClockDrawingDataset(td.Dataset):
    def __init__(self, data, labels):
        self.data = np.asarray(data)
        self.labels = np.asarray(labels)

        labels = np.unique(self.labels)
        for i in range(len(labels)):
            self.labels[self.labels == labels[i]] = i

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # features = self.transform(self.data[index, :])
        features = self.data[index, :].reshape(1, -1)
        label = self.labels[index]

        return features, label

class SmallNet(nn.Module):
    def __init__(self, n_classes, input_size=352):
        super(SmallNet, self).__init__()
        if n_classes == 2:
            self.output_size = 1
        else:
            self.output_size = n_classes

        self.input_size = input_size
        self.size = 1010
        self.n_filters = self.size

        self.fc1010 = nn.Sequential(nn.Linear(self.input_size, 10),
                                nn.Sigmoid(),
                                nn.Linear(10, 10),
                                nn.Sigmoid(),
                                nn.Linear(10, self.output_size))

        self.fc2010 = nn.Sequential(nn.Linear(self.input_size, 20),
                                    nn.Sigmoid(),
                                    nn.Linear(20, 10),
                                    nn.Sigmoid(),
                                    nn.Linear(10, self.output_size))

        self.fc20 = nn.Sequential(nn.Linear(self.input_size, 20),
                                   nn.Sigmoid(),
                                   nn.Linear(20, self.output_size))

        self.fc25 = nn.Sequential(nn.Linear(self.input_size, 25),
                                  nn.Sigmoid(),
                                  nn.Linear(25, self.output_size))

        self.fc10 = nn.Sequential(nn.Linear(self.input_size, 10),
                                  nn.Sigmoid(),
                                  nn.Linear(10, self.output_size))

        self.fc = nn.Sequential(nn.Linear(self.input_size, self.size),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(self.size, self.output_size))

    def forward(self, x):
        if self.size == 1010:
            return self.fc1010(x)
        elif self.size == 20:
            return self.fc20(x)
        elif self.size == 25:
            return self.fc25(x)
        elif self.size == 10:
            return self.fc10(x)
        elif self.size == 2010:
            return self.fc2010(x)
        else:
            return self.fc(x)
