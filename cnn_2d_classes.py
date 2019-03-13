# ############################################################################ #
# cnn_2d_classes.py                                                            #
# Author: Glenn Dawson                                                         #
# --------------------                                                         #
# Classes and other building blocks for cnn_2d.py                              #
# ############################################################################ #

import numpy as np
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms as transforms

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
        
        self.n_filters = 8
        self.inflow = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=(3 * self.n_filters),
                                              kernel_size=(3, 5),
                                              stride=1,
                                              padding=(1, 2),
                                              dilation=1,
                                              groups=1,
                                              bias=True),
                                    nn.BatchNorm2d(3 * self.n_filters),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=(3 * self.n_filters),
                                              out_channels=self.n_filters,
                                              kernel_size=(3, 5),
                                              stride=1,
                                              padding=(1, 2),
                                              dilation=1,
                                              groups=1,
                                              bias=True),
                                    nn.BatchNorm2d(self.n_filters),
                                    nn.ReLU()
                                    )
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
        self.conv1 = nn.Sequential(nn.Conv2d(self.n_filters, 2, 1, padding=0),
                                   nn.BatchNorm2d(2),
                                   nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(),
                                nn.Linear(176, self.output_size))

    def forward(self, x):
        x = self.inflow(x)
        x = self.skip1(self.block1(x) + x)
        x = self.skip2(self.block2(x) + x)
        #x = self.skip3(self.block3(x) + x)
        #x = self.skip4(self.block4(x) + x)
        #x = self.skip5(self.block5(x) + x)
        x = self.conv1(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
