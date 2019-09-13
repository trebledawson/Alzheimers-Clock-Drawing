# ############################################################################# #
# mlp-mrmr.py                                                                   #
# Author: Glenn Dawson                                                          #
# --------------------                                                          #
# This is an experiment to apply a multi-layer perceptron to the Alzheimer's    #
# disease clock-drawing test dataset from RowanSOM. The CombinedV2Filter        #
# dataset is used. Features are restricted based on Sean McGuire's MRMR         #
# feature extraction.                                                           #
# ############################################################################# #

import os
import gc
import warnings
from datetime import date, datetime
from statistics import mean
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning
from imblearn.over_sampling import SMOTENC
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import utils

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_features = 100  # 0 < n_features <= 124
min_epochs = 50
net_size = 2010
plot_ = True
plot_compare_only = True
savefig_ = True
savedir = '.\Results\\' + str(date.today())

try:
    os.makedirs(savedir)
except FileExistsError:
    pass

opt = 'Adam'
if opt == 'Adam' or 'RMSprop':
    n_epochs = 300
    n_schedule = [25, 40, 75, 150, 225]
    n_early = 10
elif opt == 'SGD':
    n_epochs = 1000
    n_schedule = [500, 700, 800, 900, 950]
    n_early = 20
else:
    raise(ValueError, 'Invalid optimizer. Please choose "Adam", "RMSprop", '
                      'or "SGD".')

categorical_cols = utils.cat_cols

data_directory = '.\Data\CombinedV2Filter'
data_files = ['_12', '_13', '_14', '_23', '_24', '_34', '_123',
              '_234', '_1234', '']

def main():
    directory = \
        '.\Data\MRMR\FeaturesAndScoresMRMRAllFeaturesCombined125_'
    files = ['12Class', '13Class', '14Class', 'MCI_MCI', '24Class', '34Class',
             '123Class', '234Class', '4Class', '12345Class']
    performance = []
    train_sizes = []
    test_sizes = []
    for f, d in zip(files, data_files):
        if d in ['_12', '_13', '_14', '_23', '_24', '_34']:
            print('Dataset:', d[1:])
            p, t, s = train_model_2_class(directory, f, opt, d)
            performance.append(p)
            train_sizes.append(t)
            test_sizes.append(s)
        elif d in ['_123', '_234']:
            print('Dataset:', d[1:])
            p, t, s = train_model_multiclass(directory, f, 3, opt, d)
            performance.append(p)
            train_sizes.append(t)
            test_sizes.append(s)
        elif d == '_1234':
            print('Dataset:', d[1:])
            p, t, s = train_model_multiclass(directory, f, 4, opt, d)
            performance.append(p)
            train_sizes.append(t)
            test_sizes.append(s)
        else:
            print('Dataset: 12345')
            p, t, s = train_model_multiclass(directory, f, 5, opt, d)
            performance.append(p)
            train_sizes.append(t)
            test_sizes.append(s)
        print('\n')

        gc.collect()

    with open(savedir + '\mlp-mrmr-' + str(n_features) +
                      '-features\mlp-mrmr.txt', 'w') as log:
        print('mlp-mrmr experiments |', datetime.today(), file=log)
        print('n_features =', n_features, file=log)
        print('min_epochs =', min_epochs, file=log)
        print('net_size =', net_size, file=log)
        print('optimizer:', opt, file=log)
        print('---------------------------------------------', file=log)
        print('Number of features:', n_features, file=log)
        print('---------------------------------------------', file=log)
        for d, p, t, s in zip(data_files, performance, train_sizes, test_sizes):
            if d == '':
                print('Dataset: 12345', file=log)
            else:
                print('Dataset: ' + d[1:], file=log)

            print('Average of Last 10 Validation Accuracy (Original Dataset)    :',
                  p[0], '| Train / Test size:', t[0], '/', s[0], file=log)
            print('Average of Last 10 Validation Accuracy (SMOTE-NC Before TTS) :',
                  p[1], '| Train / Test size:', t[1], '/', s[1], file=log)
            print('Average of Last 10 Validation Accuracy (SMOTE-NC After TTS)  :',
                  p[2], '| Train / Test size:', t[2], '/', s[2], file=log)
            print('---------------------------------------------', file=log)

    plt.show()

    print('Done.')


def load_data(directory, f, d):
    feature_idx = pd.read_csv(directory + f + '.csv').iloc[:n_features, :]
    data = pd.read_csv(data_directory + d + '.csv')
    labels = data.iloc[:, 0]
    data = data.loc[:, feature_idx.iloc[:, 0]]

    cat_cols = sorted(list(
        set(feature_idx.iloc[:, 0].values.tolist()) & set(utils.cat_cols)))
    cat_cols_idx = sorted([data.columns.get_loc(c) for c in cat_cols])

    d_train, d_test, y_train, y_test = tts(data, labels,
                                           test_size=0.3)
    test_idx = list(d_test.index.values)

    print('Generating oversampled datasets...')
    # SMOTE-NC Before
    d_train_b, y_train_b = SMOTENC(
        categorical_features=cat_cols_idx,
        k_neighbors=5, random_state=42).fit_resample(data, labels)

    d_train_b = np.delete(d_train_b, test_idx, axis=0)
    y_train_b = np.delete(y_train_b, test_idx, axis=0)

    d_test_b = deepcopy(d_test)

    # SMOTE-NC After
    d_train_a, y_train_a = SMOTENC(
        categorical_features=cat_cols_idx,
        k_neighbors=5, random_state=42).fit_resample(d_train, y_train)

    d_test_a = deepcopy(d_test)

    # Scale numeric features only
    print('Scaling numeric features...')
    scaler = StandardScaler()
    for i in range(d_train.shape[1]):
        col = data.columns[i]
        if col in categorical_cols:
            continue

        # Original
        d_train[[col]] = scaler.fit_transform(d_train[[col]])
        d_test[[col]] = scaler.transform(d_test[[col]])

        # SMOTE-NC Before
        d_train_b[:, i] = np.ravel(scaler.fit_transform(
            d_train_b[:, i].reshape(-1, 1)))

        d_test_b[[col]] = scaler.transform(d_test_b[[col]])

        # SMOTE-NC After
        d_train_a[:, i] = np.ravel(scaler.fit_transform(
            d_train_a[:, i].reshape(-1, 1)))

        d_test_a[[col]] = scaler.transform(d_test_a[[col]])

    # Original
    train_ldr = td.DataLoader(utils.ClockDrawingDataset(d_train, y_train),
                              batch_size=10,
                              shuffle=True,
                              num_workers=0)
    test_ldr = td.DataLoader(utils.ClockDrawingDataset(d_test, y_test),
                             batch_size=10,
                             shuffle=False,
                             num_workers=0)

    # SMOTE-NC Before
    train_ldr_b = td.DataLoader(utils.ClockDrawingDataset(d_train_b,
                                                            y_train_b),
                                batch_size=10,
                                shuffle=True,
                                num_workers=0)
    test_ldr_b = td.DataLoader(utils.ClockDrawingDataset(d_test_b,
                                                           y_test),
                               batch_size=10,
                               shuffle=False,
                               num_workers=0)

    # SMOTE-NC After
    train_ldr_a = td.DataLoader(utils.ClockDrawingDataset(d_train_a,
                                                            y_train_a),
                                batch_size=10,
                                shuffle=True,
                                num_workers=0)
    test_ldr_a = td.DataLoader(utils.ClockDrawingDataset(d_test_a, y_test),
                               batch_size=10,
                               shuffle=False,
                               num_workers=0)

    return [train_ldr, train_ldr_b, train_ldr_a], \
           [test_ldr, test_ldr_b, test_ldr_a]


def train_model_2_class(directory, f, opt, d):
    print('Loading data...')
    train_ldrs, test_ldrs = load_data(directory, f, d)

    tr_loss = []
    tr_acc = []
    vl_loss = []
    vl_acc = []
    train_sizes = []
    test_sizes = []
    for train_ldr, test_ldr in zip(train_ldrs, test_ldrs):
        train_sizes.append(len(train_ldr.dataset))
        test_sizes.append(len(test_ldr.dataset))
        net = utils.SmallNet(2, n_features).to(device)
        net.size = net_size
        net.n_filters = net.size
        criterion = nn.BCEWithLogitsLoss()

        if opt == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=0.0005, nesterov=True)
        elif opt == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=0.001,
                                   betas=(0.9, 0.999), weight_decay=0.0005,
                                   amsgrad=False)
        elif opt == 'RMSprop':
            optimizer = optim.RMSprop(net.parameters(), lr=0.01,
                                      weight_decay=0.0005, momentum=0.9)
        else:
            raise ValueError('Invalid optimizer selected. Choose \'SGD\' or '
                             '\'Adam\'.')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=n_schedule,
                                                   gamma=0.1)

        print('Training...')
        print('Filters per layer:', net.n_filters)
        print('Criterion:', criterion)
        print('Optimizer:', opt)

        losses = [[], []]
        accs = [[], [0]]
        early_stopping = 0
        for epoch in range(n_epochs):
            # Training
            net.training = True
            train_correct = 0
            train_total = 0
            train_loss = 0.0
            for local_batch, local_labels in train_ldr:
                # Transfer to GPU
                local_batch = local_batch.to(device, dtype=torch.float)
                local_labels = local_labels.view(-1, 1).to(device,
                                                           dtype=torch.float)

                # Train
                optimizer.zero_grad()

                # Forward + backward + optimize
                logits = net(local_batch).view(-1, 1)
                loss = criterion(logits, local_labels)
                loss.backward()
                optimizer.step()

                # Tracking
                train_loss += loss.item()
                outputs = torch.sigmoid(logits)
                predicted = (outputs >= 0.5).view(-1).to(device,
                                                         dtype=torch.long)
                local_labels = local_labels.view(-1).to(device, dtype=torch.long)
                train_total += local_labels.size(0)
                train_correct += (predicted == local_labels).sum().item()

            train_acc = train_correct / train_total
            scheduler.step()

            # Validation
            net.training = False
            val_correct = 0
            val_total = 0
            val_loss = 0
            with torch.no_grad():
                for local_batch, local_labels in test_ldr:
                    # Transfer to GPU
                    local_batch = local_batch.to(device, dtype=torch.float)
                    local_labels = local_labels.view(-1, 1).to(device,
                                                               dtype=torch.float)

                    # Test
                    logits = net(local_batch).view(-1, 1)
                    loss = criterion(logits, local_labels)

                    # Tracking
                    val_loss += loss.item()
                    outputs = torch.sigmoid(logits)
                    predicted = (outputs >= 0.5).view(-1).to(device,
                                                             dtype=torch.long)
                    local_labels = local_labels.view(-1).to(device,
                                                            dtype=torch.long)
                    val_total += local_labels.size(0)
                    val_correct += (predicted == local_labels).sum().item()

            val_acc = val_correct / val_total

            losses[0].append(train_loss)
            losses[1].append(val_loss)
            accs[0].append(train_acc)
            accs[1].append(val_acc)

            if val_acc <= accs[1][-2]:
                early_stopping += 1
            elif early_stopping > 0:
                early_stopping -= 1

            early = False
            if early_stopping >= n_early and epoch > min_epochs:
                early = True

            if epoch % 10 == 9 or early:
                print('Epoch:', epoch + 1,
                      '| Train Acc:', round(train_acc, 8),
                      '| Train Loss:', round(train_loss, 8),
                      '| Val Acc:', round(val_acc, 8),
                      '| Val Loss:', round(val_loss, 8),
                      '| Early:', early_stopping)

            if early:
                print('Early stopping.')
                break

        accs[1] = accs[1][1:]

        tr_loss.append(losses[0])
        tr_acc.append(accs[0])
        vl_loss.append(losses[1])
        vl_acc.append(accs[1])

    best = [mean(a[-10:]) for a in vl_acc]
    if plot_:
        # Plot loss and accuracy
        savedir_ = savedir + '\mlp-mrmr-' + str(n_features) + '-features\\' + \
                   d[1:] + '\\'
        plot(savedir_, d, tr_loss, tr_acc, vl_loss, vl_acc, best)

    return best, train_sizes, test_sizes


def train_model_multiclass(directory, f, n_classes, opt, d):
    print('Loading data...')
    train_ldrs, test_ldrs = load_data(directory, f, d)

    tr_loss = []
    tr_acc = []
    vl_loss = []
    vl_acc = []
    train_sizes = []
    test_sizes = []
    for train_ldr, test_ldr in zip(train_ldrs, test_ldrs):
        train_sizes.append(len(train_ldr.dataset))
        test_sizes.append(len(test_ldr.dataset))
        net = utils.SmallNet(n_classes, n_features).to(device)
        net.size = net_size
        net.n_filters = net.size
        criterion = nn.CrossEntropyLoss()

        if opt == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=0.0005, nesterov=True)
        elif opt == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=0.001,
                                   betas=(0.9, 0.999), weight_decay=0.0005,
                                   amsgrad=False)
        elif opt == 'RMSprop':
            optimizer = optim.RMSprop(net.parameters(), lr=0.01,
                                      weight_decay=0.0005, momentum=0.9)
        else:
            raise ValueError('Invalid optimizer selected. Choose \'SGD\' or '
                             '\'Adam\'.')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=n_schedule,
                                                   gamma=0.1)

        print('Training...')
        print('Filters per layer:', net.n_filters)
        print('Criterion:', criterion)
        print('Optimizer:', opt)

        losses = [[], []]
        accs = [[], [0]]
        early_stopping = 0
        for epoch in range(n_epochs):
            # Training
            net.training = True
            train_correct = 0
            train_total = 0
            train_loss = 0.0
            for local_batch, local_labels in train_ldr:
                # Transfer to GPU
                local_batch = local_batch.to(device, dtype=torch.float)
                local_labels = local_labels.view(-1).to(device, dtype=torch.long)

                # Train
                optimizer.zero_grad()

                # Forward + backward + optimize
                logits = net(local_batch).view(-1, n_classes)
                loss = criterion(logits, local_labels)
                loss.backward()
                optimizer.step()

                # Tracking
                train_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                train_total += local_labels.size(0)
                train_correct += (predicted == local_labels).sum().item()

            train_acc = train_correct / train_total
            scheduler.step()

            # Validation
            net.training = False
            val_correct = 0
            val_total = 0
            val_loss = 0
            with torch.no_grad():
                for local_batch, local_labels in test_ldr:
                    # Transfer to GPU
                    local_batch = local_batch.to(device, dtype=torch.float)
                    local_labels = local_labels.to(device)

                    # Test
                    logits = net(local_batch).view(-1, n_classes)
                    loss = criterion(logits, local_labels)

                    # Tracking
                    val_loss += loss.item()
                    predicted = torch.argmax(logits, dim=1)
                    val_total += local_labels.size(0)
                    val_correct += (predicted == local_labels).sum().item()

            val_acc = val_correct / val_total

            losses[0].append(train_loss)
            losses[1].append(val_loss)
            accs[0].append(train_acc)
            accs[1].append(val_acc)

            if val_acc <= accs[1][-2]:
                early_stopping += 1
            elif early_stopping > 0:
                early_stopping -= 1

            early = False
            if early_stopping >= n_early and epoch > min_epochs:
                early = True

            if epoch % 10 == 9 or early:
                print('Epoch:', epoch + 1,
                      '| Train Acc:', round(train_acc, 8),
                      '| Train Loss:', round(train_loss, 8),
                      '| Val Acc:', round(val_acc, 8),
                      '| Val Loss:', round(val_loss, 8),
                      '| Early:', early_stopping)

            if early:
                print('Early stopping.')
                break

        accs[1] = accs[1][1:]

        tr_loss.append(losses[0])
        tr_acc.append(accs[0])
        vl_loss.append(losses[1])
        vl_acc.append(accs[1])

    best = [mean(a[-10:]) for a in vl_acc]
    if plot_:
        if d == '':
            d = '_12345'
        # Plot loss and accuracy
        savedir_ = savedir + '\mlp-mrmr-' + str(n_features) \
                   + '-features\\' + d[1:] + '\\'
        plot(savedir_, d, tr_loss, tr_acc, vl_loss, vl_acc, best)

    return best, train_sizes, test_sizes

def plot(savedir_, d, tr_loss, tr_acc, vl_loss, vl_acc, best):
    try:
        os.makedirs(savedir_)
    except FileExistsError:
        pass

    plt.figure()

    plt.subplot(221)
    plt.title('Training Loss')
    plt.plot(tr_loss[0], label='Original')
    plt.plot(tr_loss[1], '--', label='SMOTE-NC Before TTS')
    plt.plot(tr_loss[2], '--', label='SMOTE-NC After TTS')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(222)
    plt.title('Training Accuracy')
    plt.plot(tr_acc[0], label='Original')
    plt.plot(tr_acc[1], '--', label='SMOTE-NC Before TTS')
    plt.plot(tr_acc[2], '--', label='SMOTE-NC After TTS')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.subplot(223)
    plt.title('Validation Loss')
    plt.plot(vl_loss[0], label='Original')
    plt.plot(vl_loss[1], '--', label='SMOTE-NC Before TTS')
    plt.plot(vl_loss[2], '--', label='SMOTE-NC After TTS')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(224)
    plt.title('Validation Accuracy')
    plt.plot(vl_acc[0], label='Original')
    plt.plot(vl_acc[1], '--', label='SMOTE-NC Before TTS')
    plt.plot(vl_acc[2], '--', label='SMOTE-NC After TTS')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.suptitle('MLP Training History on ' + d[1:] + ' Dataset')

    if savefig_:
        plt.savefig(fname=(savedir_ + '0-compare.pdf'),
                    format='pdf',
                    orientation='landscape')

    if not plot_compare_only:
        plt.figure()

        plt.subplot(121)
        plt.title('Original Loss')
        plt.plot(tr_loss[0], label='Training')
        plt.plot(vl_loss[0], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        plt.subplot(122)
        plt.title('Original Accuracy')
        plt.plot(tr_acc[0], label='Training')
        plt.plot(vl_acc[0], label='Validation')
        plt.axhline(y=best[0], xmin=0, xmax=len(vl_acc[0]), linewidth=0.5, color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.suptitle('MLP Training History on ' + d[1:] + ' Dataset, '
                                                                 'Original')

        if savefig_:
            plt.savefig(fname=(savedir_ + '1-original.pdf'),
                        format='pdf',
                        orientation='landscape')

        plt.figure()

        plt.subplot(121)
        plt.title('SMOTE-NC (Before TTS) Loss')
        plt.plot(tr_loss[1], label='Training')
        plt.plot(vl_loss[1], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        plt.subplot(122)
        plt.title('SMOTE-NC (Before TTS) Accuracy')
        plt.plot(tr_acc[1], label='Training')
        plt.plot(vl_acc[1], label='Validation')
        plt.axhline(y=best[1], xmin=0, xmax=len(vl_acc[1]), linewidth=0.5, color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.suptitle('MLP Training History on ' + d[1:] + ' Dataset, '
                                                                 'SMOTE-NC Before '
                                                                 'TTS')

        if savefig_:
            plt.savefig(fname=(savedir_ + '2-smote-nc-before.pdf'),
                        format='pdf',
                        orientation='landscape')

        plt.figure()

        plt.subplot(121)
        plt.title('SMOTE-NC (After TTS) Loss')
        plt.plot(tr_loss[2], label='Training')
        plt.plot(vl_loss[2], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()

        plt.subplot(122)
        plt.title('SMOTE-NC (After TTS) Accuracy')
        plt.plot(tr_acc[2], label='Training')
        plt.plot(vl_acc[2], label='Validation')
        plt.axhline(y=best[2], xmin=0, xmax=len(vl_acc[2]), linewidth=0.5, color='r')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend()

        plt.suptitle('MLP Training History on ' + d[1:] + ' Dataset, '
                                                                 'SMOTE-NC After '
                                                                 'TTS')

        if savefig_:
            plt.savefig(fname=(savedir_ + '3-smote-nc-after.pdf'),
                        format='pdf',
                        orientation='landscape')



if __name__ == '__main__':
    main()
