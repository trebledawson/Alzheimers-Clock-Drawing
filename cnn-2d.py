# ############################################################################# #
# cnn-2d.py                                                                     #
# Author: Glenn Dawson                                                          #
# --------------------                                                          #
# This is an experiment to apply a two-dimensional convolutional neural network #
# to the Alzheimer's disease clock-drawing test dataset from RowanSOM. The      #
# CombinedV2Filter dataset is used, with each patient's Command and Copy data   #
# being vertically concatenated. In order to facilitate this concatenation, two #
# features from the Command dataset are eliminated: The                         #
# "comm_ClockFaceNonClockFaceNoNoiseLatency" feature and the                    #
# "CFNonCFNoNoiseTerminator" feature. The concatenated features are then passed #
# through a two-dimensional ResNet.                                             #
# ############################################################################# #

import gc
import warnings
from collections import Counter
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
import cnn_2d_classes

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.mode.chained_assignment = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

smote_ = True
opt = 'Adam'
if opt == 'Adam':
    n_epochs = 50
    n_schedule = [25, 40]
    n_early = 10
elif opt == 'SGD':
    n_epochs = 100
    n_schedule = [30, 60, 80, 90, 95]
    n_early = 10

def main():
    directory = '.\Data\CombinedV2Filter'
    files = ['_12', '_13', '_14', '_23', '_24', '_34', '_123', '_234',
             '_1234', '']
    performance = []

    for f in files:
        if f in ['_12', '_13', '_14', '_23', '_24', '_34']:
            print('Dataset:', f[1:])
            performance.append(train_model_2_class(directory, f, opt))
        elif f in ['_123', '_234']:
            print('Dataset:', f[1:])
            performance.append(train_model_multiclass(directory, f, 3, opt))
        elif f == '_1234':
            print('Dataset:', f[1:])
            performance.append(train_model_multiclass(directory, f, 4, opt))
        else:
            print('Dataset: 12345')
            performance.append(train_model_multiclass(directory, f, 5, opt))
        print('\n')

        gc.collect()

    for f, p in zip(files, performance):
        if f == '':
            print('Best performance on 12345 :', p)
        else:
            print('Best performance on', f[1:], ':', p)

    plt.show()

def load_data(filepath):
    data = pd.read_csv(filepath)
    labels = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    data = data.drop(['comm_ClockFaceNonClockFaceNoNoiseLatency',
                      'CFNonCFNoNoiseTerminator'], axis=1)
    categorical_cols = ['comm_DwgTotStrokes',
                        'copy_DwgTotStrokes_A',
                        'ClockFace1TotStrokes',
                        'ClockFace1TotStrokes_A',
                        'CFNonCFTerminator',
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
    cat_cols_idx = sorted([data.columns.get_loc(c) for c in categorical_cols])

    d_train, d_test, y_train, y_test = tts(data, labels,
                                           test_size=0.3,
                                           random_state=42)
    if smote_:
        print('Oversampling with SMOTE_NC...')
        d_train, y_train = SMOTENC(categorical_features=cat_cols_idx,
                                   k_neighbors=5,
                                   random_state=42).fit_resample(data, labels)

    print('Training set size:', Counter(y_train),
          '| Testing set size:', Counter(y_test))

    # Scale numeric features only
    print('Scaling numeric features...')
    scaler = StandardScaler()
    for i in range(d_train.shape[1]):
        col = data.columns[i]
        if col in categorical_cols:
            continue

        if smote_:
            d_train[:, i] = np.ravel(scaler.fit_transform(
                d_train[:, i].reshape(-1, 1)))
        else:
            d_train[[col]] = scaler.fit_transform(d_train[[col]])

        d_test[[col]] = scaler.transform(d_test[[col]])

    train_ldr = td.DataLoader(cnn_2d_classes.ClockDrawingDataset2D(d_train,
                                                                   y_train),
                              batch_size=10,
                              shuffle=True,
                              num_workers=1)
    test_ldr = td.DataLoader(cnn_2d_classes.ClockDrawingDataset2D(d_test,
                                                                  y_test),
                             batch_size=10,
                             shuffle=False,
                             num_workers=1)
    return train_ldr, test_ldr

def train_model_2_class(directory, f, opt):
    filepath = directory + f
    # net = cnn_2d_classes.ResNet1D(cnn_2d_classes.ResidualBlock,
    #                              [3, 8, 24, 3]).to(device)
    net = cnn_2d_classes.Net(2).to(device)
    criterion = nn.BCEWithLogitsLoss()

    if opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)
    elif opt == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=0.001,
                               betas=(0.9, 0.999), weight_decay=0.0005,
                               amsgrad=False)
    else:
        raise ValueError('Invalid optimizer selected. Choose \'SGD\' or '
                         '\'Adam\'.')
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=n_schedule,
                                               gamma=0.1)
    print('Loading data...')
    train_ldr, test_ldr = load_data(filepath + '.csv')

    print('Training...')
    print('Filters per layer:', net.n_filters)
    print('Criterion:', criterion)
    print(optimizer)

    losses = [[], [100]]
    accs = [[], []]
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
            local_labels = local_labels.view(-1, 1).to(device, dtype=torch.float)

            # Train
            optimizer.zero_grad()

            # Forward + backward + optimize
            logits = net(local_batch)
            loss = criterion(logits, local_labels)
            loss.backward()
            optimizer.step()

            # Tracking
            train_loss += loss.item()
            outputs = torch.sigmoid(logits)
            predicted = (outputs >= 0.5).view(-1).to(device, dtype=torch.long)
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
                logits = net(local_batch)
                loss = criterion(logits, local_labels)

                # Tracking
                val_loss += loss.item()
                outputs = torch.sigmoid(logits)
                predicted = (outputs >= 0.5).view(-1).to(device,
                                                         dtype=torch.long)
                local_labels = local_labels.view(-1).to(device, dtype=torch.long)
                val_total += local_labels.size(0)
                val_correct += (predicted == local_labels).sum().item()

        val_acc = val_correct / val_total

        losses[0].append(train_loss)
        losses[1].append(val_loss)
        accs[0].append(train_acc)
        accs[1].append(val_acc)

        if val_loss >= losses[1][-2]:
            early_stopping += 1
        elif early_stopping > 0:
            early_stopping -= 1

        early = False
        if early_stopping == n_early:
            early = True

        print('Epoch:', epoch + 1,
              '| Train Acc:', round(train_acc, 8),
              '| Train Loss:', round(train_loss, 8),
              '| Val Acc:', round(val_acc, 8),
              '| Val Loss:', round(val_loss, 8),
              '| Early:', early_stopping)

        if early:
            print('Early stopping.')
            break

    losses[1] = losses[1][1:]

    # Plot loss and accuracy
    plt.figure()

    plt.subplot(121)
    plt.plot(losses[0], label='Training')
    plt.plot(losses[1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(accs[0], label='Training')
    plt.plot(accs[1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.suptitle('2D ConvNet Training History on ' + f[1:] + ' Dataset')

    return max(accs[1])

def train_model_multiclass(directory, f, n_classes, opt):
    filepath = directory + f
    # net = cnn_2d_classes.ResNet1D(cnn_2d_classes.ResidualBlock,
    #                              [3, 8, 24, 3]).to(device)
    net = cnn_2d_classes.Net(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if opt == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)
    elif opt == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=0.001,
                               betas=(0.9, 0.999), weight_decay=0.0005,
                               amsgrad=False)
    else:
        raise ValueError('Invalid optimizer selected. Choose \'SGD\' or '
                         '\'Adam\'.')

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=n_schedule,
                                               gamma=0.1)

    print('Loading data...')
    train_ldr, test_ldr = load_data(filepath + '.csv')

    print('Training...')
    print('Filters per layer:', net.n_filters)
    print('Criterion:', criterion)
    print(optimizer)

    losses = [[], [100]]
    accs = [[], []]
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
            logits = net(local_batch)
            loss = criterion(logits, local_labels)
            loss.backward()
            optimizer.step()

            # Tracking
            train_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            train_total += local_labels.size(0)
            train_correct += (predicted == local_labels).sum().item()

        scheduler.step()
        train_acc = train_correct / train_total

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
                logits = net(local_batch)
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

        if val_loss >= losses[1][-2]:
            early_stopping += 1
        elif early_stopping > 0:
            early_stopping -= 1

        early = False
        if early_stopping == n_early:
            early = True

        print('Epoch:', epoch + 1,
              '| Train Acc:', round(train_acc, 8),
              '| Train Loss:', round(train_loss, 8),
              '| Val Acc:', round(val_acc, 8),
              '| Val Loss:', round(val_loss, 8),
              '| Early:', early_stopping)

        if early:
            print('Early stopping.')
            break

    losses[1] = losses[1][1:]

    # Plot loss and accuracy
    plt.figure()

    plt.subplot(121)
    plt.plot(losses[0], label='Training')
    plt.plot(losses[1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(122)
    plt.plot(accs[0], label='Training')
    plt.plot(accs[1], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    if f == '':
        plt.suptitle('2D ConvNet Training History on 12345 Dataset')
    else:
        plt.suptitle('2D ConvNet Training History on ' + f[1:] + ' Dataset')

    return max(accs[1])


if __name__ == '__main__':
    main()