from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.optimizers import adam
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Print flags
print_ = False

# Suppress TensorFlow warning messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TensorFlow configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Load data
CleanedDataPath = '.\Data\MRMR\DataMRMRAllFeaturesCombined100_24Class.csv'

my_data = np.genfromtxt(CleanedDataPath, delimiter=',')
#USED TO BE 0:, 1:len(my_data)+1]
d = my_data[:, 1:]
l = my_data[:, 0]

# use SMOTE algorithm to balance classes
sm = SMOTE(k_neighbors=5)
X_res, y_res = sm.fit_sample(d, l)

if print_:
    print('Length of original dataset:', l.shape[0])
    print('Length of oversampled dataset:', y_res.shape[0])
    print('Number of added samples:', len(y_res)-len(l))

y_res = y_res + np.concatenate((np.zeros(len(l)), 10*np.ones(len(y_res)-len(l))),axis=0)
#print('Resampled dataset shape {}'.format(Counter(y_res)))

if print_:
    print('y_res:')
    print(y_res)

####################################################################################################################
k = 20
skf = StratifiedKFold(n_splits=k, shuffle=True)
nnscores = []
forestscores = []

epochs = 5000
batchsize = 5

truelabels = []
predictedlabels = []

plt.figure()
for index, (train_indices, val_indices) in enumerate(skf.split(X_res, y_res)):
    print("Training on fold " + str(index + 1) + '/' + str(k))
    # extract real data from smote
    # Generate batches from indices
    xtrain, xval = X_res[train_indices], X_res[val_indices]
    ytrain, yval = y_res[train_indices], y_res[val_indices]

    real_indx = [] #indexes of real validation data

    # Dealing with synthetic samples in validation split
    for i in range(len(ytrain)): #reduce all smote in training by 10
        if ytrain[i] >= 10:
            ytrain[i] -= 10

    for i in range(len(yval)): #reduce all smote in validation by 10 and create array of real values
        if yval[i] >= 10:
            yval[i] -= 10
        else:
            real_indx.append(i)

    if print_:
        print('val_indices:')
        print(val_indices)
        print('real_indx:')
        print(real_indx)
        print('Length of yval:', yval.shape[0])
        print('Length of real_indx:', len(real_indx))

    # One-hot encoding
    ytrain_hot = np_utils.to_categorical(ytrain)
    yval_hot = np_utils.to_categorical(yval)

    # Subtracting one from each label
    ytrain_hot = ytrain_hot[:, 1:len(ytrain_hot[:, 0])]
    yval_hot = yval_hot[:, 1:len(yval_hot[:, 0])]

    #######################################################################################################################
    '''
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                warm_start=False, class_weight=None)
    rf.fit(xtrain, ytrain)
    forestscores.append(rf.score(xval[real_indx], yval[real_indx]) * 100)
    '''

    # print('Random Forest feature importances')
    # print(rf.feature_importances_)
    # print('Random Forest Accuracy')
    # print(rf.score(xval,yval))


    #######################################################################################################################
    class Get_Val_Acc(Callback):
        def on_train_begin(self, logs={}):
            # Initialize running record of validation accuracies
            self.accuracies = []

        def on_epoch_end(self, epoch, logs={}):
            # Add current epoch's validation accuracy to running record
            self.accuracies.append(logs.get('val_acc'))

            # Maintain the highest validation accuracy over all epochs
            self.max_acc = max(self.accuracies)

            # If the current epoch's validation accuracy is the highest...
            if logs.get('val_acc') == self.max_acc:

                # ...store the predictions for the current epoch only
                global predicted
                predicted = model.predict(xval[real_indx, :])



    ##############################################################################################################################

    # Experiment: Single output node vs. one-hot encoding

    # Define neural network
    model = Sequential()
    model.add(Dense(20, input_shape=(len(xtrain[0, :]),), activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(len(ytrain_hot[0, :]), activation='sigmoid'))
    a = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=a, loss='binary_crossentropy',
                  metrics=['accuracy'])
    #model.summary()

    # Callback()
    patience = 50
    earlyStop = EarlyStopping(monitor='val_loss', patience=patience,
                              min_delta=0, verbose=0, mode='auto')
    acc = Get_Val_Acc()  # object used to find max acc of training
    history = model.fit(xtrain, ytrain_hot, epochs=epochs,
                        callbacks=[earlyStop, acc],
                        validation_data=(xval[real_indx, :],
                                         yval_hot[real_indx, :]),
                        batch_size=batchsize, verbose=0, shuffle='true')

    ytrue = yval[real_indx]-1

    ypred = []
    for i in range(0, len(predicted)):
        ypred.append(np.argmax(predicted[i, :]))
        predictedlabels.append(np.argmax(predicted[i, :]))

    truelabels = np.append(truelabels, ytrue)
    score = accuracy_score(ytrue, ypred)

    print('Validation split size:', len(ytrue))
    print('Best Performance:', score)
    print('ytrue:', ytrue)
    print('ypred:', ypred)
    print('------------')
    plt.plot(history.history['val_acc'],
             label='Fold %d | Score: %.2f%% | Training Set Size: %d | '
                   'Validation Set Size: %d' %
                   (index + 1, score * 100, len(ytrain), len(ytrue)))

    # evaluate the model
    #print("max validation %s: %.2f%%" % (model.metrics_names[1], acc.max_acc * 100))
    nnscores.append(score * 100)
plt.grid()
plt.legend(loc='lower right')
plt.title('Training History (Validation Accuracy) | Reported Score: %.2f%% ('
          '+/- %.2f%%)' % (np.mean(nnscores), np.std(nnscores)))
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
# model.save('nnweights')
#print(nnscores)
print(str(k) + ' fold Cross-Validation Accuracy:')
print('Neural Network:')
print("%.2f%% (+/- %.2f%%)" % (np.mean(nnscores), np.std(nnscores)))
#print('Random Forest:')
#print("%.2f%% (+/- %.2f%%)" % (np.mean(forestscores), np.std(forestscores)))

predictedlabels = np.asarray(predictedlabels)

print('predicted labels: ', predictedlabels)
print('true labels: ', truelabels)

####################################################################################################################

from pandas_ml import ConfusionMatrix

cnf = ConfusionMatrix(truelabels, predictedlabels)
import matplotlib.pyplot as plt
cnf.print_stats()

#####################################################################################################################
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    plt.rc('font', **font)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(truelabels, predictedlabels)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['MCI1','AD'], normalize=True,
                     title='MCI1 vs AD')

#plt.figure()
#plot_confusion_matrix(cnf_matrix, class_size,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, class_size, normalize=True,
#                    title='Normalized confusion matrix')


import numpy as np
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

m,h = mean_confidence_interval(nnscores)

print("%.2f%% (+/- %.2f%%)" % (m, h))


plt.show()