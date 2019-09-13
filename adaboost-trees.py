import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    directory = '.\Data\CombinedV2Filter'
    files = ['_12', '_13', '_14', '_23', '_24', '_34', '_123', '_234',
             '_1234', '']
    performance = []

    for f in files:
        print('Loading data...')
        filepath = directory + f + '.csv'
        data = pd.read_csv(filepath)
        labels = data.iloc[:, 0]
        data = data.iloc[:, 1:]
        d_train, d_test, y_train, y_test = tts(data, labels,
                                               test_size=0.3,
                                               random_state=42)
        scaler = StandardScaler()
        d_train = scaler.fit_transform(d_train)
        d_test = scaler.transform(d_test)

        print('Initializing base classifier...')
        BaseClassifier = DecisionTreeClassifier(criterion='gini',
                                                splitter='best',
                                                max_depth=1,
                                                min_samples_split=2,
                                                min_samples_leaf=1,
                                                min_weight_fraction_leaf=0.0,
                                                max_features=None,
                                                random_state=7,
                                                max_leaf_nodes=None,
                                                min_impurity_decrease=0.0,
                                                class_weight=None,
                                                presort=False)

        print('Initializing boosted ensemble...')
        ensemble = AdaBoostClassifier(BaseClassifier,
                                        n_estimators=1000,
                                        learning_rate=1.0,
                                        algorithm='SAMME.R',
                                        random_state=3901)

        print('Fitting model on training data...')
        ensemble.fit(d_train, y_train)

        test_errors = []

        print('Predicting on test data...')
        for prediction in ensemble.staged_predict(d_test):
            test_errors.append(1. - accuracy_score(y_test, prediction))

        n_trees = len(ensemble)
        classifier_errors = ensemble.estimator_errors_[:n_trees]

        print('Plotting...')
        plt.figure()
        plt.subplot(121)
        plt.plot(range(1, n_trees + 1),
                 test_errors,
                 c='black', label='Test Errors')
        plt.legend()
        plt.grid()
        plt.xlabel('Number of Trees')
        plt.ylabel('Test Error')

        plt.subplot(122)
        plt.plot(range(1, n_trees + 1),
                 classifier_errors,
                 c='black', label='Classifier Error')
        plt.legend()
        plt.grid()
        plt.xlabel('Number of Trees')
        plt.ylabel('Estimator Error')

        performance.append(test_errors[-1])

        print('Finished', f)
        print('------------')

    for f, p in zip(files, performance):
        if f == '':
            print('Performance on 12345 :', p)
        else:
            print('Performance on', f[1:], ':', p)

    plt.show()
    print('------------')
    print('Done.')
if __name__ == '__main__':
    main()