import argparse
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

SGD = "SGDClassifier"
GAUSSIAN_NB = "GaussianNB"
RANDOM_FOREST = "RandomForestClassifier"
MLP = "MLPClassifier"
ADA_BOOST = "AdaBoostClassifier"

CLASSIFIERS = [SGD, GAUSSIAN_NB, RANDOM_FOREST, MLP, ADA_BOOST]


def initClassifier(clfName):
    ''' This function initializes a classifier 

    Parameters:
        clfName : classifier to be initialized
    Returns:
        initialized classifier
    '''

    if clfName == SGD:
        return SGDClassifier()
    elif clfName == GAUSSIAN_NB:
        return GaussianNB()
    elif clfName == RANDOM_FOREST:
        return RandomForestClassifier(max_depth=5, n_estimators=10)
    elif clfName == MLP:
        return MLPClassifier(alpha=0.05)
    elif clfName == ADA_BOOST:
        return AdaBoostClassifier()


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # accuracy = (tp+tn) / total
    return np.trace(C)/np.sum(C)


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    # recall = tp / (tp+fn)
    return np.diag(C)/np.sum(C, axis=1)


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    # recall = tp / (tp+fp)
    return np.diag(C)/np.sum(C, axis=0)


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    print("Running Experiment 3.1 - Comparing Classifiers")

    # first element stores index, second element stores accuracy of best clf
    iBest = (0, -2**31)

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i, clfName in enumerate(CLASSIFIERS):
            print(f"Training {clfName} classifier...")

            # init classifier
            clf = initClassifier(clfName)
            # train classifier with training data
            clf.fit(X_train, y_train)
            # test classifier on testing data
            y_pred = clf.predict(X_test)
            # obtain the confusion matrix
            C = confusion_matrix(y_test, y_pred)
            # get accuracy
            acc = accuracy(C)

            outf.write(f'Results for {clfName}:\n')
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall(C)]}\n')
            outf.write(
                f'\tPrecision: {[round(item, 4) for item in precision(C)]}\n')
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')

            # update best classifier index and accuracy
            if acc > iBest[1]:
                iBest = (i, acc)

            print("✓ Completed\n")

    print(
        f"Best classifier is {CLASSIFIERS[iBest[0]]} classifier with accuracy {iBest[1]}.\n")

    return iBest[0]


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print("Running Experiment 3.2 - Amount of Training Data")

    X_1k = []
    y_1k = []
    clfName = CLASSIFIERS[iBest]

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        for num_train in [1000, 5000, 10000, 15000, 20000]:
            print(f"Training {clfName} classifier with {num_train} samples...")

            # get random samples from the training set
            randRows = np.random.choice(
                X_train.shape[0], num_train, replace=False)

            if num_train == 1000:
                # save 1K random samples for output
                X_1k = X_train[randRows, :]
                y_1k = y_train[randRows]

            # init classifier
            clf = initClassifier(clfName)
            # train classifier with the random rows of training data
            clf.fit(X_train[randRows, :], y_train[randRows])
            # test classifier on testing data
            y_pred = clf.predict(X_test)
            # obtain the confusion matrix
            C = confusion_matrix(y_test, y_pred)
            # get accuracy
            acc = accuracy(C)

            outf.write(f'{num_train}: {acc:.4f}\n')
            print(f"✓ Completed: Accuracy {acc}\n")

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print("Running Experiment 3.3 - Feature Analysis")

    clfName = CLASSIFIERS[i]

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        for k_feat in [5, 50]:
            print(
                f"Selecting {k_feat} best features for {clfName} classifier...")

            # get selector for k best features and fit them on the full training data
            selector = SelectKBest(f_classif, k_feat)
            selector.fit_transform(X_train, y_train)

            # get p values from selector
            p_values = selector.pvalues_
            outf.write(
                f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')

            if k_feat == 5:
                # train the best clf for the 1K training set using 5 best features
                X_1k_new = selector.fit_transform(X_1k, y_1k)
                top_5_1k = selector.get_support(indices=True)
                clf = initClassifier(clfName)
                clf.fit(X_1k_new, y_1k)
                y_pred = clf.predict(selector.fit_transform(X_test, y_test))
                C = confusion_matrix(y_test, y_pred)
                # obtain accuracy
                accuracy_1k = accuracy(C)

                # train the best clf for the full training set using 5 best features
                X_full = selector.fit_transform(X_train, y_train)
                top_5_full = selector.get_support(indices=True)
                clf = initClassifier(clfName)
                clf.fit(X_full, y_train)
                y_pred = clf.predict(selector.fit_transform(X_test, y_test))
                C = confusion_matrix(y_test, y_pred)
                # obtain accuracy
                accuracy_full = accuracy(C)

            print("✓ Completed\n")

        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(
            f'Chosen feature intersection: {set(np.intersect1d(top_5_1k, top_5_full))}\n')
        outf.write(f'Top-5 at higher: {set(top_5_full)}\n')

        print(f"Top 5 features are {set(top_5_full)}.\n")


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    print("Running Experiment 3.4 - Cross-Fold Validation")

    # join training and testing data to obtain full dataset
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    nsplits = 5

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # use KFold to get 5 splits/folds of training/testing data
        kf = KFold(n_splits=nsplits, shuffle=True)
        splits = kf.split(X)

        # array to store accuracies for each fold
        all_5_fold_accs = []

        for train_index, test_index in splits:
            print(
                f"Validating on fold {len(all_5_fold_accs)}...")

            # array to store accuracies for each classifier
            kfold_accuracies = []

            # split the training and testing data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # train all classifiers for each fold
            for clfName in CLASSIFIERS:
                print(f"Training {clfName} classifier...")

                clf = initClassifier(clfName)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                C = confusion_matrix(y_test, y_pred)
                kfold_accuracies.append(accuracy(C))

                print("✓ Completed")

            outf.write(
                f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            all_5_fold_accs.append(kfold_accuracies)
            print("✓ Completed\n")

        # accuracies across the 5 folds for the best classifier
        bestAccs = [all_5_fold_accs[f][i] for f in range(nsplits)]

        p_values = []
        for c in range(len(CLASSIFIERS)):
            if c == i:
                continue

            # accuracies across the 5 folds for the secondary classifier
            accsToCompare = [all_5_fold_accs[f][c] for f in range(nsplits)]
            p_values.append(ttest_rel(bestAccs, accsToCompare).pvalue)

        outf.write(
            f'p-values: {[round(pval, 4) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # load data and split into train and test.
    data = np.load(args.input)["arr_0"]
    train, test = train_test_split(data, test_size=0.2, train_size=0.8)

    X_train = train[:, :173]
    y_train = train[:, 173:].ravel()
    X_test = test[:, :173]
    y_test = test[:, 173:].ravel()

    output_dir = args.output_dir or "."

    # complete each classification experiment, in sequence.
    iBest = class31(output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(output_dir, X_train, X_test, y_train, y_test, iBest)
