from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.utils import resample
import numpy as np
import argparse
import sys
import os
import csv
from scipy import stats



def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    correct = 0.0
    total = 0.0
    for i in range(len(C)):
        correct += C[i][i]
        total += np.sum(C[i])
    return correct/total

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    recall = []
    for i in range(len(C)):
        recall.append(C[i][i]/np.sum(C[i]))

    return recall


def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    precision = []

    for i in range(len(C)):
        precision.append(C[i][i]/np.sum(C[:][i]))

    return precision


def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    data = np.load(filename)
    data = data[data.files[0]]
    X_train, X_test, y_train, y_test = train_test_split(data[...,:-1], data[...,-1], train_size=0.8)

    print("Training Linear SVM")
    svm_lin = SVC(kernel = 'linear', max_iter=10000).fit(X_train, y_train)
    print("Training Radial SVM")
    svm_rad = SVC(kernel='rbf', max_iter=10000,gamma=2).fit(X_train, y_train)
    print("Training RFC")
    RFC = RandomForestClassifier(max_depth=5, n_estimators=10).fit(X_train, y_train)
    print("Training MLP")
    MLP = MLPClassifier(alpha=0.05).fit(X_train, y_train)
    print("Training Adaboost")
    Ada = AdaBoostClassifier().fit(X_train, y_train)
    classifiers = {0:svm_lin, 1:svm_rad, 2:RFC, 3:MLP, 4:Ada}
    with open('a1_3.1.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        iBest = 0
        best_accuracy = 0
        for i in range(5):
            y_pred = classifiers[i].predict(X_test)
            C = confusion_matrix(y_test, y_pred)
            acc = accuracy(C)
            entry = [i]+[acc]+recall(C)+precision(C)
            for row in C:
                for val in row:
                    entry.append(val)
            writer.writerow(entry)
            if acc>best_accuracy:
                iBest = i
                best_accuracy = acc








    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    if iBest == 0:
        clf = SVC(kernel = 'linear',max_iter=10000)
    elif iBest == 1:
        clf = SVC(kernel='rbf',max_iter=10000, gamma=2)
    elif iBest == 2:
        clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    elif iBest == 3:
        clf = MLPClassifier(alpha=0.05)
    elif iBest ==4:
        clf = AdaBoostClassifier()
    sizes = [1000, 5000, 10000, 15000, 20000]
    X_1k, y_1k=resample(X_train, y_train, n_samples = 1000)
    with open('a1_3.2.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        accuracies = []
        for size in sizes:
            new_x_train, new_y_train = resample(X_train, y_train, n_samples = size)
            clf.fit(new_x_train,new_y_train)
            y_pred = clf.predict(X_test)
            C = confusion_matrix(y_test, y_pred)
            accuracies.append(accuracy(C))
        writer.writerow(accuracies)




    return (X_1k, y_1k)

def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    sizes = [5,10,20,30,40,50]
    if i == 0:
        clf32 = SVC(kernel = 'linear',max_iter=10000)
        clf1 = SVC(kernel = 'linear',max_iter=10000)
    elif i == 1:
        clf32 = SVC(kernel='rbf', max_iter=10000,gamma=2)
        clf1 = SVC(kernel='rbf', max_iter=10000,gamma=2)
    elif i == 2:
        clf32 = RandomForestClassifier(max_depth=5, n_estimators=10)
        clf1 = RandomForestClassifier(max_depth=5, n_estimators=10)
    elif i == 3:
        clf32 = MLPClassifier(alpha=0.05)
        clf1 = MLPClassifier(alpha=0.05)
    elif i ==4:
        clf32 = AdaBoostClassifier()
        clf1 = AdaBoostClassifier()

    with open('a1_3.3.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        for num in sizes:
            selector32 = SelectKBest(f_classif,k = num)
            selector32.fit_transform(X_train, y_train)
            # selector1 = SelectKBest(f_classif,k = num)
            # selector1.fit_transform(X_1k, y_1k)

            pvals32 = sorted(selector32.pvalues_)
            # pvals1 = sorted(selector1.pvalues_)


            writer.writerow([num]+pvals32[:num])
        # print(pvals32[:10])
        # print(pvals1[:10])
        # for i in range(50):
        #     idx32 = np.where(selector32.pvalues_ == pvals32[i])
        #     idx1 = np.where(selector1.pvalues_ == pvals1[i])
        #     print(idx1, idx32)
        selector = SelectKBest(f_classif, k = num)
        X_new_32 = selector.fit_transform(X_train, y_train)
        X_new_1 = selector.transform(X_1k)
        X_test_new = selector.transform(X_test)
        clf32.fit(X_new_32, y_train)
        clf1.fit(X_new_1, y_1k)
        y_pred32 = clf32.predict(X_test_new)
        y_pred1 = clf1.predict(X_test_new)
        C32 = confusion_matrix(y_test, y_pred32)
        C1 = confusion_matrix(y_test,y_pred1)
        writer.writerow([accuracy(C1)] + [accuracy(C32)])








def class34( filename, i ):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
    '''
    data = np.load(filename)
    data = data[data.files[0]]
    kfold = KFold(n_splits = 5, shuffle = True)
    X = data[..., :-1]
    y = data[..., -1]

    cross = np.empty((5,5))
    j=0
    for train_idx, test_idx in kfold.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        for i in range(5):
            
            if i == 0:
                clf = SVC(kernel='linear', max_iter=10000)
            if i == 1:
                clf = SVC(kernel='rbf', max_iter=10000, gamma = 2)  # default is rdf
            if i == 2:
                clf = RandomForestClassifier(max_depth=5, n_estimators=10)
            if i == 3:
                clf = MLPClassifier(alpha=0.05)
            if i == 4:
                clf = AdaBoostClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            C = confusion_matrix(y_test, y_pred)
            cross[j][i] = accuracy(C)
        j+=1
    with open('./a1_3.4.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in cross:
            writer.writerow(row)
        p = []
        for col in range(5):
            if col!=i:
                S = stats.ttest_rel(cross[:][col],  cross[:][i])
                p.append(S[1])

        writer.writerow(p)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    X_train, X_test, y_train, y_test,iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test,iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)
