
import numpy as np
import feature_core

import csv
import os

#from sklearn.svm import SVC

# one file
path1 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/open"
path2 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/walk"

x_pathWrite = "/Users/keqinli/PycharmProjects/mobile_team/x.txt"
y_pathWrite = "/Users/keqinli/PycharmProjects/mobile_team/y.txt"

def eee():
    path1 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/open"
    files1 = os.listdir(path1)
    results = np.empty((0,19), int)
    for file in files1:
        tmppath = path1 + "/" + file
        lines = np.loadtxt(tmppath, dtype=int)
        feature_mat1 = lines.reshape(lines.size, 1)
        result = feature_core.sequence_feature(feature_mat1, 5, 4)
        results = np.append(results, result, axis=0)

    X_0 = np.array(results)
    #print(X_0.size)
    Y_0 = np.zeros(X_0.size)

    return X_0
    # mat = np.matrix(X_0)
    # with open(x_pathWrite, 'w') as f:
    #     for line in mat:
    #         np.savetxt(x_pathWrite, line, newline=" ")
    #
    #
    # with open(y_pathWrite, 'w') as f:
    #     np.savetxt(x_pathWrite, Y_0, newline=" ")

    # clf = SVC()
    # clf.fit(X_0, Y_0)

def rrr():
    path2 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/walk"
    files1 = os.listdir(path2)
    results = np.empty((0,19), int)
    for file in files1:
        tmppath = path1 + "/" + file
        lines = np.loadtxt(tmppath, dtype=int)
        feature_mat1 = lines.reshape(lines.size, 1)
        result = feature_core.sequence_feature(feature_mat1, 5, 4)
        results = np.append(results, result, axis=0)

    X_0 = np.array(results)
    #print(X_0.size)
    Y_0 = np.zeros(X_0.size)

    return X_0
def main():

    #Train
    X_0 = eee()
    X_1 = rrr()

    X = np.vstack((X_0, X_1))
    y = np.array([0] * len(X_0) + [1] * len(X_1))

    #Test


    #test("/Users/keqinli/PycharmProjects/mobile_team/clear_data/open")
    #test("/Users/keqinli/PycharmProjects/mobile_team/clear_data/walk")

if __name__ == '__main__':
    main()

