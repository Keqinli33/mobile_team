
import numpy as np
import feature_core

import csv
import os

from sklearn.svm import SVC

# one file

path2 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/walk"

x_pathWrite = "/Users/keqinli/PycharmProjects/mobile_team/x.txt"
y_pathWrite = "/Users/keqinli/PycharmProjects/mobile_team/y.txt"
#files = os.listdir(path)

def test():
    path1 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/open"
    files1 = os.listdir(path1)
    results = np.empty((0,19), int)
    for file in files1:
        path = path1 + "/" + file
        lines = np.loadtxt(path, dtype=int)
        feature_mat1 = lines.reshape(lines.size, 1)
        result = feature_core.sequence_feature(feature_mat1, 5, 4)
        results = np.append(results, result, axis=0)

    X_0 = np.array(results)
    print(X_0.size)
    Y_0 = np.zeros(X_0.size)

    mat = np.matrix(X_0)
    with open(x_pathWrite, 'wb') as f:
        for line in mat:
            np.savetxt(f, line)


    with open(y_pathWrite, 'wb') as f:
        for 



    # clf = SVC()
    # clf.fit(X_0, Y_0)



    #print(Y_0)
    #print(Y_0.size)

if __name__ == '__main__':
    test()

