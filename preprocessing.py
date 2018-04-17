
import numpy as np
import feature_core

import csv
import os

#from sklearn.svm import SVC

# one file
path1 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/open"
path2 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/walk"

x0_pathWrite = "/Users/keqinli/PycharmProjects/mobile_team/x0.txt"
x1_pathWrite = "/Users/keqinli/PycharmProjects/mobile_team/x1.txt"

def process(path1):
    #path1 = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/open"
    files1 = os.listdir(path1)
    results = np.empty((0,8), float)
    for file in files1:
        tmppath = path1 + "/" + file
        lines = np.loadtxt(tmppath, dtype=int)
        feature_mat1 = lines.reshape(lines.size, 1)
        result = feature_core.sequence_feature(feature_mat1, 5, 4)[:,:8]
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
def main():

    #Train
    # X_0 = process(path1)
    # # for i, row in enumerate(X_0):
    # #     for j, x in enumerate(row):
    # #         if not isinstance(x, float):
    # #             print i, j, x
    # #print X_0
    # np.savetxt(x0_pathWrite, X_0)
    # X_1 = process(path2+"/"+"shuqi")
    # X_2 = process(path2 + "/" + "xiao")
    # X = np.vstack((X_1, X_2))
    # np.savetxt(x1_pathWrite, X)


    #Test
    X_0 = process("/Users/keqinli/PycharmProjects/mobile_team/clear_data/test/open")
    np.savetxt("/Users/keqinli/PycharmProjects/mobile_team/test0.txt", X_0)

    X_1 = process("/Users/keqinli/PycharmProjects/mobile_team/clear_data/test/walk")
    np.savetxt("/Users/keqinli/PycharmProjects/mobile_team/test1.txt", X_1)


if __name__ == '__main__':
    main()