
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
    results = np.empty((0,15), float)
    for file in files1:
        tmppath = path1 + "/" + file
        lines = np.loadtxt(tmppath, dtype=int)
        feature_mat1 = lines.reshape(lines.size, 1)
        result = feature_core.sequence_feature(feature_mat1, 100, 10)[:,[0,1,2,3,4,6,7,10,11,12,14,15,16,17,18]]
        # for i, row in enumerate(result):
        #     for j, x in enumerate(row):
        #         if not isinstance(x, float) and not isinstance(x, int):
        #             print j
        #[:,[0,1,2,3,4,6,7,8,10,11,12,13]]
        #np.savetxt("/Users/keqinli/PycharmProjects/mobile_team/tmp.txt", np.array(result))
        #print result
        results = np.append(results, result, axis=0)


    X_0 = np.array(results)
    #print(X_0.size)
    #Y_0 = np.zeros(X_0.size)
    # for i, row in enumerate(X_0):
    #     for j, x in enumerate(row):
    #         if not isinstance(x, float):
    #             print j
    #print X_0

    return X_0

def main():

    #Train
    X_0 = process(path1)
    # for i, row in enumerate(X_0):
    #     for j, x in enumerate(row):
    #         if not isinstance(x, float):
    #             print i, j, x
    #print X_0
    np.savetxt(x0_pathWrite, X_0)
    X_1 = process(path2+"/"+"shuqi")
    X_2 = process(path2 + "/" + "xiao")
    X = np.vstack((X_1, X_2))
    np.savetxt(x1_pathWrite, X)


    #Test
    T_0 = process("/Users/keqinli/PycharmProjects/mobile_team/clear_data/test/open")
    np.savetxt("/Users/keqinli/PycharmProjects/mobile_team/test0.txt", T_0)

    T_1 = process("/Users/keqinli/PycharmProjects/mobile_team/clear_data/test/walk")
    np.savetxt("/Users/keqinli/PycharmProjects/mobile_team/test1.txt", T_1)


if __name__ == '__main__':
    main()