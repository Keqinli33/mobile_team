
import numpy as np
import feature_core

import csv
import os

# one file
path = "/Users/keqinli/PycharmProjects/mobile_team/clear_data/open/1.txt"
#files = os.listdir(path)

def test():
    lines = list()
    # with open(path) as file:
    #     for line in file:
    #         #line = line.strip()  # or some other preprocessing
    #         lines.append(np.loadtxt(f, skiprows=1, dtype=int))  # storing everything in memory!

    lines = np.loadtxt(path, dtype=int)

    narray =   np.array(lines)
    feature_mat = narray.reshape(narray.size, 1)
    #a = np.arange(0, 10).reshape((10, 1))
    print(feature_core.sequence_feature(feature_mat, 5, 4))




if __name__ == '__main__':
    test()

