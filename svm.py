import numpy as np
from sklearn import svm

train0 = "/Users/keqinli/PycharmProjects/mobile_team/x0.txt"
train1 = "/Users/keqinli/PycharmProjects/mobile_team/x1.txt"

test0 = "/Users/keqinli/PycharmProjects/mobile_team/test0.txt"
test1 = "/Users/keqinli/PycharmProjects/mobile_team/test1.txt"

def get_input():

    # Train
    X_0 = np.loadtxt(train0)
    X_1 = np.loadtxt(train1)
    X = np.vstack((X_0, X_1))
    y = np.array([0] * len(X_0) + [1] * len(X_1))

    # Test
    X_test_0 = np.loadtxt(test0)
    X_test_1 = np.loadtxt(test1)
    X_test = np.vstack((X_test_0, X_test_1))
    y_test = np.array([0] * len(X_test_0) + [1] * len(X_test_1))

    return X, y, X_test, y_test

def main():

    X, y, X_test, y_test = get_input()
    model = svm.SVC(kernel="linear", C=1, gamma=1)

    # Train
    model.fit(X, y)

    # Predict
    y_predict = model.predict(X_test)
    print("Correctness: %f" % ((y_test == y_predict).sum() / len(y_test)))

if __name__ == "__main__":
    main()