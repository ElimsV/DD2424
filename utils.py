import numpy as np


def softmax(mat):
    mat_exp = np.exp(mat)
    out = mat_exp / np.sum(mat_exp, axis=0)
    return out


if __name__ == '__main__':
    mat = np.asarray([[1, 2], [3, 4]])
    print(softmax(mat))
