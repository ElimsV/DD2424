import numpy as np


def softmax(mat):
    try:
        mat_exp = np.exp(mat)
        out = mat_exp / np.sum(mat_exp, axis=0)
    except FloatingPointError:
        out = np.full_like(mat, fill_value=np.finfo(float).eps)
        out[np.argmax(mat)] = 1 - (mat.size-1) * np.finfo(float).eps
    return out


if __name__ == '__main__':
    mat = np.asarray([[1, 2], [3, 4]])
    print(softmax(mat))
