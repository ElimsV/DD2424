import os
import numpy as np


def char2ind(char, char_tuple, ind_mat, l):
    char_tuple = tuple(char_tuple)
    assert char in char_tuple, "Illegal character input!"
    ind = char_tuple.index(char)
    ind_onehot = ind_mat[ind, :]
    ind_onehot = np.reshape(ind_onehot, [l, 1])
    return ind_onehot


def ind2char(ind_onehot, char_tuple, l):
    char_tuple = tuple(char_tuple)
    assert np.size(ind_onehot)==l, "Illegal one hot ind!"
    ind = np.where(ind_onehot==1)[0][0]
    char = char_tuple[ind]
    return char


class Load_Data:
    def __init__(self, path):
        assert os.path.exists(path)
        self.file_path = path
        self.unique_chars = []
        self.K = 0
        self.ind_mat = 0

    def load_data(self):
        with open(self.file_path, 'r') as f:
            file_data = f.read()
            self.unique_chars = list(set(file_data))
            self.K = len(self.unique_chars)
            self.ind_mat = np.identity(self.K)
        print("Data loaded!")
        print("Unique chars: ", self.unique_chars)
        print("K: ", self.K)
        return file_data, self.unique_chars, self.K

    def char_to_int(self):
        return dict((c, i) for i, c in enumerate(self.unique_chars))

    def int_to_char(self):
        return dict((i, c) for i, c in enumerate(self.unique_chars))

if __name__ == "__main__":
    path = "./goblet.txt"
    Data = Load_Data(path)
    file_data = Data.load_data()[0]  # load_data returns a tuple of 3 elements
    # print(file_data)
    # char_list = unique_data(path)
    # l = len(char_list)
    #
    # ind_mat = np.identity(l)
    # char_tuple = tuple(char_list)
    # print(char_tuple)

    # test char2ind
    c = file_data[25]
    print("c: ", c)
    ind = char2ind(c, Data.unique_chars, Data.ind_mat, Data.K)
    # print(ind.T.shape)
    # print(np.dot(ind.T, ind))

    # test ind2char
    char = ind2char(ind, Data.unique_chars, Data.K)
    print(char)

    char2int = Data.char_to_int()
    int2char = Data.int_to_char()
    # test char_to_int
    c = file_data[25]
    print(char2int[c])
    # test ind_to_char
    print(int2char[char2int[c]])
