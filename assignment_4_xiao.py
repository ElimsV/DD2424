import numpy as np
from data_preprocess import char2ind, ind2char, Load_Data
from model import RNN

if __name__ == '__main__':
    path = "./goblet.txt"
    data_loader = Load_Data(path)
    file_data = data_loader.load_data()[0]  # load_data returns a tuple of 3 elements

    # test char2ind
    c = file_data[23]
    print("c: ", c)
    ind = char2ind(c, data_loader.unique_chars, data_loader.ind_mat, data_loader.K)
    print(ind.T.shape)
    print(np.dot(ind.T, ind))

    # test ind2char
    char = ind2char(ind, data_loader.unique_chars, data_loader.K)
    print(char)
