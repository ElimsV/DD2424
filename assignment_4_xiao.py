import numpy as np
from data_preprocess import char2ind, ind2char, Load_Data
from model import RNN
from utils import softmax, check_grad
import config as cfg

if __name__ == '__main__':
    path = "./data/goblet.txt"
    data_loader = Load_Data(path)
    file_data = data_loader.load_data()[0]  # load_data returns a tuple of 3 elements

    char2int = data_loader.char_to_int()  # dict char to int
    int2char = data_loader.int_to_char()  # dict int to char

        ######test compute loss######
    # init hidden state
    # h0 = np.random.standard_normal([rnn_net.m, 1])
    h0 = np.zeros([5, 1])
    seq_len = 25
    # init rnn network
    rnn_net = RNN(data_loader.K, h0, seq_len)

    X_onehot = np.zeros([rnn_net.K, seq_len])
    target_onehot = np.zeros([rnn_net.K, seq_len])
    X_int = [char2int[ch] for ch in file_data[:seq_len]]
    target_int = [char2int[ch] for ch in file_data[1:seq_len + 1]]
    X_onehot[X_int, range(seq_len)] = 1
    target_onehot[target_int, range(seq_len)] = 1

    # loss = rnn_net.compute_loss(X_onehot, target_onehot)
    # print(loss)

    ######test gradient computation######
    # grads = rnn_net.backward_pass(X_onehot, target_onehot)
    # for grad in grads:
    #     print(grad)

    check_grad(rnn_net, X_onehot, target_onehot)
