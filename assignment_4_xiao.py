import numpy as np
import os
from data_preprocess import char2ind, ind2char, Load_Data
from model import RNN
import config as cfg
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tag = "10epochs"
    h_prev = np.zeros([cfg.M, 1])

    # load data
    path = cfg.PATH
    data_loader = Load_Data(path)
    file_data = data_loader.load_data()[0]  # load_data returns a tuple of 3 elements
    # file_data = file_data[:25000]
    data_len = len(file_data)

    # char int converter
    char2int = data_loader.char_to_int()  # dict char to int
    int2char = data_loader.int_to_char()  # dict int to char

    # init rnn network
    rnn_net = RNN(data_loader.K, h_prev, cfg.BATCH_SIZE, data_loader.unique_chars)

    # generate onehot representation of training data and label
    X_onehot = np.zeros([rnn_net.K, data_len])
    target_onehot = np.zeros([rnn_net.K, data_len])
    X_int = [char2int[ch] for ch in file_data]
    target_int = [char2int[ch] for ch in file_data[1:]+file_data[0]]
    X_onehot[X_int, range(data_len)] = 1
    target_onehot[target_int, range(data_len)] = 1
    del file_data, X_int, target_int

    # start training
    smooth_loss_acc = rnn_net.train(X_onehot, target_onehot, h_prev, int2char, epoch_num=cfg.EPOCH, batch_size=cfg.BATCH_SIZE)
    print("Smoothed loss:")
    print(smooth_loss_acc)

    # save results
    loss_save_path = os.path.join(cfg.SAVE_PATH, tag + '_loss.npy')
    fig_save_path = os.path.join(cfg.SAVE_PATH, tag + '_loss.png')
    np.save(loss_save_path, smooth_loss_acc)

    fig = plt.figure()
    plt.plot(range(len(smooth_loss_acc)), smooth_loss_acc)
    plt.xlabel("Iterations (x100)")
    plt.ylabel("Smoothed Loss")
    # plt.show()
    plt.savefig(fig_save_path, dpi=300)

