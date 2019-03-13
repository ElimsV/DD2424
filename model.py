import numpy as np
from data_preprocess import char2ind, ind2char, Load_Data
from utils import softmax

class RNN():
    def __init__(self, K):
        # Hyper-parameters
        self.m = 100  # hidden state
        self.eta = 0.1  # learning rate
        self.seq_length = 25
        self.K = K
        self.sig = 0.01

        # Network parameters
        self.b = np.zeros([self.m, 1])
        self.c = np.zeros([self.K, 1])
        self.U = np.random.standard_normal(size=[self.m, self.K]) * self.sig
        self.W = np.random.standard_normal(size=[self.m, self.m]) * self.sig
        self.V = np.random.standard_normal(size=[self.K, self.m]) * self.sig


    def forward_pass(self, h_prev, x):
        # x: K*1
        # h: m*1
        # o: K*1
        # p: K*1
        a = np.dot(self.W, h_prev) + np.dot(self.U, x) + self.b
        h = np.tanh(a)
        o = np.dot(self.V, h) + self.c
        p = softmax(o)
        return h, p

    def generate_chars(self, h0, x0, batch_size):
        """
        Generate text of length batch_size
        :param h0: init hidden state
        :param x0: init one-hot input
        :param batch_size: output length
        :return:
        """
        count = 0
        h_prev = h0
        x = x0

        int_list = []
        while count < batch_size:
            h_prev, p = self.forward_pass(h_prev, x)
            int_max = np.argmax(p)
            int_list.append(int_max)

            # update next x
            ind = np.random.choice(self.K, 1, p=p.reshape(self.K), replace=False)
            x = np.zeros([self.K, 1])
            x[ind] = 1

            count += 1

        assert len(int_list) == batch_size, "Erroneous output length!"
        return int_list

    def predict_prob(self, h_prev, X):
        """
        predict prob output
        :param h_prev: previous hidden state
        :param X: K*seq_len
        :return prob: K*seq_len, prob output
        """
        Y = np.zeros_like(X)
        prob = np.zeros_like(X)
        for t in range(X.shape[1]):
            h_prev, p = self.forward_pass(h_prev, X[:, t].reshape([self.K, 1]))
            prob[:, t] = p.reshape(self.K)  # probability output
        return prob, h_prev

    def predict_onehot(self, h_prev, X):
        """
        predict onehot output
        :param h_prev: previous hidden state
        :param X: K*seq_len
        :return Y: K*seq_len, one-hot output
        """
        prob, _ = self.predict_prob(h_prev, X)
        Y = np.zeros_like(prob)
        int_max = np.argmax(prob, axis=0)
        Y[int_max, range(prob.shape[1])] = 1  # onehot output
        return Y, h_prev

    def compute_loss(self, h_prev, X, target):
        assert X.shape == target.shape, "X and target shape error!"
        prob, _ = self.predict_prob(h_prev, X)
        tmp = np.sum(prob * target, axis=0)
        tmp[tmp == 0] = np.finfo(float).eps
        return - np.sum(np.log(tmp))


if __name__ == "__main__":
    path = "./goblet.txt"
    data_loader = Load_Data(path)
    file_data = data_loader.load_data()[0]  # load_data returns a tuple of 3 elements

    char2int = data_loader.char_to_int()  # dict char to int
    int2char = data_loader.int_to_char()  # dict int to char
    
    # ######test generation chars######
    # # init input char
    # c0 = 'a'
    # int_0 = char2int[c0]
    # x0 = np.zeros([data_loader.K, 1])
    # x0[int_0] = 1
    #
    # # init rnn network
    # rnn_net = RNN(data_loader.K)
    #
    # # init hidden state
    # # h0 = np.random.standard_normal([rnn_net.m, 1])
    # h0 = np.zeros([rnn_net.m, 1])
    #
    # # output length
    # n = 20
    #
    # # generate text
    # int_list = rnn_net.generate_chars(h0, x0, n)
    # text = ''
    # for i in int_list:
    #     text = text + int2char[i]
    # print(text)

    ######test compute loss######
    # init rnn network
    rnn_net = RNN(data_loader.K)

    # init hidden state
    # h0 = np.random.standard_normal([rnn_net.m, 1])
    h0 = np.zeros([rnn_net.m, 1])

    seq_len = 25
    X_onehot = np.zeros([rnn_net.K, seq_len])
    target_onehot = np.zeros([rnn_net.K, seq_len])
    X_int = [char2int[ch] for ch in file_data[:seq_len]]
    target_int = [char2int[ch] for ch in file_data[1:seq_len+1]]
    X_onehot[X_int, range(seq_len)] = 1
    target_onehot[target_int, range(seq_len)] = 1

    loss = rnn_net.compute_loss(h0, X_onehot, target_onehot)
    print(loss)