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
        return h, o, p

    def generate_chars(self, h0, x0, n):
        """
        Generate text of length n
        :param h0: init hidden state
        :param x0: init one-hot input
        :param n: output length
        :return:
        """
        count = 0
        h_prev = h0
        x = x0

        int_list = []
        while count < n:
            h_prev, o, p = self.forward_pass(h_prev, x)
            int_max = np.argmax(p)
            int_list.append(int_max)

            # update next x
            ind = np.random.choice(self.K, 1, p=p.reshape(self.K), replace=False)
            x = np.zeros([self.K, 1])
            x[ind] = 1

            count += 1

        assert len(int_list) == n, "Erroneous output length!"
        return int_list


if __name__ == "__main__":
    path = "./goblet.txt"
    data_loader = Load_Data(path)
    file_data = data_loader.load_data()[0]  # load_data returns a tuple of 3 elements

    char2int = data_loader.char_to_int()  # dict char to int
    int2char = data_loader.int_to_char()  # dict int to char
    
    ######test generation chars######
    # init input char
    c0 = 'a'
    int_0 = char2int[c0]
    x0 = np.zeros([data_loader.K, 1])
    x0[int_0] = 1

    # init rnn network
    rnn_net = RNN(data_loader.K)

    # init hidden state
    h0 = np.random.standard_normal([rnn_net.m, 1])

    # output length
    n = 20

    # generate text
    int_list = rnn_net.generate_chars(h0, x0, n)
    text = ''
    for i in int_list:
        text = text + int2char[i]
    print(text)
