import numpy as np
from data_preprocess import char2ind, ind2char, Load_Data
from utils import softmax

class RNN():
    def __init__(self, K, h_prev, batch_size):
        """
        Build a RNN
        Cannot handle inputs size smaller than batch size
        :param K: scalar, encoding dim
        :param h_prev: m*1 2d array
        :param batch_size: scalar
        """
        # Hyper-parameters
        self.m = h_prev.shape[0]  # hidden state
        self.eta = 0.1  # learning rate
        self.seq_length = batch_size
        self.K = K
        self.sig = 0.01

        # Network parameters
        self.b = np.zeros([self.m, 1])
        self.c = np.zeros([self.K, 1])
        self.U = np.random.standard_normal(size=[self.m, self.K]) * self.sig
        self.W = np.random.standard_normal(size=[self.m, self.m]) * self.sig
        self.V = np.random.standard_normal(size=[self.K, self.m]) * self.sig

        # Intent variables for backward pass
        self.Y = np.zeros([self.K, self.seq_length])
        self.prob = np.zeros([self.K, self.seq_length])
        self.h_prev = h_prev  # 2d array
        self.h = np.zeros([self.m, self.seq_length])
        self.a = np.zeros([self.m, self.seq_length])

        # gradients
        self.grad_o = np.zeros([self.K, self.seq_length])
        self.grad_a = np.zeros([self.m, self.seq_length])
        self.grad_h = np.zeros([self.m, self.seq_length])
        self.grad_b = np.zeros_like(self.b)
        self.grad_c = np.zeros_like(self.c)
        self.grad_U = np.zeros_like(self.U)
        self.grad_W = np.zeros_like(self.W)
        self.grad_V = np.zeros_like(self.V)



    def forward_pass(self, h_prev, x):
        # x: K*1
        # h: m*1
        # o: K*1
        # p: K*1
        a = np.dot(self.W, h_prev) + np.dot(self.U, x) + self.b
        h = np.tanh(a)
        o = np.dot(self.V, h) + self.c
        p = softmax(o)
        return h, p, a

    def backward_pass(self, X, target):
        """
        Calculate grads
        :param X: onehot
        :param target: onehot
        :return:
        """
        self.grad_o = self.prob - target
        self.grad_c = np.sum(self.grad_o, axis=1)
        self.grad_V = np.dot(self.grad_o, self.h.T)

        for t in range(self.seq_length)[::-1]:
            if t == self.seq_length - 1:
                self.grad_h[:, t] = self.grad_o[:, t].dot(self.V)
            else:
                self.grad_h[:, t] = self.grad_o[:, t].dot(self.V) + self.grad_a[:, t+1].dot(self.W)
            self.grad_a[:, t] = self.grad_h[:, t].dot(np.diag(1 - self.h[:, t] ** 2))

        h_tmp = np.hstack((self.h_prev, self.h[:, :-1]))
        self.grad_W = self.grad_a.dot(h_tmp.T)
        self.grad_U = self.grad_a.dot(X.T)
        self.grad_b = np.sum(self.grad_a, axis=1)
        return self.grad_b, self.grad_c, self.grad_U, self.grad_W, self.grad_V

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
            h_prev, p, _= self.forward_pass(h_prev, x)
            int_max = np.argmax(p)
            int_list.append(int_max)

            # update next x
            ind = np.random.choice(self.K, 1, p=p.reshape(self.K), replace=False)
            x = np.zeros([self.K, 1])
            x[ind] = 1

            count += 1

        assert len(int_list) == batch_size, "Erroneous output length!"
        return int_list

    def predict_prob(self, X):
        """
        predict prob output
        :param X: K*seq_len
        :return prob: K*seq_len, prob output
        """
        h_prev = self.h_prev
        for t in range(X.shape[1]):
            h_prev, p, a = self.forward_pass(h_prev, X[:, t].reshape([self.K, 1]))
            self.prob[:, t] = p.reshape(self.K)  # probability output
            self.h[:, t] = h_prev.reshape(self.m)
            self.a[:, t] = a.reshape(self.m)
        return self.prob

    def predict_onehot(self, X):
        """
        predict onehot output
        :param X: K*seq_len
        :return Y: K*seq_len, one-hot output
        """
        self.prob = self.predict_prob(X)
        int_max = np.argmax(self.prob, axis=0)
        self.Y[int_max, range(self.prob.shape[1])] = 1  # onehot output
        return self.Y

    def compute_loss(self, X, target):
        assert X.shape == target.shape, "X and target shape error!"
        self.prob = self.predict_prob(X)
        tmp = np.sum(self.prob * target, axis=0)
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
    # init hidden state
    # h0 = np.random.standard_normal([rnn_net.m, 1])
    h0 = np.zeros([100, 1])
    seq_len = 25
    # init rnn network
    rnn_net = RNN(data_loader.K, h0, seq_len)

    X_onehot = np.zeros([rnn_net.K, seq_len])
    target_onehot = np.zeros([rnn_net.K, seq_len])
    X_int = [char2int[ch] for ch in file_data[:seq_len]]
    target_int = [char2int[ch] for ch in file_data[1:seq_len+1]]
    X_onehot[X_int, range(seq_len)] = 1
    target_onehot[target_int, range(seq_len)] = 1

    loss = rnn_net.compute_loss(X_onehot, target_onehot)
    print(loss)

    ######test gradient computation######
    # grad_b, grad_c, grad_U, grad_W, grad_V
    grads = rnn_net.backward_pass(X_onehot, target_onehot)
    for grad in grads:
        print(grad)