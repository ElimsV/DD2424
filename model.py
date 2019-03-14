import numpy as np
from data_preprocess import char2ind, ind2char, Load_Data
from utils import softmax, check_grad
import config as cfg

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
        self.eta = cfg.LEARNING_RATE  # learning rate
        self.seq_length = batch_size
        self.K = K
        self.sig = cfg.SIG

        # Network parameters
        self.b = np.zeros([self.m, 1])
        self.c = np.zeros([self.K, 1])
        self.U = np.random.standard_normal(size=[self.m, self.K]) * self.sig
        self.W = np.random.standard_normal(size=[self.m, self.m]) * self.sig
        self.V = np.random.standard_normal(size=[self.K, self.m]) * self.sig
        self.paras = [self.b, self.c, self.U, self.W, self.V]

        # Intent variables for backward pass
        self.Y = np.zeros([self.K, self.seq_length])
        self.h_prev = h_prev  # 2d array
        self.h = np.zeros([self.m, self.seq_length])
        self.a = np.zeros([self.m, self.seq_length])

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
        assert X.shape[1] == target.shape[1], "X and target seq_len error!"
        seq_length = X.shape[1]
        # grad_o = np.zeros([self.K, seq_length])
        grad_a = np.zeros([self.m, seq_length])
        grad_h = np.zeros([self.m, seq_length])
        # grad_b = np.zeros_like(self.b)
        # grad_c = np.zeros_like(self.c)
        # grad_U = np.zeros_like(self.U)
        # grad_W = np.zeros_like(self.W)
        # grad_V = np.zeros_like(self.V)
        
        prob = self.predict_prob(X)
        grad_o = prob - target
        grad_c = np.sum(grad_o, axis=1).reshape(self.c.shape)
        grad_V = np.dot(grad_o, self.h.T)

        for t in range(self.seq_length)[::-1]:
            if t == self.seq_length - 1:
                grad_h[:, t] = grad_o[:, t].dot(self.V)
            else:
                grad_h[:, t] = grad_o[:, t].dot(self.V) + grad_a[:, t+1].dot(self.W)
            grad_a[:, t] = grad_h[:, t].dot(np.diag(1 - self.h[:, t] ** 2))

        h_tmp = np.hstack((self.h_prev, self.h[:, :-1]))
        grad_W = grad_a.dot(h_tmp.T)
        grad_U = grad_a.dot(X.T)
        grad_b = np.sum(grad_a, axis=1).reshape(self.b.shape)

        # gradient clip
        grads = [grad_b, grad_c, grad_U, grad_W, grad_V]
        grads_clip = []
        for grad in grads:
            grad_clip = np.maximum(np.minimum(grad, 5), -5)
            grads_clip.append(grad_clip)
        return grads_clip  # grad_b, grad_c, grad_U, grad_W, grad_V

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
        prob = np.zeros([self.K, self.seq_length])
        for t in range(X.shape[1]):
            h_prev, p, a = self.forward_pass(h_prev, X[:, t].reshape([self.K, 1]))
            prob[:, t] = p.reshape(self.K)  # probability output
            self.h[:, t] = h_prev.reshape(self.m)
            self.a[:, t] = a.reshape(self.m)
        return prob

    def predict_onehot(self, X):
        """
        predict onehot output
        :param X: K*seq_len
        :return Y: K*seq_len, one-hot output
        """
        prob = self.predict_prob(X)
        int_max = np.argmax(prob, axis=0)
        self.Y[int_max, range(prob.shape[1])] = 1  # onehot output
        return self.Y

    def compute_loss(self, X, target):
        assert X.shape == target.shape, "X and target shape error!"
        prob = self.predict_prob(X)
        tmp = np.sum(prob * target, axis=0)
        tmp[tmp == 0] = np.finfo(float).eps
        return - np.sum(np.log(tmp))


if __name__ == "__main__":
    path = "./data/goblet.txt"
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
    h0 = np.zeros([5, 1])
    seq_len = 25
    # init rnn network
    rnn_net = RNN(data_loader.K, h0, seq_len)

    X_onehot = np.zeros([rnn_net.K, seq_len])
    target_onehot = np.zeros([rnn_net.K, seq_len])
    X_int = [char2int[ch] for ch in file_data[:seq_len]]
    target_int = [char2int[ch] for ch in file_data[1:seq_len+1]]
    X_onehot[X_int, range(seq_len)] = 1
    target_onehot[target_int, range(seq_len)] = 1

    # loss = rnn_net.compute_loss(X_onehot, target_onehot)
    # print(loss)

    ######test gradient computation######
    # grads = rnn_net.backward_pass(X_onehot, target_onehot)
    # for grad in grads:
    #     print(grad)

    check_grad(rnn_net, X_onehot, target_onehot)
