import random
import numpy as np
from data_preprocess import char2ind, ind2char, Load_Data
from utils import softmax, check_grad
from optimizer import AdaGrad
import config as cfg

class RNN:
    def __init__(self, K, h_prev, batch_size, unique_chars):
        """
        Build a RNN
        Cannot handle inputs size smaller than batch size
        :param K: scalar, encoding dim
        :param h_prev: m*1 2d array
        :param batch_size: scalar
        """
        # Data characteristics
        self.unique_chars = unique_chars

        # Hyper-parameters
        self.m = h_prev.shape[0]  # hidden state
        self.eta = cfg.LEARNING_RATE  # learning rate
        self.epsilon = cfg.EPSILON
        self.seq_length = batch_size
        self.K = K
        self.sig = cfg.SIG

        # Network parameters
        b = np.zeros([self.m, 1])
        c = np.zeros([self.K, 1])
        U = np.random.standard_normal(size=[self.m, self.K]) * self.sig
        W = np.random.standard_normal(size=[self.m, self.m]) * self.sig
        V = np.random.standard_normal(size=[self.K, self.m]) * self.sig
        self.paras = {"b": b, "c": c, "U": U, "W": W, "V": V}

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
        a = np.dot(self.paras["W"], h_prev) + np.dot(self.paras["U"], x) + self.paras["b"]
        h = np.tanh(a)
        o = np.dot(self.paras["V"], h) + self.paras["c"]
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
        # grad_b = np.zeros_like(self.paras["b"])
        # grad_c = np.zeros_like(self.paras["c"])
        # grad_U = np.zeros_like(self.paras["U"])
        # grad_W = np.zeros_like(self.paras["W"])
        # grad_V = np.zeros_like(self.paras["V"])
        
        prob = self.predict_prob(X)
        loss = self.compute_loss(prob, target)

        grad_o = prob - target
        grad_c = np.sum(grad_o, axis=1).reshape(self.paras["c"].shape)
        grad_V = np.dot(grad_o, self.h.T)

        for t in range(self.seq_length)[::-1]:
            if t == self.seq_length - 1:
                grad_h[:, t] = grad_o[:, t].dot(self.paras["V"])
            else:
                grad_h[:, t] = grad_o[:, t].dot(self.paras["V"]) + grad_a[:, t+1].dot(self.paras["W"])
            grad_a[:, t] = grad_h[:, t].dot(np.diag(1 - self.h[:, t] ** 2))

        h_tmp = np.hstack((self.h_prev, self.h[:, :-1]))
        grad_W = grad_a.dot(h_tmp.T)
        grad_U = grad_a.dot(X.T)
        grad_b = np.sum(grad_a, axis=1).reshape(self.paras["b"].shape)

        # gradient clip
        grads = [grad_b, grad_c, grad_U, grad_W, grad_V]
        grads_clip = []
        for grad in grads:
            grad_clip = np.maximum(np.minimum(grad, 5), -5)
            grads_clip.append(grad_clip)
        return grads_clip, loss  # grad_b, grad_c, grad_U, grad_W, grad_V

    def generate_chars(self, h0, x0, text_len):
        """
        Generate text of length text_len
        :param h0: init hidden state
        :param x0: init one-hot input
        :param text_len: output length
        :return:
        """
        count = 0
        h_prev = h0
        x = x0

        int_list = []
        while count < text_len:
            h_prev, p, _ = self.forward_pass(h_prev, x)
            int_max = np.argmax(p)
            int_list.append(int_max)

            # update next x
            ind = np.random.choice(self.K, 1, p=p.reshape(self.K), replace=False)
            # ind = int_max
            x = np.zeros([self.K, 1])
            x[ind] = 1

            count += 1

        assert len(int_list) == text_len, "Erroneous output length!"
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

    def compute_loss(self, prob, target):
        tmp = np.sum(prob * target, axis=0)
        tmp[tmp == 0] = np.finfo(float).eps
        return - np.sum(np.log(tmp))

    def train(self, X_all, target_all, h_prev, int2char, char2int, epoch_num=cfg.EPOCH, batch_size=cfg.BATCH_SIZE):
        # check validity
        assert X_all.shape[1] == target_all.shape[1], "X and target length mismatch!"
        dataset_size = X_all.shape[1]
        assert dataset_size > batch_size, "Batch size larger than dataset size!"
        # initialize
        pt_start = 0
        epoch = 1
        iteration = 1
        opt = AdaGrad(eta=self.eta, epsilon=self.epsilon)
        momentums = [np.zeros_like(self.paras[p]) for p in self.paras]
        smooth_loss = - np.log(1 / self.K) * self.seq_length
        smooth_loss_acc = [smooth_loss]

        print("*" * 30, " Starting epoch {}/{} ".format(epoch, epoch_num), "*" * 30)
        while epoch <= epoch_num:
            # synthesize short text
            if iteration % cfg.SYN_STEP == 1 and iteration < cfg.SYN_BOUND:
                text = ''
                x0 = np.zeros([self.K, 1])
                x0[random.randint(0, self.K - 1)] = 1
                text_ints = self.generate_chars(self.h_prev, x0, cfg.SHORT_TEXT_LENGTH)
                for text_int in text_ints:
                    text += int2char[text_int]
                print("Synthesized text before {} iteration:".format(iteration))
                print(text)

            # set up batch for training
            pt_end = (pt_start + batch_size) % dataset_size
            if pt_start < pt_end:
                batch_ind = list(range(pt_start, pt_end))
            else:
                batch_ind = list(range(pt_start, dataset_size))+list(range(pt_end))
                assert len(batch_ind) == batch_size, "Batch indexing error! Iterate to the end."
                epoch += 1
                if epoch > epoch_num:
                    break
                print("*" * 30, " Starting epoch {}/{} ".format(epoch, epoch_num), "*" * 30)
                # reset h_prev
                self.h_prev = h_prev
                # TODO: reset iteration
            X_batch = X_all[:, batch_ind]
            target_batch = target_all[:, batch_ind]

            # execute batch training
            grads, loss = self.backward_pass(X_batch, target_batch)
            self.paras, momentums = opt.update(self.paras, grads, momentums)

            # update h_prev
            self.h_prev = self.h[:, -1].reshape(self.h_prev.shape)

            # record loss
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            if iteration % 100 == 0:
                print("Iteration {}      Current smoothed loss: {}".format(iteration, smooth_loss))
                smooth_loss_acc.append(smooth_loss)

            # update counters
            pt_start = pt_end
            iteration += 1

        print("*"*30, " Finished training! ", "*"*30)

        # synthesize a passage with the last model
        passage = ''
        x0 = np.zeros([self.K, 1])
        x0[char2int[' ']] = 1
        passage_ints = self.generate_chars(self.h_prev, x0, cfg.PASSAGE_LENGTH)
        for passage_int in passage_ints:
            passage += int2char[passage_int]
        print("Synthesized passage with the finest model:")
        print(passage)

        return smooth_loss_acc


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
    rnn_net = RNN(data_loader.K, h0, seq_len, data_loader.unique_chars)

    X_onehot = np.zeros([rnn_net.K, seq_len])
    target_onehot = np.zeros([rnn_net.K, seq_len])
    X_int = [char2int[ch] for ch in file_data[:seq_len]]
    target_int = [char2int[ch] for ch in file_data[1:seq_len+1]]
    X_onehot[X_int, range(seq_len)] = 1
    target_onehot[target_int, range(seq_len)] = 1

    # prob = rnn_net.predict_prob(X_onehot)
    # loss = rnn_net.compute_loss(prob, target_onehot)
    # print(loss)

    ######test gradient computation######
    # grads, loss = rnn_net.backward_pass(X_onehot, target_onehot)
    # for grad in grads:
    #     print(grad)

    check_grad(rnn_net, X_onehot, target_onehot)
