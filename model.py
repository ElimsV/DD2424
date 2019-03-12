import numpy as np

class RNN():
    def __init__(self, K):
        # Hyper-parameters
        self.m = 100  # hidden state
        self.eta = 0.1  # learning rate
        self.seq_length = 25
        self.K = K
        self.sig = 0.01

        # Network parameters
        self.b = np.zeros([self.K, 1])
        self.c = np.zeros([self.K, 1])
        self.U = np.random.standard_normal(size=[self.m, self.K]) * self.sig
        self.W = np.random.standard_normal(size=[self.m, self.m]) * self.sig
        self.V = np.random.standard_normal(size=[self.K, self.m]) * self.sig

    def forward_pass(self):
        