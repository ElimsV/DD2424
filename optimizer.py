import numpy as np
import config as cfg

class AdaGrad():
    def __init__(self, eta=cfg.LEARNING_RATE, epsilon=cfg.EPSILON):
        self.eta = eta
        self.epsilon = epsilon

    def update(self, paras, grads, momentums):
        # [grad_b, grad_c, grad_U, grad_W, grad_V]
        for i, para_key in enumerate(paras):
            momentums[i] += grads[i] ** 2
            paras[para_key] -= self.eta / np.sqrt(momentums[i] + self.epsilon) * grads[i]
        return paras, momentums