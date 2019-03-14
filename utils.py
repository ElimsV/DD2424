import numpy as np
import copy


def softmax(mat):
    try:
        mat_exp = np.exp(mat)
        out = mat_exp / np.sum(mat_exp, axis=0)
    except FloatingPointError:
        out = np.full_like(mat, fill_value=np.finfo(float).eps)
        out[np.argmax(mat)] = 1 - (mat.size-1) * np.finfo(float).eps
    return out


def check_grad(net, X, target, delta=1e-4):
    """
    Check gradients
    :param net: RNN object
    :param X: onehot
    :param target: onehot
    :param h_prev: previous hidden state
    :param delta: for numerically calculate gradient
    :return:
    """
    print("Calculate gradients analytically!")
    grads = net.backward_pass(X, target)
    print(grads)

    print("Calculate gradients numerically!")
    paras = net.paras
    grad_nums = []
    for para_ind, para in enumerate(paras):
        grad_num = np.zeros_like(para)
        for i in range(para.shape[0]):
            for j in range(para.shape[1]):
                net_try = copy.deepcopy(net)
                net_try.paras[para_ind][i, j] -= delta
                c1 = net_try.compute_loss(X, target)
                net_try = copy.deepcopy(net)
                net_try.paras[para_ind][i, j] += delta
                c2 = net_try.compute_loss(X, target)
                grad_num[i, j] = (c2 - c1) / (2 * delta)
        grad_nums.append(grad_num)
    print(grad_nums)

    print("Checking Grads:")
    res_paras = []
    for k in range(len(grad_nums)):
        res = np.average(np.absolute(grads[k] - grad_nums[k])) / np.amax(np.absolute(grad[k]) + np.absolute(grad_nums[k]))
        res_paras.append(res)

    print("Gradient differences:")
    print("grad_b =====> ", res_paras[0])
    print("grad_c =====> ", res_paras[1])
    print("grad_U =====> ", res_paras[2])
    print("grad_W =====> ", res_paras[3])
    print("grad_V =====> ", res_paras[4])


if __name__ == '__main__':
    mat = np.asarray([[1, 2], [3, 4]])
    print(softmax(mat))
