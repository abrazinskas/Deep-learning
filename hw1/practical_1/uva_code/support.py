import numpy as np

# generatilized L2 norm for matrices, assuming that # of rows are # of features
def L2(W):
    vec_norms = np.sum(W**2, axis=0)
    return np.sum(vec_norms)

# softmax function
def sm(x):
    num = np.exp(x)
    return num / np.sum(num, axis=1, keepdims=True)


# cross entropy loss
# y is a vector with labels, y_hat is a matrix
def total_ce_loss(y, y_hat):
    m = y.shape[0]
    return - np.sum(np.log(y_hat[range(m), y]))/float(m)

# # single data point ce loss
# # where y assumed to be a the correct label
# def ce_loss(y, y_hat):
#     return np.log(y_hat[y])

# wrt to the input to the softmax
def ce_loss_grad(y, y_hat):
    grad = - y_hat
    grad[y] += 1
    return grad


def total_ce_loss_grad(y, y_hat):
    k = y_hat.shape[1]  # number of classes
    m = y_hat.shape[0]  # size of the mini-batch
    counts = np.zeros((k,))
    for i in range(m):
        counts[y[i]] += 1
    pred = np.sum(y_hat, axis=0)
    return - (counts - pred)/float(m)


# def dce_loss_dx(x, y):
#     k = np.unique(y) # number of classes
#     m = float(y.shape[0])
#     pred = sm(x) # [m x k]
#     # prepare I
#     I = np.zeros((m, k))
#     for i in range(k):
#         I[:,i] = y==i
#     temp = I - pred
#     # compute gradient
#     dW = - temp)/m
#     db = - np.sum(temp, axis=0)/m
#     return dW, db