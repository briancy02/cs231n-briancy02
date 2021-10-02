from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    

    for i in range(num_train):
      scores = X[i].dot(W)
      correct_score = scores[y[i]]
      loss += (-1*correct_score + np.log(np.exp(scores).sum()))
      for c in range(num_classes):
        adjusted_scores = scores - np.max(scores)
        softmax = np.exp(adjusted_scores) / np.sum(np.exp(adjusted_scores))
        dW[:,c] += X[i] * softmax[c]
      dW[:,y[i]] -= X[i]
    loss /= num_train
    loss += reg * (W**2).sum()
    dW /= num_train
    dW += reg * 2 * W
    

    # d(X[i].dot(W))/d(W) = X[i] therefore dL/dW = dL/df * X[i]
    # So just find dLi / dfyi for yi = j and yi != j
    # when you do chain rule on summation part, the derivative of sum(e^fj) is e^fj
    # https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function

    # Bug encountered with gradient check was that gradient check uses the loss value. the loss value was false because
    # the correct_score class was not included in summation
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    correct_score = scores[np.arange(num_train), y]
    loss = (-1 * correct_score.sum() + np.log(np.exp(scores).sum(axis=1)).sum()) / num_train + np.sum(W**2)
    stablized_scores = scores - np.max(scores, axis=1).reshape((num_train, 1))
    softmax = np.exp(stablized_scores) / np.sum(np.exp(stablized_scores), axis=1).reshape((num_train,1))
    softmax[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax)
    dW /= num_train
    dW += reg * 2 * W

    # 1.Have to sum inside log experession then sum the logs. Cant just sum the two d array
    # 2. Need to sum stablized_scores in axis=1 and then reshape to get desired softmaxes
    # 3. had same problem with stablized scores. always check shapes of values being calculated to avoid miscalculations in wrong axis
  


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
