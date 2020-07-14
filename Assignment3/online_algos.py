
import numpy as np
import utils as utl


def online_gradient_descent(batch_data, T, init, G, D):
    '''
    :param batch_data: ratios data of size (n_ratios, T)
    :param T: Horizon of repeated game
    :param init: Initial point x_1
    :param G: Lipschitz coeeficient
    :param D: Diameter
    :return:
    '''

    xs = [init]
    eta = D / (G * np.sqrt(T))  # fixed step size
    rt = batch_data[:, 0]   # initial ratios

    for t in range(T-1):
        # adapting step size
        # eta = D / (G * np.sqrt(t+1))
        # compute online gradient
        grad = utl.get_grad(xs[-1], rt)
        # perform OGD update (with projection)
        xs.append(utl.simplex_projection(xs[-1] - eta * grad))
        # observe next data point
        rt = batch_data[:, t + 1]
    return xs


def online_exponential_gradient(batch_data, T, init, G, D):
    '''
    Similar to OGD with only difference in update rule
    '''

    xs = [init]
    eta = D / (G * np.sqrt(2 * T))  # fixed step size
    rt = batch_data[:, 0]  # initial ratios

    for t in range(T - 1):
        # compute online gradient
        grad = utl.get_grad(xs[-1], rt)
        # perform OEG update (softmax)
        xt = xs[-1] * np.exp(-eta * grad) / np.sum(xs[-1] * np.exp(-eta * grad))
        # add to iterates (no need for projection as xt in the simplex)
        xs.append(xt)
        # observe next data point
        rt = batch_data[:, t + 1]
    return xs


def online_newton_step(batch_data, T, init, G, D, alpha=1):
    '''
    Most parameters are similar to OGD, and the rest are
     :param alpha: Strong convexity parameter
     '''

    n = init.shape[0]   # dimensionality
    xs = [init]
    gamma = 0.5 * min(1 / (4 * G * D), alpha)
    epsilon = 1 / ((gamma * D) ** 2)
    eta = 1 / gamma  # fixed step size
    A = epsilon * np.identity(n)   # initial matrix

    rt = batch_data[:, 0]  # initial ratios
    for t in range(T - 1):
        # compute online gradient
        grad = utl.get_grad(xs[-1], rt)
        # Rank-1 update
        A += np.outer(grad, grad)
        # weighted gradient update
        y = xs[-1] - eta * np.matmul(np.linalg.inv(A), grad)
        # project w.r.t A
        xs.append(utl.simplex_projection_wrt_matrix(y, A, D))
        # observe next data point
        rt = batch_data[:, t + 1]
    return xs
