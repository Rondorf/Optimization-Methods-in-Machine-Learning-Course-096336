
import os
import numpy as np
from scipy.linalg import svd, diagsvd
import matplotlib.pyplot as plt
import seaborn as sns
from gradient_descent import gradient_descent

sns.set(style="darkgrid")


def set_min_max_singular_vals(A, sigma_min, sigma_max):
    '''
        set minimal and maximal singular values for matrix A
    :param A: matrix of size (m, d)
    :param sigma_min: minimal singular value (>= 0)
    :param sigma_max: maximal singular value (>= max(0, sigma_min))
    :return:
    '''

    assert sigma_min >= 0 and sigma_max >= max(0, sigma_min), 'singular values are either negative or invalid.'

    m, d = A.shape
    U, S, VT = svd(A)   # decompose A
    s_min, s_max = S[-1], S[0]  # singular values are in descending order
    # shift and scale singular values to be in range [sigma_min, sigma_max]
    S = ((S - s_min) / (s_max - s_min)) * (sigma_max - sigma_min) + sigma_min
    S_full = diagsvd(S, m, d)    # create full S matrix

    return U.dot(S_full).dot(VT)


def l2_ball_proj(x, r):
    if np.linalg.norm(x) <= r:
        return x
    else:
        return r * (x / np.linalg.norm(x))


def sample_uniform_ball(r, dim):
    '''
        sample point uniformly in the L2 ball of dim n and radius r
    :param r:
    :param n:
    :return:
    '''
    a = np.random.randn(dim)  # draw random normal coordinates
    a = a / np.linalg.norm(a)   # normalize to have unit norm
    a *= np.random.uniform(0, r)   # multiply by random radius which is smaller than r
    return a


def get_optimal_value(LS_oracle, args):
    if LS_oracle.get_strong_convexity() < 1e-6:     # if A^T A is not invertible - get f* by GD
        x0 = sample_uniform_ball(args.r, args.d)
        beta = LS_oracle.get_smoothness()
        gradient_iterates = gradient_descent(x0, grad_func=LS_oracle.get_grad, step_size=1 / beta,
                                             max_steps=10 * args.num_grad_steps,
                                             proj=lambda x: l2_ball_proj(x, args.R))
        return LS_oracle.get_value(gradient_iterates[-1])
    else:
        return LS_oracle.get_value(LS_oracle.get_analytic_solution())


def load_experiments(dir):
    '''

    :param dir: directory of output files
    :return: arrays of data -- rows are experiments, columns are results
    '''
    experiments = os.listdir(dir)
    array = []
    for exp in experiments:
        array.append(np.load(os.path.join(dir, exp)))

    return np.vstack(array)


def plot_convergence_curve(x, y, label, mode=None, **kwargs):
    """
    Takes as input an x-value (number of frames)
    and a matrix of y-values (rows: runs, columns: results)
    """

    y = y[:, :len(x)]

    # get the mean (only where we have data) and compute moving average
    mean = np.sum(y, axis=0) / (np.sum(y != 0, axis=0))
    p = plt.plot(x, mean, linewidth=2, label=label, **kwargs)

    if mode == 'std':
        # compute standard deviation
        std = np.std(y, axis=0)
        # set interval
        interval = [mean - std, mean + std]
    elif mode == 'ci':
        # compute confidence interval
        ci = 1.96 * np.std(y) / np.mean(y)
        # set intervals
        interval = [mean - ci, mean + ci]
    else:
        return

    plt.gca().fill_between(x, interval[0], interval[1],
                           facecolor=p[0].get_color(), alpha=0.3)




