
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import seaborn as sns

sns.set(style="darkgrid")


# --- Data processing ---
def load_data_mat_file_to_np_arr(path):
    mat = sio.loadmat(path)
    data = mat["A"]
    print("Loaded {} assets over {} days".format(data.shape[0], data.shape[1]))
    return data


def get_ratios(assets):
    '''

    :param assets: array of size (n_assets, prices)
    :return:
    '''
    ratios = assets[:, 1:] / assets[:, :-1]
    return ratios


def get_short_asset_revenues(assets):
    ratios = assets[:, :-1] / assets[:, 1:]
    return ratios


def get_all_revenues(assets):
    ratios = get_ratios(assets)
    short_revs = get_short_asset_revenues(assets)
    return np.concatenate((ratios, short_revs), axis=0)


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


# --- ORPS utils ---
def simplex_projection(x):
    '''
    Projection onto simplex.
    Based on https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    '''
    if np.sum(x) > 1 + 1e-6 or (x < 0).any():
        d = x.shape[0]
        u = np.sort(x)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(d) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        lamda = cssv[cond][-1] / float(rho)
        w = np.maximum(x - lamda, 0)
    else:
        w = x
    return w


def simplex_projection_wrt_matrix(x, A, D, tolerance=1e-4):
    '''
    Projection onto simplex w.r.t PD matrix A. Computes projection numerically by solving with GD:
        min_{K}{(z-x)^T A (z-x)}
    '''

    n_steps = 50  # max steps for GD to prevent infinite loop
    G = 2 * np.linalg.norm(A, 2) * D   # Lipschitz coefficient of quadratic objective
    z_next = sample_from_simplex(x.shape[0])
    z = -np.inf * np.ones_like(z_next)
    t = 1
    while np.linalg.norm(z_next - z) > tolerance and t <= n_steps:
        z = z_next
        eta = D / (G * np.sqrt(t))   # decreasing step size
        grad = 2 * np.matmul(A, z - x)  # gradient of quadratic function
        z_next = simplex_projection(z - eta * grad)     # gradient update
        t += 1
    return z_next


def find_best_portfolio_in_hindsight(ratios):
    '''
    Finds best portfolio in hindsight by numerically solving constrained problem
        min_{K}{sum(log(r_t^T x))}
    '''

    T = ratios.shape[1] + 1
    d = ratios.shape[0]
    n_steps = 200  # max steps for GD to prevent infinite loop
    # G = T * compute_lipschitz_coeff(ratios)   # Lipschitz coefficient
    G = 2   # Lipschitz coefficient
    D = np.sqrt(2)   # Diameter of simplex
    x_next = sample_from_simplex(d)
    x = -np.inf * np.ones_like(x_next)
    t = 1
    while np.linalg.norm(x_next - x) > 1e-6 and t <= n_steps:
        x = x_next
        eta = D / (G * np.sqrt(t))   # decreasing step size
        grad = np.sum([get_grad(x, r) for r in ratios.T], axis=0)
        x_next = simplex_projection(x - eta * grad)     # gradient update
        t += 1
    return x_next


def sample_from_simplex(d):
    x = np.random.rand(d)
    return simplex_projection(x)


def compute_lipschitz_coeff(ratios, norm=2):
    T = ratios.shape[-1] + 1
    bounds_per_round = np.zeros(T - 1)

    for t in range(T-1):
        rev = ratios[:, t]
        bounds_per_round[t] = np.linalg.norm(rev, ord=norm) / np.min(rev)
    return np.max(bounds_per_round)


def get_loss(x, r):
    return -np.log(np.matmul(x, r))


def get_grad(x, r):
    return -r / np.matmul(x, r)


def compute_regret(xs, ratios):
    T = len(xs) + 1
    regret = np.zeros(T-1)
    cumulative_loss = 0
    for t in range(T-1):
        cumulative_loss += get_loss(xs[t], ratios[:, t])
        x_star = find_best_portfolio_in_hindsight(ratios[:, :t + 1])
        regret[t] = cumulative_loss - np.sum([get_loss(x_star, r) for r in ratios[:, :t + 1].T])
    return regret


def find_best_asset(ratios):
    total_ratios = np.prod(ratios, axis=-1)
    return np.argmax(total_ratios)


def get_portfolio_wealth_update(x, r):
    return np.matmul(x, r)


def get_portfolio_wealth_per_round(xs, ratios):
    '''
    Computes the portfolio wealth vs. number of rounds (percentage)
    :param xs: iterates of algorithm
    :param ratios:
    :return:
    '''

    T = len(xs) + 1
    wealth = np.zeros(T)
    wealth[0] = 100     # initial wealth
    for t in range(T-1):
        wealth[t+1] = wealth[t] * get_portfolio_wealth_update(xs[t], ratios[:, t])
    return wealth - 100


def get_wealth_best_asset_per_round(ratios, W1=100):
    best_asset = find_best_asset(ratios)
    rev = ratios[best_asset]
    T = len(rev) + 1
    wealth = np.zeros(T)
    wealth[0] = W1     # initial wealth
    for t in range(T - 1):
        wealth[t + 1] = wealth[t] * rev[t]
    return wealth


def check_validity_of_solutions(xs):
    valid = True
    for x in xs:
        if sum(x) > 1.0001 or (x < 0).any():
            valid = False
    return valid


# --- Plotting ---
def plot_results(x, y, label, mode=None, **kwargs):
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


