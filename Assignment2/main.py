
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from gradient_descent import *
from objective import OracleLS
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=200, help="Number of data points")
parser.add_argument("--d", type=int, default=10, help="Dimension (number of features)")
parser.add_argument("--sigma-min", type=int, default=0.2, help="Minimal singular value of matrix A")
parser.add_argument("--sigma-max", type=int, default=2, help="Maximal singular value of matrix A")
parser.add_argument("--mini-batch", nargs='+', default=[5, 10, 100], help="Mini-batch sizes for mini-batch SGD")
parser.add_argument("--r", type=float, default=1, help="Radius of L2 ball to draw solution x* from")
parser.add_argument("--noise-var", type=float, default=0.001, help="Variance of noise added to Ax*")
parser.add_argument("--num-grad-steps", type=int, default=500, help="Number of gradient steps")
parser.add_argument("--seed", type=int, default=17, help="Random seed")
parser.add_argument("--num-experiments", type=int, default=25, help="Number of experiments for averaging")
parser.add_argument("--metric", type=str, default='opt_gap', help="one of: 'value' (f(x)), 'opt_gap' (f(x) - f*)")
parser.add_argument("--save-results", type=bool, default=False, help="Whether to save results")
parser.add_argument("--save-image", type=bool, default=False, help="Whether to save convergence image")
parser.add_argument("--image-filename", type=str, default='opt_gap', help="If save-image=True")
parser.add_argument("--data-path", type=str, default='outputs/', help="Directory to save/load data from")
parser.add_argument("--run-experiments", type=bool, default=False,
                    help="whether to run experiments or use existing ones")

args = parser.parse_args()


def experiment(seed):
    np.random.seed(seed)    # set random seed to reproduce results

    # set problem parameters
    A = np.random.randn(args.m, args.d)     # draw random matrix
    A = set_min_max_singular_vals(A, args.sigma_min, args.sigma_max)  # set min and max singular values
    x_star = sample_uniform_ball(args.r, args.d)  # draw solution
    noise = np.sqrt(args.noise_var) * np.random.randn(args.m)     # draw random noise
    b = A.dot(x_star) + noise

    # create Least Squares oracle
    LS_oracle = OracleLS(A, b)
    R = 5 * args.r      # radius of L2 ball to search optimal solution
    # get optimal value - either analytically by grad=0 or empirically by performing GD
    if args.metric == 'opt_gap':
        optimal_value = get_optimal_value(LS_oracle, args)
    # initial iterate (sampled in l2 ball of radius R)
    x0 = sample_uniform_ball(R, args.d)

    # --- SGD ---
    alpha = LS_oracle.get_strong_convexity()
    beta = LS_oracle.get_smoothness()
    sgd_iterates = stochastic_gradient_descent(x0, so=lambda x: LS_oracle.get_stochastic_grad(x, batch_size=1),
                                               num_steps=args.num_grad_steps,
                                               batch_size=1, m=args.m, alpha=alpha,
                                               proj=lambda x: l2_ball_proj(x, R))
    # --- Mini-batch SGD ---
    # mini_batch_sizes = [5, 10, 100]
    mb_sgd_iterates_list = []
    for k in args.mini_batch:
        mb_sgd_iterates = stochastic_gradient_descent(x0,
                                                      so=lambda x: LS_oracle.get_stochastic_grad(x, batch_size=k),
                                                      num_steps=args.num_grad_steps,
                                                      batch_size=k, m=args.m, alpha=alpha,
                                                      proj=lambda x: l2_ball_proj(x, R))
        mb_sgd_iterates_list.append(mb_sgd_iterates)

    # --- SVRG ---
    svrg_iterates = svrg(x0, alpha, beta, args.m,
                         stochastic_grad_func=lambda x, idx: LS_oracle.get_stochastic_grad_by_index(x, idx),
                         grad_func=LS_oracle.get_grad, max_steps=args.num_grad_steps,
                         proj=lambda x: l2_ball_proj(x, R))

    # get metric values (function value over time or optimality gap over time)
    if args.metric == 'value':
        sgd_values = np.array([LS_oracle.get_value(x) for x in sgd_iterates])
        mb_sgd_values_list = [np.array([LS_oracle.get_value(x) for x in mb_sgd_iterates])
                              for mb_sgd_iterates in mb_sgd_iterates_list]
        svrg_values = np.array([LS_oracle.get_value(x) for x in svrg_iterates])

    elif args.metric == 'opt_gap':
        sgd_values = np.array([LS_oracle.get_value(x) - optimal_value for x in sgd_iterates])
        mb_sgd_values_list = [np.array([LS_oracle.get_value(x) - optimal_value for x in mb_sgd_iterates])
                              for mb_sgd_iterates in mb_sgd_iterates_list]
        svrg_values = np.array([LS_oracle.get_value(x) - optimal_value for x in svrg_iterates])
    else:
        raise NotImplementedError

    if args.save_results:
        # create directories if doesn't exists
        os.makedirs(os.path.join(args.data_path, 'sgd/'), exist_ok=True)
        for k in args.mini_batch:
            os.makedirs(os.path.join(args.data_path, 'mb_sgd_{}/'.format(k)), exist_ok=True)
        os.makedirs(os.path.join(args.data_path, 'svrg/'), exist_ok=True)
        # save data arrays
        np.save(os.path.join(args.data_path, 'sgd/seed__{}'.format(seed)), sgd_values)
        for (k, mb_sgd_values) in zip(args.mini_batch, mb_sgd_values_list):
            np.save(os.path.join(args.data_path, 'mb_sgd_{}/seed__{}'.format(k, seed)),
                    mb_sgd_values)
        np.save(os.path.join(args.data_path, 'svrg/seed__{}'.format(seed)), svrg_values)


if __name__ == '__main__':
    if args.run_experiments:
        for exp in range(args.num_experiments):
            experiment(seed=args.seed + exp)

    try:
        sgd_array = load_experiments(os.path.join(args.data_path, 'sgd/'))
        mb_sgd_arrays = []
        for k in args.mini_batch:
            mb_sgd_array = load_experiments(os.path.join(args.data_path, 'mb_sgd_{}/'.format(k)))
            mb_sgd_arrays.append(mb_sgd_array)
        svrg_array = load_experiments(os.path.join(args.data_path, 'svrg/'))
    except OSError:
        print('Data doesn\'t exists. Change save_results flag to True and try again (consider mini-batch size).')

    plot_convergence_curve(range(sgd_array.shape[1]), sgd_array, label='SGD')
    for k, mb_sgd_array in zip(args.mini_batch, mb_sgd_arrays):
        plot_convergence_curve(range(mb_sgd_array.shape[1]), mb_sgd_array,
                               label='Mini-batch SGD (k={})'.format(k))
    plot_convergence_curve(range(svrg_array.shape[1]), svrg_array, label='SVRG')

    plt.xlabel('Iterations')
    if args.metric == 'value':
        plt.ylabel('Objective value ')
    elif args.metric == 'opt_gap':
        plt.ylabel('Optimality gap $(f -f^*)$')
        plt.yscale('log')
    plt.legend()

    if args.save_image:
        plt.savefig(os.path.join(args.data_path, args.image_filename))
    plt.show()
