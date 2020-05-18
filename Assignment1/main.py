
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from gradient_descent import *
from objective import OracleLS
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=10, help="Number of data points")
parser.add_argument("--d", type=int, default=5, help="Dimension (number of features)")
parser.add_argument("--sigma-min", type=int, default=1, help="Minimal singular value of matrix A")
parser.add_argument("--sigma-max", type=int, default=3, help="Maximal singular value of matrix A")
parser.add_argument("--r", type=float, default=2, help="Radius of L2 ball to draw solution x* from")
parser.add_argument("--noise-var", type=float, default=0.01, help="Variance of noise added to Ax*")
parser.add_argument("--num-grad-steps", type=int, default=50, help="Number of gradient steps")
parser.add_argument("--seed", type=int, default=17, help="Random seed")
parser.add_argument("--num-experiments", type=int, default=25, help="Random seed")
parser.add_argument("--metric", type=str, default='opt_gap', help="one of: 'value' (f(x)), 'opt_gap' (f(x) - f*)")
parser.add_argument("--save-results", type=bool, default=True, help="Whether to save results")
parser.add_argument("--save-image", type=bool, default=False, help="Whether to save convergence image")
parser.add_argument("--image-filename", type=str, default='value_plot_PD', help="If save-image=True")
parser.add_argument("--data-path", type=str, default='outputs/', help="Directory to save/load data from")
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

    # --- "Non-smooth" sub-gradient method ---
    L = LS_oracle.get_lipschitzness(R)
    subgrad_iterates = gradient_descent(x0, grad_func=LS_oracle.get_grad, step_size='diminishing',
                                        max_steps=args.num_grad_steps, proj=lambda x: l2_ball_proj(x, R),
                                        diam=R, Lipschitz=L)
    # --- Smooth gradient method ---
    beta = LS_oracle.get_smoothness()
    gradient_iterates = gradient_descent(x0, grad_func=LS_oracle.get_grad, step_size=1/beta,
                                         max_steps=args.num_grad_steps, proj=lambda x: l2_ball_proj(x, R))
    # --- Accelerated gradient method ---
    acc_gradient_iterates = accelerated_gradient_descent(x0, grad_func=LS_oracle.get_grad,
                                                         step_size=1/beta, max_steps=args.num_grad_steps,
                                                         proj=lambda x: l2_ball_proj(x, R))

    # get metric values (function value over time or optimality gap over time)
    if args.metric == 'value':
        subgrad_values = np.array([LS_oracle.get_value(x) for x in subgrad_iterates])
        gradient_values = np.array([LS_oracle.get_value(x) for x in gradient_iterates])
        acc_gradient_values = np.array([LS_oracle.get_value(x) for x in acc_gradient_iterates])
    elif args.metric == 'opt_gap':
        subgrad_values = np.array([LS_oracle.get_value(x) - optimal_value for x in subgrad_iterates])
        gradient_values = np.array([LS_oracle.get_value(x) - optimal_value for x in gradient_iterates])
        acc_gradient_values = np.array([LS_oracle.get_value(x) - optimal_value for x in acc_gradient_iterates])
    else:
        raise NotImplementedError

    if args.save_results:
        # create directories if doesn't exists
        os.makedirs(os.path.join(args.data_path, 'sub_gd/'), exist_ok=True)
        os.makedirs(os.path.join(args.data_path, 'gd/'), exist_ok=True)
        os.makedirs(os.path.join(args.data_path, 'agd/'), exist_ok=True)
        # save data arrays
        np.save(os.path.join(args.data_path, 'sub_gd/seed__{}'.format(seed)), subgrad_values)
        np.save(os.path.join(args.data_path, 'gd/seed__{}'.format(seed)), gradient_values)
        np.save(os.path.join(args.data_path, 'agd/seed__{}'.format(seed)), acc_gradient_values)


if __name__ == '__main__':
    for exp in range(args.num_experiments):
        experiment(seed=args.seed + exp)

    try:
        subgrad_array = load_experiments(os.path.join(args.data_path, 'sub_gd/'))
        gradient_array = load_experiments(os.path.join(args.data_path, 'gd/'))
        acc_gradient_array = load_experiments(os.path.join(args.data_path, 'agd/'))
    except OSError:
        print('Data doesn\'t exists. Change save_results flag to True and try again.')

    plot_convergence_curve(range(subgrad_array.shape[1]), subgrad_array, label='Sub-GD')
    plot_convergence_curve(range(gradient_array.shape[1]), gradient_array, label='GD')
    plot_convergence_curve(range(acc_gradient_array.shape[1]), acc_gradient_array, label='AGD')

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
