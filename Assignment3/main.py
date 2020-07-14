
import os
import numpy as np
import matplotlib.pyplot as plt
import utils as utl
from online_algos import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=17, help="Random seed")
parser.add_argument("--num-experiments", type=int, default=25, help="Number of experiments for averaging")
parser.add_argument("--save-results", type=bool, default=False, help="Whether to save results")
parser.add_argument("--save-image", type=bool, default=True, help="Whether to save convergence image")
parser.add_argument("--plot-best-in-hindsight", type=bool, default=False,
                    help="Whether to plot best stock/portfolio in hindsight")
parser.add_argument("--image-filename", type=str, default='yield_wo_best', help="If save-image=True")
# parser.add_argument("--regret-image-filename", type=str, default='regret', help="If save-image=True")
parser.add_argument("--data-path", type=str, default='outputs/', help="Directory to save/load data from")
parser.add_argument("--run-experiments", type=bool, default=False,
                    help="whether to run experiments or use existing ones")

args = parser.parse_args()


def experiment(ratios, seed):
    np.random.seed(seed)    # set random seed to reproduce results
    T = ratios.shape[-1]  # number of rounds
    d = ratios.shape[0]   # dimensionality

    x1 = utl.sample_from_simplex(d)     # sample random init point
    # --- Online algorithms ---
    # OGD
    ogd_iterates = online_gradient_descent(ratios, T, x1,
                                           G=utl.compute_lipschitz_coeff(ratios, norm=2),
                                           D=np.sqrt(2))
    ogd_wealth = utl.get_portfolio_wealth_per_round(ogd_iterates, ratios)
    ogd_regret = utl.compute_regret(ogd_iterates, ratios)

    # OEG
    oeg_iterates = online_exponential_gradient(ratios, T, x1,
                                               G=utl.compute_lipschitz_coeff(ratios, norm=np.inf),
                                               D=np.sqrt(np.log(d)))
    oeg_wealth = utl.get_portfolio_wealth_per_round(oeg_iterates, ratios)
    oeg_regret = utl.compute_regret(oeg_iterates, ratios)

    # ONS
    ons_iterates = online_newton_step(ratios, T, x1,
                                      G=utl.compute_lipschitz_coeff(ratios, norm=2),
                                      D=np.sqrt(2))
    ons_wealth = utl.get_portfolio_wealth_per_round(ons_iterates, ratios)
    ons_regret = utl.compute_regret(ons_iterates, ratios)

    if args.save_results:
        # create directories if doesn't exists
        os.makedirs(os.path.join(args.data_path, 'ogd/'), exist_ok=True)
        os.makedirs(os.path.join(args.data_path, 'oeg/'), exist_ok=True)
        os.makedirs(os.path.join(args.data_path, 'ons/'), exist_ok=True)
        # os.makedirs(os.path.join(args.data_path, 'ogd_regret/'), exist_ok=True)
        # os.makedirs(os.path.join(args.data_path, 'oeg_regret/'), exist_ok=True)
        # os.makedirs(os.path.join(args.data_path, 'ons_regret/'), exist_ok=True)
        # save wealth arrays
        np.save(os.path.join(args.data_path, 'ogd/seed__{}'.format(seed)), ogd_wealth)
        np.save(os.path.join(args.data_path, 'oeg/seed__{}'.format(seed)), oeg_wealth)
        np.save(os.path.join(args.data_path, 'ons/seed__{}'.format(seed)), ons_wealth)
        # save regret arrays
        # np.save(os.path.join(args.data_path, 'ogd_regret/seed__{}'.format(seed)), ogd_regret)
        # np.save(os.path.join(args.data_path, 'oeg_regret/seed__{}'.format(seed)), oeg_regret)
        # np.save(os.path.join(args.data_path, 'ons_regret/seed__{}'.format(seed)), ons_regret)


if __name__ == '__main__':
    path = 'data/data_490_1000.mat'
    # load data
    assets = utl.load_data_mat_file_to_np_arr(path)
    # compute ratios (including short assets case)
    ratios = utl.get_all_revenues(assets)
    T = ratios.shape[1] + 1

    if args.run_experiments:
        for exp in range(args.num_experiments):
            experiment(ratios, seed=args.seed + exp)

    try:
        ogd_wealth = utl.load_experiments(os.path.join(args.data_path, 'ogd/'))
        oeg_wealth = utl.load_experiments(os.path.join(args.data_path, 'oeg/'))
        ons_wealth = utl.load_experiments(os.path.join(args.data_path, 'ons/'))
        # ogd_regret = utl.load_experiments(os.path.join(args.data_path, 'ogd_regret/'))
        # oeg_regret = utl.load_experiments(os.path.join(args.data_path, 'oeg_regret/'))
        # ons_regret = utl.load_experiments(os.path.join(args.data_path, 'ons_regret/'))
    except OSError:
        print('Data doesn\'t exists. Change save_results flag to True and try again (consider mini-batch size).')

    # --- Performance in hindsight ---
    if args.plot_best_in_hindsight:
        # Get performance of best portfolio in hindsight
        best_portfolio_in_hindsight = utl.find_best_portfolio_in_hindsight(ratios)
        best_portfolio_array = utl.get_portfolio_wealth_per_round([best_portfolio_in_hindsight for _ in range(T-1)],
                                                                  ratios)
        # Get performance of best asset in hindsight
        best_stock_array = utl.get_wealth_best_asset_per_round(ratios)

    # Plot wealth over time
    utl.plot_results(np.arange(T), ogd_wealth, label='OGD')
    utl.plot_results(np.arange(T), oeg_wealth, label='OEG')
    utl.plot_results(np.arange(T), ons_wealth, label='ONS')
    if args.plot_best_in_hindsight:
        utl.plot_results(np.arange(T), np.expand_dims(best_stock_array, 0), label='Best Asset (in HS)')
        utl.plot_results(np.arange(T), np.expand_dims(best_portfolio_array, 0), label='Best Portfolio (in HS)')
    plt.ylabel('Portfolio yield (%)')
    plt.xlabel('Trading rounds/days')
    plt.title('Portfolio yield vs. number of trading round')
    plt.legend()

    # Plot regret over time
    # utl.plot_results(np.arange(T-1), ogd_regret, label='OGD')
    # utl.plot_results(np.arange(T-1), oeg_regret, label='OEG')
    # utl.plot_results(np.arange(T-1), ons_regret, label='ONS')
    # # plt.plot(np.arange(T-1), np.arange(T-1), label='ONS')
    # plt.ylabel('Cumulative regret')
    # plt.xlabel('Trading rounds/days')
    # plt.title('Cumulative regret vs. number of trading round')
    # plt.legend()

    if args.save_image:
        plt.savefig(os.path.join(args.data_path, args.image_filename))
        # plt.savefig(os.path.join(args.data_path, args.regret_image_filename))
    plt.show()
