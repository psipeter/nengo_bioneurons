import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ['plot_tuning_curves']

def plot_tuning_curves(
    encoders,
    xhat_pre,
    act_bio,
    n_neurons=10,
    figname='plots/tuning_curves.png',
    n_eval_points=20):

    fig, ax = plt.subplots(1, 1)

    for i in range(n_neurons):  # act_bio.shape[1]
        x_dot_e = np.dot(
            xhat_pre,
            np.sign(encoders[i]))
        x_dot_e_vals = np.linspace(
            np.min(x_dot_e),
            np.max(x_dot_e), 
            num=n_eval_points)
        Hz_mean = np.zeros((x_dot_e_vals.shape[0]))
        Hz_stddev = np.zeros_like(Hz_mean)

        for xi in range(x_dot_e_vals.shape[0] - 1):
            ts_greater = np.where(x_dot_e_vals[xi] < xhat_pre)[0]
            ts_smaller = np.where(xhat_pre < x_dot_e_vals[xi + 1])[0]
            ts = np.intersect1d(ts_greater, ts_smaller)
            if ts.shape[0] > 0: Hz_mean[xi] = np.average(act_bio[ts, i])
            if ts.shape[0] > 1: Hz_stddev[xi] = np.std(act_bio[ts, i])

        bioplot = ax.plot(x_dot_e_vals[:-2], Hz_mean[:-2])  # , label='%s' %i
        ax.fill_between(x_dot_e_vals[:-2],
            Hz_mean[:-2]+Hz_stddev[:-2],
            Hz_mean[:-2]-Hz_stddev[:-2],
            alpha=0.5)

        # if np.any(Hz_mean[:-2]+Hz_stddev[:-2] > max_rate):
        #     warnings.warn('warning: neuron %s over max_rate' %i)
        # if np.all(Hz_mean[:-2]+Hz_stddev[:-2] < min_rate):
        #     warnings.warn('warning: neuron %s under min_rate' %i)

    ax.legend()
    fig.savefig(figname)