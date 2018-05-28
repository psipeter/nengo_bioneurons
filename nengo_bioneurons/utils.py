import nengo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nengolib.signal import s

__all__ = ['plot_tuning_curves', 'make_stimulus', 'norms']

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

def make_stimulus(signal, freq, amp, seed):       
    if signal == 'cos':
        return nengo.Node(output=lambda t: np.cos(freq*t))
    elif signal == 'sin':
        return nengo.Node(output=lambda t: np.sin(freq*t))
    elif signal == 'ramp':
        return nengo.Node(output=lambda t: 1.0/freq*(t<freq))
    elif signal == 'white_noise':
        return nengo.Node(nengo.processes.WhiteSignal(
            period=100,
            high=freq,
            rms=amp,
            seed=seed))


def norms(signal, freq, amp, ss, tau, t, dt=0.001, plot=False):
    lpf=nengo.Lowpass(tau)
    with nengo.Network() as model:
        stim = make_stimulus(signal, freq, amp, ss)
        p_stim = nengo.Probe(stim, synapse=None)
        p_integral = nengo.Probe(stim, synapse=1/s)
    with nengo.Simulator(model, progress_bar=False, dt=dt) as sim:
        sim.run(t, progress_bar=False)
    stimulus = sim.data[p_stim]
    target = sim.data[p_integral]
    target_f = lpf.filt(sim.data[p_integral], dt=dt)
    norm_s = np.max(np.abs(stimulus))
    norm = np.max(np.abs(target))
    norm_f = np.max(np.abs(target_f))
    if plot:
        plt.plot(sim.trange(), stimulus, label='stim', alpha=0.5)
        plt.plot(sim.trange(), target, label='integral', alpha=0.5)
        plt.plot(sim.trange(), lpf.filt(sim.data[p_integral]/norm_f, dt=dt), label='target', alpha=0.5)
        plt.legend()
        plt.show()
    return norm, norm_s, norm_f