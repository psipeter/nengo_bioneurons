import numpy as np
import nengo
from nengo_bioneurons import BahlNeuron

def test_connection_multi_synapse_section(Simulator, plt):
	pre_neurons = 100
	bio_neurons = 30
	tau = 0.01
	tau2 = 0.02
	radius = 1
	n_syn = 5
	n_syn2 = 3
	n_syn3 = 7
	t_test = 10.0
	dim = 1
	freq = 0.5 * np.pi
	d_out = np.zeros((bio_neurons, dim))

	network_seed = 1
	sim_seed = 2
	bio_seed = 3

	with nengo.Network(seed=network_seed) as network:
		stim = nengo.Node(lambda t: np.cos(freq*t))
		pre = nengo.Ensemble(
			n_neurons=pre_neurons,
			dimensions=dim,
			neuron_type=nengo.LIF())
		bio = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim,
			radius=radius,
			# encoders=nengo.dists.Uniform(-1e0,1e0),
			# gain=nengo.dists.Uniform(-1e2,1e2),
			# bias=nengo.dists.Uniform(-3e0,3e0),
			neuron_type=BahlNeuron(),
			seed=bio_seed)
		lif = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim,
			radius=radius,
			neuron_type=nengo.LIF(),
			seed=bio_seed)

		stim_pre = nengo.Connection(stim, pre,
			synapse=tau)
		pre_bio = nengo.Connection(pre, bio,
			syn_sec={
				'apical': {'n_syn': n_syn, 'syn_type': 'ExpSyn', 'tau': [tau]},
				'tuft': {'n_syn': n_syn2, 'syn_type': 'Exp2Syn', 'tau': [tau, tau2]},
				'basal': {'n_syn': n_syn3, 'syn_type': 'ExpSyn', 'tau': [tau]},
			})
		pre_lif = nengo.Connection(pre, lif,
			synapse=tau)

		p_stim = nengo.Probe(stim)
		p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		p_pre = nengo.Probe(pre, synapse=tau)
		p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		p_bio = nengo.Probe(bio, synapse=tau, solver=nengo.solvers.NoSolver(d_out))
		p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		p_lif = nengo.Probe(lif, synapse=tau)

	with Simulator(network, seed=sim_seed) as sim:
		sim.run(t_test)

	# print np.sum(sim.data[p_bio_act])
	# assert False

	# plt.plot(sim.trange(), sim.data[p_pre_act], label='pre')
	# plt.plot(sim.trange(), sim.data[p_bio_act], label='bio')
	# plt.xlabel('time (s)')
	# plt.ylabel('a(t)')
	# plt.legend()

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1)

	n_eval_points = 20
	xhat_pre = sim.data[p_pre]
	act_bio = sim.data[p_bio_act]
	act_lif = sim.data[p_lif_act]
	for i in range(bio_neurons):
		x_dot_e_bio = np.dot(
			xhat_pre,
			np.sign(bio.encoders[i]))
		x_dot_e_lif = np.dot(
			xhat_pre,
			np.sign(sim.data[lif].encoders)[i])
		x_dot_e_vals_bio = np.linspace(
			np.min(x_dot_e_bio),
			np.max(x_dot_e_bio), 
			num=n_eval_points)
		x_dot_e_vals_lif = np.linspace(
			np.min(x_dot_e_lif),
			np.max(x_dot_e_lif), 
			num=n_eval_points)
		# print x_dot_e_vals_lif, x_dot_e_vals_bio, np.min(xhat_pre), np.min(sim.data[p_stim])
		# assert False
		Hz_mean_bio = np.zeros((x_dot_e_vals_bio.shape[0]))
		Hz_stddev_bio = np.zeros_like(Hz_mean_bio)
		Hz_mean_lif = np.zeros((x_dot_e_vals_lif.shape[0]))
		Hz_stddev_lif = np.zeros_like(Hz_mean_lif)

		for xi in range(x_dot_e_vals_bio.shape[0] - 1):
			ts_greater = np.where(x_dot_e_vals_bio[xi] < xhat_pre)[0]
			ts_smaller = np.where(xhat_pre < x_dot_e_vals_bio[xi + 1])[0]
			ts = np.intersect1d(ts_greater, ts_smaller)
			if ts.shape[0] > 0: Hz_mean_bio[xi] = np.average(act_bio[ts, i])
			if ts.shape[0] > 1: Hz_stddev_bio[xi] = np.std(act_bio[ts, i])
		for xi in range(x_dot_e_vals_lif.shape[0] - 1):
			ts_greater = np.where(x_dot_e_vals_lif[xi] < xhat_pre)[0]
			ts_smaller = np.where(xhat_pre < x_dot_e_vals_lif[xi + 1])[0]
			ts = np.intersect1d(ts_greater, ts_smaller)
			if ts.shape[0] > 0: Hz_mean_lif[xi] = np.average(act_lif[ts, i])
			if ts.shape[0] > 1: Hz_stddev_lif[xi] = np.std(act_lif[ts, i])

		# lifplot = ax.plot(x_dot_e_vals_lif[:-2], Hz_mean_lif[:-2])
		# ax.fill_between(x_dot_e_vals_lif[:-2],
		# 	Hz_mean_lif[:-2]+Hz_stddev_lif[:-2],
		# 	Hz_mean_lif[:-2]-Hz_stddev_lif[:-2],
		# 	alpha=0.5)
		bioplot = ax.plot(x_dot_e_vals_bio[:-2], Hz_mean_bio[:-2])
		ax.fill_between(x_dot_e_vals_bio[:-2],
			Hz_mean_bio[:-2]+Hz_stddev_bio[:-2],
			Hz_mean_bio[:-2]-Hz_stddev_bio[:-2],
			alpha=0.5)

	# plt.ylim(ymin=0)
	# plt.xlabel('$x \cdot e$')
	# plt.ylabel('firing rate')
	# plt.title('Tuning Curves')
	# plt.legend()
	fig.savefig('plots/multi_synapse_tuning_curves.png')