import numpy as np
import nengo
from nengolib.signal import s, nrmse
from nengolib.synapses import Lowpass  #, DoubleExp
from nengo_bioneurons import BahlNeuron, build_filter, evolve_h_d_out, plot_tuning_curves
import matplotlib.pyplot as plt
from seaborn import kdeplot
from nengo.utils.matplotlib import rasterplot

def test_fb_gradual(Simulator):
	pre_neurons = 50
	bio_neurons = 50
	dt = 0.001
	tau = 0.1
	radius = 1
	n_syn = 1
	t_train = 1.0  # 10
	t_test = 1.0  # 10
	dim = 1
	reg = 0.5

	pass1_sig = 'cos'
	pass1_freq = 1*2*np.pi
	pass1_seed = 1
	pass2_sig = 'cos'
	pass2_freq = 1*2*np.pi
	pass2_seed = 2
	pass3_sig = 'cos'
	pass3_freq = 1*2*np.pi
	pass3_seed = 5

	network_seed = 1
	sim_seed = 2
	ens_seed = 3
	conn_seed = 4
	sig_seed = 5

	T_bio_bios = [0.0, 1.0]  # np.linspace(0.0, 1.0, 3)
	T_inter_bios = np.ones(len(T_bio_bios)) - T_bio_bios

	inter_type = nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)  #  BahlNeuron()  #  
	bio_type = nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)  #  BahlNeuron()  #  

	syn_locs = np.random.RandomState(seed=ens_seed).uniform(0, 1, size=(bio_neurons, bio_neurons, n_syn))

	def make_network(
		d_bio_out,
		d_inter_bio,
		d_bio_bio,
		tau,
		T_inter_bio,
		T_bio_bio,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		sig_type,
		sig_freq,
		sig_seed,
		inter_type,
		bio_type,
		syn_locs=None,
		sig_norm=1.0):

		with nengo.Network(seed=network_seed) as network:

			if sig_type == 'sin':
				stim = nengo.Node(lambda t: np.sin(sig_freq * t))
			if sig_type == 'cos':
				stim = nengo.Node(lambda t: np.cos(sig_freq * t))
			elif sig_type == 'white_noise':
				stim = nengo.Node(nengo.processes.WhiteSignal(
					period=100,
					high=sig_freq,
					rms=0.5,
					seed=sig_seed))
			pre = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				neuron_type=nengo.LIF(),
				seed=ens_seed,
				label='pre')
			bio = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				radius=radius,
				neuron_type=bio_type,
				seed=ens_seed,
				label='bio')
			lif = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				radius=radius,
				neuron_type=nengo.LIF(),
				seed=ens_seed)
			pre2 = nengo.Ensemble(
				n_neurons=pre_neurons,
				dimensions=dim,
				neuron_type=nengo.LIF(),
				seed=ens_seed)
			inter = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				radius=radius,
				neuron_type=inter_type,
				seed=ens_seed,
				label='inter')
			target = nengo.Node(
				size_in=dim,
				label='target')

			stim_pre = nengo.Connection(stim, pre,
				synapse=None,
				seed=conn_seed)
			pre_bio = nengo.Connection(pre, bio,
				sec='apical',
				n_syn=n_syn,
				syn_type='ExpSyn',
				tau_list=[tau],
				# transform=tau,
				transform=tau*sig_norm,
				seed=conn_seed)
			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				# transform=tau,
				transform=tau*sig_norm,
				seed=conn_seed)
			# stim_pre2 = nengo.Connection(stim, pre2,
			# 	synapse=1/s,  # compute integral
			# 	seed=conn_seed)
			# pre2_inter = nengo.Connection(pre2, inter,
			# 	sec='apical',
			# 	n_syn=n_syn,
			# 	syn_type='ExpSyn',
			# 	tau_list=[0.01],  # todo: account for this filter in target
			# 	transform=sig_norm,
			# 	seed=conn_seed)
			stim_inter = nengo.Connection(stim, inter,
				synapse=1/s,
				seed=conn_seed,
				transform=sig_norm)
			stim_target = nengo.Connection(stim, target,
				synapse=1/s,
				transform=sig_norm)

			inter_bio = nengo.Connection(inter, bio,
				sec='apical',
				n_syn=n_syn,
				syn_type='ExpSyn',
				tau_list=[tau],
				syn_locs=syn_locs,
				synapse=tau,  # for alif testing
				transform=T_inter_bio,		
				solver=nengo.solvers.NoSolver(d_inter_bio),
				seed=conn_seed)
			bio_bio = nengo.Connection(bio, bio,
				sec='apical',
				n_syn=n_syn,
				syn_type='ExpSyn',
				tau_list=[tau],
				syn_locs=syn_locs,
				synapse=tau,  # for alif testing
				transform=T_bio_bio,
				solver=nengo.solvers.NoSolver(d_bio_bio),
				seed=conn_seed)
			lif_lif = nengo.Connection(lif, lif,
				synapse=tau,
				seed=conn_seed)

			network.p_stim = nengo.Probe(stim)
			network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=tau)
			network.p_pre = nengo.Probe(pre, synapse=tau)
			network.p_pre2 = nengo.Probe(pre2, synapse=tau)
			network.p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=tau)
			network.p_bio = nengo.Probe(bio, synapse=tau,
				solver=nengo.solvers.NoSolver(d_bio_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=tau)
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_inter_act = nengo.Probe(inter.neurons, 'spikes', synapse=tau)
			network.p_inter = nengo.Probe(inter, synapse=tau,
				solver=nengo.solvers.NoSolver(d_inter_bio))
			network.p_target = nengo.Probe(target, synapse=tau)

			network.inter_bio = inter_bio
			network.bio_bio = bio_bio
			network.p_bio_spikes = nengo.Probe(bio.neurons, 'spikes', synapse=None)
			network.p_inter_spikes = nengo.Probe(inter.neurons, 'spikes', synapse=None)

		return network

	'''
	pass #1: compute d_inter_bio using activities and filtered target
	'''
	T_inter_bio = 0.0
	T_bio_bio = 0.0

	network = make_network(
		np.zeros((bio_neurons, dim)),
		np.zeros((bio_neurons, dim)),
		np.zeros((bio_neurons, dim)),
		tau,
		T_inter_bio,
		T_bio_bio,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		pass1_sig,
		pass1_freq,
		pass1_seed,
		inter_type,
		bio_type,
		syn_locs,
		sig_norm=pass1_freq)

	with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
		sim.run(t_train)
	a_inter = sim.data[network.p_inter_act]
	x_target = sim.data[network.p_target]
	d_inter_bio, _ = nengo.solvers.LstsqL2(reg=reg)(a_inter, x_target)
	x_inter = np.dot(a_inter, d_inter_bio)
	e_inter = nrmse(x_inter, target=x_target)

	# encoders = None
	# a_inter = sim.data[network.p_inter_act]
	# x_pre2 = sim.data[network.p_pre2]
	# for ens in network.ensembles:
	# 	if ens.label == 'bio':
	# 		encoders = sim.data[ens].encoders
	# 		break
	# plot_tuning_curves(encoders, x_pre2, a_inter, n_neurons=20)

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_inter, label='inter, nrmse=%.5f' %e_inter)
	ax.plot(sim.trange(), x_target, label='target', ls='--')
	ax.legend()
	fig.savefig('plots/ff_inter_estimate_oracle')

	'''
	pass #2.n: use the oracle method to calculate d_bio_out/d_bio_bio
	using bio's activities and filtered target
	when some combination "ideal" spikes from inter and the recurrent spikes from bio
	are fed into bio.
	Note: regularization values are important
	'''
	x_bios = []
	e_bios = []
	s_bios = []
	a_bios = []
	s_inters = []
	a_inters = []
	d_bio_outs = [np.zeros((bio_neurons, dim))]
	d_bio_bios = [np.zeros((bio_neurons, dim))]

	for n in range(len(T_inter_bios)):
		d_bio_out = d_bio_outs[-1]
		d_bio_bios.append(d_bio_out)
		d_bio_bio = d_bio_bios[-1]
		T_bio_bio = T_bio_bios[n]
		T_inter_bio = T_inter_bios[n]

		network = make_network(
			d_bio_out,
			d_inter_bio,
			d_bio_bio,
			tau,
			T_inter_bio,
			T_bio_bio,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			pass2_sig,
			pass2_freq,
			pass2_seed,
			inter_type,
			bio_type,
			syn_locs,
			sig_norm=pass2_freq)

		with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
			sim.run(t_train)

		times = sim.trange()
		s_bio = sim.data[network.p_bio_spikes]
		a_bio = sim.data[network.p_bio_act]
		s_inter = sim.data[network.p_inter_spikes]
		a_inter = sim.data[network.p_inter_act]
		x_target = sim.data[network.p_target]
		x_inter = sim.data[network.p_inter]
		d_bio_out, _ = nengo.solvers.LstsqL2(reg=reg)(a_bio, x_target)
		x_bio = np.dot(a_bio, d_bio_out)

		d_bio_outs.append(d_bio_out)
		x_bios.append(x_bio)
		e_bios.append(nrmse(x_bio, target=x_target))
		s_bios.append(s_bio)
		a_bios.append(a_bio)
		a_inters.append(a_inter)
		s_inters.append(s_inter)

	# Plot the estimates over the course of the transition
	fig, ax = plt.subplots(1, 1)
	for n in range(len(x_bios)):
		ax.plot(times, x_bios[n], label='bio T_bio_bio=%.3f, nrmse=%.3f'
			%(T_bio_bios[n], e_bios[n]))
	ax.plot(times, x_target, label='target')
	# ax.plot(times, x_inter, label='inter')
	ax.legend()
	fig.savefig('plots/fb_gradual_transition_oracle_estimates')

	# Plot the spikes the course of the transition
	fig, axs = plt.subplots(1, len(x_bios)+1, figsize=(3*(len(x_bios)+1),8), sharex=True, sharey=True)
	for n in range(len(x_bios)):
		rasterplot(times, s_inters[n], ax=axs[0])
		rasterplot(times, s_bios[n], ax=axs[n+1])
		axs[0].set(title='inter')
		axs[n+1].set(title='T_bio_bio=%0.3f' %T_bio_bios[n])
	fig.savefig('plots/fb_gradual_transition_oracle_spikes')

	'''
	pass #3: test the accuracy of bio when these decoders/filters
	are used for the recurrent connection
	'''
	d_bio_out = d_bio_outs[-1]
	d_bio_bio = d_bio_bios[-1]
	T_bio_bio = 1.0
	T_inter_bio = 0.0

	network = make_network(
		d_bio_out,
		d_inter_bio,
		d_bio_bio,
		tau,
		T_inter_bio,
		T_bio_bio,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		pass3_sig,
		pass3_freq,
		pass3_seed,
		inter_type,  # not used during testing
		bio_type,
		syn_locs,
		sig_norm=pass3_freq)  # todo: normalize white_noise

	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_test)
	a_bio = sim.data[network.p_bio_act]
	x_inter = sim.data[network.p_inter]
	x_target = sim.data[network.p_target]
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	e_bio = nrmse(x_bio, target=x_target)
	e_lif = nrmse(x_lif, target=x_target)

	# encoders = None
	# for ens in network.ensembles:
	# 	if ens.label == 'bio':
	# 		encoders = sim.data[ens].encoders
	# 		break
	# plot_tuning_curves(encoders, x_inter, a_bio)

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, nrmse=%.5f' %e_bio)
	ax.plot(sim.trange(), x_lif, label='lif, nrmse=%.5f' %e_lif)
	ax.plot(sim.trange(), x_target, label='target', ls='--')
	ax.legend()
	fig.savefig('plots/fb_gradual_oracle_bio_bio_estimates')

	fig, ax = plt.subplots(1, 1)
	if np.sum(d_inter_bio) > 0: kdeplot(d_inter_bio.squeeze(), label='inter_bio')
	if np.sum(d_bio_bio) > 0: kdeplot(d_bio_bio.squeeze(), label='bio_bio')
	if np.sum(d_bio_out) > 0: kdeplot(d_bio_out.squeeze(), label='bio_out')
	ax.legend()
	fig.savefig('plots/fb_gradual_oracle_bio_bio_decoders')