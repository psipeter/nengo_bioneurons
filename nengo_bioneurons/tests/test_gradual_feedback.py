import numpy as np
import nengo
from nengolib.signal import s
from nengolib.synapses import Lowpass  #, DoubleExp
from nengo_bioneurons import BahlNeuron, build_filter, evolve_h_d_out, plot_tuning_curves
import matplotlib.pyplot as plt
from seaborn import kdeplot
from nengo.utils.matplotlib import rasterplot

def test_fb_gradual(Simulator):
	pre_neurons = 100
	bio_neurons = 100
	dt = 0.001
	tau = 0.05
	radius = 1
	n_syn = 1
	t_train = 5.0  # 10
	t_test = 5.0  # 10
	dim = 1

	pass1_sig = 'cos'
	pass1_freq = 1
	pass1_seed = 1
	pass2_sig = 'cos'
	pass2_freq = 1
	pass2_seed = 2
	pass3_sig = 'cos'
	pass3_freq = 1
	pass3_seed = 3

	network_seed = 1
	sim_seed = 2
	ens_seed = 3
	conn_seed = 4
	sig_seed = 5
	evo_seed = 6

	t_evo = 10.0  # 10.0
	n_threads = 10
	evo_popsize = 10
	evo_gen = 5  # 4
	zeros_init = []
	poles_init = [-1e2, -1e2]
	zeros_delta = []
	poles_delta = [1e1, 1e1]

	T_inter_bios = [1.0, 0.5, 0.0]
	T_bio_bios = np.ones(len(T_inter_bios)) - T_inter_bios

	H_dir = '/home/pduggins/nengo_bioneurons/nengo_bioneurons/tests/filters/'
	H_inter = 'inter_bahl_2'  #  'inter_alif_3'  #
	H_bio = 'bio_bahl_14'  #  'bio_alif_7'  #  

	inter_type = BahlNeuron()  #  nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)  #  
	bio_type = BahlNeuron()  #  nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)  #  

	syn_locs = np.random.RandomState(seed=ens_seed).uniform(0, 1, size=(bio_neurons, bio_neurons, n_syn))

	def make_network(
		d_bio_out,
		d_inter_bio,
		d_bio_bio,
		tau_rise_bio_out,
		tau_fall_bio_out,
		tau_rise_inter_bio,
		tau_fall_inter_bio,
		tau_rise_bio_bio,
		tau_fall_bio_bio,
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
		syn_locs=None):

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
				gain=nengo.dists.Uniform(-1e2, 1e2),  # default
				bias=nengo.dists.Uniform(-1e1, 1e1),  # 3x default
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
				sec='tuft',
				n_syn=n_syn,
				syn_type='ExpSyn',
				tau_list=[tau],
				transform=tau,
				seed=conn_seed)
			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				transform=tau,
				seed=conn_seed)
			stim_pre2 = nengo.Connection(stim, pre2,
				synapse=1/s,  # compute integral
				seed=conn_seed)
			pre2_inter = nengo.Connection(pre2, inter,
				sec='tuft',
				n_syn=n_syn,
				syn_type='Exp2Syn',
				tau_list=[0.01, 0.01],  # todo: account for this filter in target
				seed=conn_seed)
			# stim_inter = nengo.Connection(stim, inter,
			# 	synapse=1/s,
			# 	seed=conn_seed)
			stim_target = nengo.Connection(stim, target,
				synapse=1/s)

			inter_bio = nengo.Connection(inter, bio,
				sec='tuft',
				n_syn=n_syn,
				syn_type='Exp2Syn',
				tau_list=[tau_rise_inter_bio, tau_fall_inter_bio],
				syn_locs=syn_locs,
				synapse=build_filter([], [-1.0/tau_rise_inter_bio, -1.0/tau_fall_inter_bio]),  # for alif testing
				transform=T_inter_bio,		
				solver=nengo.solvers.NoSolver(d_inter_bio),
				seed=conn_seed)
			bio_bio = nengo.Connection(bio, bio,
				sec='tuft',
				n_syn=n_syn,
				syn_type='Exp2Syn',
				tau_list=[tau_rise_bio_bio, tau_fall_bio_bio],
				syn_locs=syn_locs,
				synapse=build_filter([], [-1.0/tau_rise_bio_bio, -1.0/tau_fall_bio_bio]),  # for alif testing
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
			network.p_bio_act = nengo.Probe(bio.neurons, 'spikes',
				synapse=build_filter([], [-1.0/tau_rise_bio_out, -1.0/tau_fall_bio_out]))
			network.p_bio = nengo.Probe(bio,
				synapse=build_filter([], [-1.0/tau_rise_bio_out, -1.0/tau_fall_bio_out]),
				solver=nengo.solvers.NoSolver(d_bio_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=tau)
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_inter_act = nengo.Probe(inter.neurons, 'spikes',
				synapse=build_filter([], [-1.0/tau_rise_inter_bio, -1.0/tau_fall_inter_bio]))
			network.p_inter = nengo.Probe(inter,
				synapse=build_filter([], [-1.0/tau_rise_inter_bio, -1.0/tau_fall_inter_bio]),
				solver=nengo.solvers.NoSolver(d_inter_bio))
			network.p_target = nengo.Probe(target, synapse=tau)

			network.inter_bio = inter_bio
			network.bio_bio = bio_bio
			network.p_bio_spikes = nengo.Probe(bio.neurons, 'spikes', synapse=None)
			network.p_inter_spikes = nengo.Probe(inter.neurons, 'spikes', synapse=None)

		return network

	'''
	pass #1: compute d_inter_bio using activities and filtered target
	since inter will be adapting, we need to evolve readout filters and decoders,
	which will later be used on the synapses of the inter_bio connection
	'''
	try:
		zeros_inter_evo = np.load(H_dir+H_inter+'.npz')['zeros']
		poles_inter_evo = np.load(H_dir+H_inter+'.npz')['poles']
		d_inter_evo = np.load(H_dir+H_inter+'.npz')['decoders']
	except IOError:
		d_bio_out = np.zeros((bio_neurons, dim))
		d_inter_bio = np.zeros((bio_neurons, dim))
		d_bio_bio = np.zeros((bio_neurons, dim))
		tau_rise_bio_out = tau
		tau_fall_bio_out = tau
		tau_rise_inter_bio = tau
		tau_fall_inter_bio = tau
		tau_rise_bio_bio = tau
		tau_fall_bio_bio = tau
		T_inter_bio = 0.0
		T_bio_bio = 0.0

		network = make_network(
			d_bio_out,
			d_inter_bio,
			d_bio_bio,
			tau_rise_bio_out,
			tau_fall_bio_out,
			tau_rise_inter_bio,
			tau_fall_inter_bio,
			tau_rise_bio_bio,
			tau_fall_bio_bio,
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
			bio_type=nengo.LIF(),  # don't need to simulate bioneurons to decode inter
			syn_locs=syn_locs)
		zeros_inter_evo, poles_inter_evo, d_inter_evo = evolve_h_d_out(
			network,
			Simulator,
			sim_seed,
			t_evo,
			dt,
			tau,
			n_threads,
			evo_popsize,
			evo_gen,
			evo_seed,
			zeros_init,
			poles_init,
			zeros_delta,
			poles_delta,
			network.p_inter_act,
			network.p_target,
			H_dir,
			H_inter)

	# Test the accuracy of the inter decode
	d_bio_out = np.zeros((bio_neurons, dim))
	d_inter_bio = d_inter_evo
	d_bio_bio = np.zeros((bio_neurons, dim))
	tau_rise_bio_out = tau
	tau_fall_bio_out = tau
	tau_rise_inter_bio = -1.0 / poles_inter_evo[0]
	tau_fall_inter_bio = -1.0 / poles_inter_evo[1]
	tau_rise_bio_bio = tau
	tau_fall_bio_bio = tau
	T_inter_bio = 0.0
	T_bio_bio = 0.0

	network = make_network(
		d_bio_out,
		d_inter_bio,
		d_bio_bio,
		tau_rise_bio_out,
		tau_fall_bio_out,
		tau_rise_inter_bio,
		tau_fall_inter_bio,
		tau_rise_bio_bio,
		tau_fall_bio_bio,
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
		syn_locs=syn_locs)

	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_train)
	x_inter = sim.data[network.p_inter]
	x_target = sim.data[network.p_target]
	e_inter = nengo.utils.numpy.rmse(x_inter, x_target)

	encoders = None
	a_inter = sim.data[network.p_inter_act]
	x_pre2 = sim.data[network.p_pre2]
	for ens in network.ensembles:
		if ens.label == 'bio':
			encoders = sim.data[ens].encoders
			break
	plot_tuning_curves(encoders, x_pre2, a_inter, n_neurons=20)

	# fig, ax = plt.subplots(1, 1, figsize=(8,8))
	# rasterplot(sim.trange(), sim.data[network.p_inter_spikes], ax=ax)
	# # rasterplot(sim.trange(), sim.data[network.p_bio_spikes], ax=ax2)
	# ax.set(title='inter')
	# fig.savefig('plots/ff_inter_spikes')

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_inter, label='inter, e=%.5f' %e_inter)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/ff_inter_estimate')

	'''
	pass #2.n: compute d_bio_out/d_bio_bio using bio's activities and filtered target
	when some combination "ideal" spikes from inter and the recurrent spikes from bio
	are fed into bio.
	Assume that the readout filter parameters calculated above for inter can be used
	on the inter_bio, bio_bio, and bio_out connections
	'''
	try:
		times = np.load(H_dir+H_bio+'.npz')['times']
		x_target = np.load(H_dir+H_bio+'.npz')['x_target']
		x_bios = np.load(H_dir+H_bio+'.npz')['x_bios']
		e_bios = np.load(H_dir+H_bio+'.npz')['e_bios']
		s_bios = np.load(H_dir+H_bio+'.npz')['s_bios']
		a_bios = np.load(H_dir+H_bio+'.npz')['a_bios']
		s_inters = np.load(H_dir+H_bio+'.npz')['s_inters']
		a_inters = np.load(H_dir+H_bio+'.npz')['a_inters']
		d_bio_bios = np.load(H_dir+H_bio+'.npz')['d_bio_bios']
		d_bio_outs = np.load(H_dir+H_bio+'.npz')['d_bio_outs']
	except IOError:
		tau_rise_bio_out = -1.0 / poles_inter_evo[0]
		tau_fall_bio_out = -1.0 / poles_inter_evo[1]
		tau_rise_inter_bio = -1.0 / poles_inter_evo[0]
		tau_fall_inter_bio = -1.0 / poles_inter_evo[1]
		tau_rise_bio_bio = -1.0 / poles_inter_evo[0]
		tau_fall_bio_bio = -1.0 / poles_inter_evo[1]

		x_bios = []
		e_bios = []
		s_bios = []
		a_bios = []
		s_inters = []
		a_inters = []
		d_bio_outs = [d_inter_evo]
		d_bio_bios = [np.zeros((bio_neurons, dim))]
		for n in range(len(T_inter_bios)):
			d_inter_bio = d_inter_evo
			d_bio_out = d_bio_outs[-1]
			d_bio_bio = d_bio_bios[-1]
			T_inter_bio = T_inter_bios[n]
			T_bio_bio = T_bio_bios[n]

			network = make_network(
				d_bio_out,
				d_inter_bio,
				d_bio_bio,
				tau_rise_bio_out,
				tau_fall_bio_out,
				tau_rise_inter_bio,
				tau_fall_inter_bio,
				tau_rise_bio_bio,
				tau_fall_bio_bio,
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
				syn_locs)

			with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
				sim.run(t_train)
			s_bio = sim.data[network.p_bio_spikes]
			a_bio = sim.data[network.p_bio_act]
			s_inter = sim.data[network.p_inter_spikes]
			a_inter = sim.data[network.p_inter_act]
			x_target = sim.data[network.p_target]
			d_bio_outs.append(nengo.solvers.LstsqL2()(a_bio, x_target)[0])
			d_bio_bios.append(d_bio_outs[-1])
			x_bios.append(np.dot(a_bio, d_bio_outs[-1]))
			e_bios.append(nengo.utils.numpy.rmse(x_target, x_bios[-1]))
			s_bios.append(s_bio)
			a_bios.append(a_bio)
			a_inters.append(a_inter)
			s_inters.append(s_inter)

		# Save the trained decoders
		times = sim.trange()
		np.savez(H_dir+H_bio,
			times=times,
			x_bios=x_bios,
			e_bios=e_bios,
			s_bios=s_bios,
			a_bios=a_bios,
			a_inters=a_inters,
			s_inters=s_inters,
			x_target=x_target,
			d_bio_bios=d_bio_bios,
			d_bio_outs=d_bio_outs)

	# Plot the estimates over the course of the transition
	fig, ax = plt.subplots(1, 1)
	for n in range(len(x_bios)):
		ax.plot(times, x_bios[n], label='bio T_bio_bio=%.3f, e=%.3f'
			%(T_bio_bios[n], e_bios[n]))
	ax.plot(times, x_target, label='target')
	ax.legend()
	fig.savefig('plots/fb_gradual_transition_estimates')

	# Plot the spikes the course of the transition
	fig, axs = plt.subplots(1, len(x_bios), figsize=(3*len(x_bios),8), sharex=True, sharey=True)
	for n in range(len(x_bios)):
		# rasterplot(times, s_inters[n], ax=ax)
		rasterplot(times, s_bios[n], ax=axs[n])
		axs[n].set(title='T_bio_bio=%0.3f' %T_bio_bios[n])
	fig.savefig('plots/fb_gradual_transition_spikes')

	# Plot the total activity (inter + bio) going into bio as a function of time
	fig, axs = plt.subplots(1, len(x_bios), figsize=(3*len(x_bios),8), sharex=True, sharey=True)
	for n in range(len(x_bios)):
		axs[n].plot(times, np.sum(a_bios[n], axis=1) + np.sum(a_inters[n], axis=1))
		axs[n].set(title='T_bio_bio=%0.3f' %T_bio_bios[n])
	axs[0].set(ylabel='summed input activities (unweighted)')
	fig.savefig('plots/fb_gradual_transition_input_currents_unweighted')


	# Plot the total activity (inter + bio) going into bio as a function of time,
	# weighted by decoders

	'''
	pass #3: test the accuracy of bio when these decoders/filters
	are used for the recurrent connection
	'''
	# d_bio_out = d_bio_outs[1]
	# d_inter_bio = d_inter_evo
	# d_bio_bio = d_bio_bios[0]
	# tau_rise_bio_out = -1.0 / poles_inter_evo[0]
	# tau_fall_bio_out = -1.0 / poles_inter_evo[1]
	# tau_rise_inter_bio = -1.0 / poles_inter_evo[0]
	# tau_fall_inter_bio = -1.0 / poles_inter_evo[1]
	# tau_rise_bio_bio = -1.0 / poles_inter_evo[0]
	# tau_fall_bio_bio = -1.0 / poles_inter_evo[1]
	# T_inter_bio = 0.0
	# T_bio_bio = 1.0

	# network = make_network(
	# 	d_bio_out,
	# 	d_inter_bio,
	# 	d_bio_bio,
	# 	tau_rise_bio_out,
	# 	tau_fall_bio_out,
	# 	tau_rise_inter_bio,
	# 	tau_fall_inter_bio,
	# 	tau_rise_bio_bio,
	# 	tau_fall_bio_bio,
	# 	T_inter_bio,
	# 	T_bio_bio,
	# 	network_seed,
	# 	sim_seed,
	# 	ens_seed,
	# 	conn_seed,
	# 	pass3_sig,
	# 	pass3_freq,
	# 	pass3_seed,
	# 	inter_type,  # not used during testing
	# 	bio_type,
	# 	syn_locs)

	# with Simulator(network, seed=sim_seed, dt=dt) as sim:
	# 	sim.run(t_test)
	# a_bio = sim.data[network.p_bio_act]
	# x_inter = sim.data[network.p_inter]
	# x_target = sim.data[network.p_target]
	# x_bio = sim.data[network.p_bio]
	# x_lif = sim.data[network.p_lif]
	# e_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	# e_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	# encoders = None
	# for ens in network.ensembles:
	# 	if ens.label == 'bio':
	# 		encoders = sim.data[ens].encoders
	# 		break
	# plot_tuning_curves(encoders, x_inter, a_bio)

	# fig, ax = plt.subplots(1, 1)
	# ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
	# ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %e_lif)
	# ax.plot(sim.trange(), x_target, label='target')
	# ax.legend()
	# fig.savefig('plots/fb_gradual_bio_bio_estimates')

	# # fig, ax = plt.subplots(1, 1)
	# # kdeplot(d_inter_bio.squeeze(), label='inter_bio')
	# # kdeplot(d_bio_bio.squeeze(), label='bio_bio')
	# # ax.legend()
	# # fig.savefig('plots/fb_gradual_bio_bio_decoders')

	# # note: full weight matrices for each of n_syn synapses, just look at 0th syn
	# # fig, ax = plt.subplots(1, 1)
	# # for syn in range(sim.data[network.inter_bio].weights.T.shape[0]):
	# # 	w_inter_bio = sim.data[network.inter_bio].weights.T[syn]
	# # 	w_bio_bio = sim.data[network.bio_bio].weights.T[syn]
	# # 	if np.sum(w_inter_bio) != 0.0: kdeplot(w_inter_bio.ravel(), label='inter_bio %s' %syn)
	# # 	if np.sum(w_bio_bio) != 0.0: kdeplot(w_bio_bio.ravel(), label='bio_bio %s' %syn)
	# # ax.legend()
	# # fig.savefig('plots/fb_gradual_bio_bio_weights')

	# fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8,16))
	# rasterplot(sim.trange(), sim.data[network.p_inter_spikes], ax=ax)
	# rasterplot(sim.trange(), sim.data[network.p_bio_spikes], ax=ax2)
	# ax.set(title='inter')
	# ax2.set(title='bio')
	# fig.savefig('plots/fb_gradual_bio_bio_spikes')