import numpy as np
import nengo
from nengolib.signal import s
from nengolib.synapses import Lowpass  #, DoubleExp
from nengo_bioneurons import BahlNeuron, build_filter, evolve_h_d_out, make_tuning_curves

def test_fb_doubleexp(Simulator, plt):
	pre_neurons = 100
	bio_neurons = 100
	dt = 0.001
	tau = 0.05
	radius = 1
	n_syn = 1
	t_train = 10.0  # 10
	t_test = 10.0  # 10
	dim = 1

	pass1_sig = 'cos'
	pass1_freq = 1
	pass1_seed = 1
	pass2_sig = 'cos'
	pass2_freq = 1
	pass2_seed = 2
	pass3_sig = 'white_noise'
	pass3_freq = 2
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
	evo_gen = 7  # 4
	zeros_init = []
	poles_init = [-1e2, -1e2]
	zeros_delta = []
	poles_delta = [1e1, 1e1]
	training_dir = '/home/pduggins/nengo_bioneurons/nengo_bioneurons/tests/filters/'
	training_file_inter = 'test_fb_inter' + \
		'_%s_bioneurons_%s_t_evo_%s_evo_popsize_%s_evo_gen'\
		%(bio_neurons, t_evo, evo_popsize, evo_gen)
	training_file_bio = 'test_fb_bio' + \
		'_%s_bioneurons_%s_t_evo_%s_evo_popsize_%s_evo_gen'\
		%(bio_neurons, t_evo, evo_popsize, evo_gen)

	inter_type = nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)  #  BahlNeuron()  # 
	bio_type = nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)  #  BahlNeuron()  # 

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
		bio_type):

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
				seed=ens_seed)
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
			target = nengo.Node(size_in=dim)

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
				# syn_locs=syn_locs,
				# synapse=DoubleExp(tau_rise_inter_bio, tau_fall_inter_bio),  # for alif testing
				synapse=build_filter([], [-1.0/tau_rise_inter_bio, -1.0/tau_fall_inter_bio]),  # for alif testing
				transform=T_inter_bio,		
				solver=nengo.solvers.NoSolver(d_inter_bio),
				seed=conn_seed)
			bio_bio = nengo.Connection(bio, bio,
				sec='tuft',
				n_syn=n_syn,
				syn_type='Exp2Syn',
				tau_list=[tau_rise_bio_bio, tau_fall_bio_bio],
				# syn_locs=syn_locs,
				# synapse=DoubleExp(tau_rise_bio_bio, tau_fall_bio_bio),  # for alif testing
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
				# synapse=DoubleExp(tau_rise_bio_out, tau_fall_bio_out))
				synapse=build_filter([], [-1.0/tau_rise_bio_out, -1.0/tau_fall_bio_out]))
			network.p_bio = nengo.Probe(bio,
				# synapse=DoubleExp(tau_rise_bio_out, tau_fall_bio_out),
				synapse=build_filter([], [-1.0/tau_rise_bio_bio, -1.0/tau_fall_bio_bio]),
				solver=nengo.solvers.NoSolver(d_bio_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=tau)
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_inter_act = nengo.Probe(inter.neurons, 'spikes',
				# synapse=DoubleExp(tau_rise_inter_bio, tau_fall_inter_bio))
				synapse=build_filter([], [-1.0/tau_rise_inter_bio, -1.0/tau_fall_inter_bio]))
			network.p_inter = nengo.Probe(inter,
				# synapse=DoubleExp(tau_rise_inter_bio, tau_fall_inter_bio),
				synapse=build_filter([], [-1.0/tau_rise_inter_bio, -1.0/tau_fall_inter_bio]),
				solver=nengo.solvers.NoSolver(d_inter_bio))
			network.p_target = nengo.Probe(target, synapse=tau)

			network.inter_bio = inter_bio
			network.p_bio_spikes = nengo.Probe(bio.neurons, 'spikes', synapse=None)
			network.p_inter_spikes = nengo.Probe(inter.neurons, 'spikes', synapse=None)

		return network

	'''
	pass #1: compute d_inter_bio using activities and filtered target
	since inter will be adapting, we need to evolve readout filters and decoders,
	which will later be used on the synapses of the inter_bio connection
	'''
	try:
		zeros_inter_evo = np.load(training_dir+training_file_inter+'.npz')['zeros']
		poles_inter_evo = np.load(training_dir+training_file_inter+'.npz')['poles']
		d_inter_evo = np.load(training_dir+training_file_inter+'.npz')['decoders']
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
			bio_type=nengo.LIF())  # don't need to simulate bioneurons to decode inter

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
			training_dir,
			training_file_inter)

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
		bio_type=nengo.LIF())

	# test_rates(network,
	# 	Simulator,
	# 	sim_seed,
	# 	'inter',
	# 	network.p_pre2,
	# 	network.p_inter_act,
	# 	t_test=10.0)

	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_train)
	x_inter = sim.data[network.p_inter]
	x_target = sim.data[network.p_target]
	e_inter = nengo.utils.numpy.rmse(x_inter, x_target)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_inter, label='inter, e=%.5f' %e_inter)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/ff_inter')

	'''
	pass #2: compute d_bio_out/d_bio_bio using bio's activities and filtered target
	when the "ideal" spikes from inter are fed to bio instead of bio's recurrent spikes.
	Since bio will be adapting, we need to evolve readout filters and decoders,
	which will later be used on the synapses of the bio_bio connection and for bio_out
	'''
	try:
		zeros_bio_evo = np.load(training_dir+training_file_bio+'.npz')['zeros']
		poles_bio_evo = np.load(training_dir+training_file_bio+'.npz')['poles']
		d_bio_evo = np.load(training_dir+training_file_bio+'.npz')['decoders']
	except IOError:
		d_bio_out = np.zeros((bio_neurons, dim))
		d_inter_bio = d_inter_evo
		d_bio_bio = np.zeros((bio_neurons, dim))
		tau_rise_bio_out = tau
		tau_fall_bio_out = tau
		tau_rise_inter_bio = -1.0 / poles_inter_evo[0]
		tau_fall_inter_bio = -1.0 / poles_inter_evo[1]
		tau_rise_bio_bio = tau
		tau_fall_bio_bio = tau
		T_inter_bio = 1.0
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
			pass2_sig,
			pass2_freq,
			pass2_seed,
			inter_type,
			bio_type)

		zeros_bio_evo, poles_bio_evo, d_bio_evo = evolve_h_d_out(
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
			network.p_bio_act,
			network.p_target,
			training_dir,
			training_file_bio)

	# Test the accuracy of the ideally-stimulated bioneurons
	d_bio_out = d_bio_evo
	d_inter_bio = d_inter_evo
	d_bio_bio = d_bio_evo
	tau_rise_bio_out = -1.0 / poles_bio_evo[0]
	tau_fall_bio_out = -1.0 / poles_bio_evo[1]
	tau_rise_inter_bio = -1.0 / poles_inter_evo[0]
	tau_fall_inter_bio = -1.0 / poles_inter_evo[1]
	tau_rise_bio_bio = -1.0 / poles_bio_evo[0]
	tau_fall_bio_bio = -1.0 / poles_bio_evo[1]
	T_inter_bio = 1.0
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
		pass2_sig,
		pass2_freq,
		pass2_seed,
		inter_type,
		bio_type)

	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_train)
	x_bio = sim.data[network.p_bio]
	x_target = sim.data[network.p_target]
	e_bio = nengo.utils.numpy.rmse(x_bio, x_target)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/fb_inter_bio')

	'''
	pass #3: test the accuracy of bio when these decoders/filters
	are used for the recurrent connection
	'''
	d_bio_out = d_bio_evo
	d_inter_bio = d_inter_evo
	d_bio_bio = d_bio_evo
	tau_rise_bio_out = -1.0 / poles_bio_evo[0]
	tau_fall_bio_out = -1.0 / poles_bio_evo[1]
	tau_rise_inter_bio = -1.0 / poles_inter_evo[0]
	tau_fall_inter_bio = -1.0 / poles_inter_evo[1]
	tau_rise_bio_bio = -1.0 / poles_bio_evo[0]
	tau_fall_bio_bio = -1.0 / poles_bio_evo[1]
	T_inter_bio = 0.0
	T_bio_bio = 1.0

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
		pass3_sig,
		pass3_freq,
		pass3_seed,
		inter_type,
		bio_type)

	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_test)
	a_bio = sim.data[network.p_bio_act]
	x_target = sim.data[network.p_target]
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	e_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	e_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	import matplotlib.pyplot as plt
	from nengo.utils.matplotlib import rasterplot
	fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8,16))
	rasterplot(sim.trange(), sim.data[network.p_inter_spikes], ax=ax)
	rasterplot(sim.trange(), sim.data[network.p_bio_spikes], ax=ax2)
	ax.set(title='inter')
	ax2.set(title='bio')
	fig.savefig('plots/fb_bio_bio_spikes')
	from seaborn import kdeplot
	fig, ax = plt.subplots(1, 1)
	kdeplot(d_inter_bio.squeeze(), label='inter_bio')
	kdeplot(d_bio_bio.squeeze(), label='bio_bio')
	ax.legend()
	fig.savefig('plots/fb_bio_bio_decoders')
	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %e_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/fb_bio_bio_estimates')