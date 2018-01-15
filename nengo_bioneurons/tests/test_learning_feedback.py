import numpy as np
import nengo
from nengolib.signal import s
from nengolib.synapses import Lowpass
from nengo_bioneurons import BahlNeuron, LearningNode, evolve_h_d_out, build_filter
import matplotlib.pyplot as plt
from seaborn import kdeplot
from nengo.utils.matplotlib import rasterplot

def test_learning_feedback(Simulator):
	pre_neurons = 100
	bio_neurons = 100
	dt = 0.001
	tau = 0.05
	radius = 1
	n_syn = 5
	dim = 1

	t_train = 10.0
	t_learn = 10.0
	t_test = 10.0
	t_evo = 10.0

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
	learning_seed = 7

	n_threads = 10
	evo_popsize = 10
	evo_gen = 3  # 4
	zeros_init = []
	poles_init = [-1e2, -1e2]
	zeros_delta = []
	poles_delta = [1e1, 1e1]
	data_dir = '/home/pduggins/nengo_bioneurons/nengo_bioneurons/tests/filters/'
	filter_dir = 'test_learning_feedback_filter_bio_4'  # bio_4, 10s 3gen
	ff_dir = 'test_learning_feedback_ff_bio_4'  # bio_4
	fb_dir = 'test_learning_feedback_fb_bio_8'
	test_dir = 'test_learning_feedback_test_bio_9'  # 6

	bio_type = BahlNeuron()  # nengo.AdaptiveLIF(tau_n=0.1, inc_n=0.01)  #  

	def make_network(
		d_bio_out,
		tau_rise,
		tau_fall,
		T_pre_bio,
		T_bio_bio,
		h_stim_target,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		learning_seed,
		learning_rate,
		sig_type,
		sig_freq,
		sig_seed,
		bio_type,
		d_pre_bio=np.zeros((pre_neurons, dim)),
		e_bio=None,
		g_bio=None,
		syn_weights=None):

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
				transform=T_pre_bio,
				seed=conn_seed)
			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				transform=T_pre_bio,
				seed=conn_seed)
			stim_target = nengo.Connection(stim, target,
				synapse=h_stim_target)

			bio_bio = nengo.Connection(bio, bio,
				sec='tuft',
				n_syn=n_syn,
				syn_type='Exp2Syn',
				tau_list=[tau_rise, tau_fall],
				synapse=build_filter([], [-1.0/tau_rise, -1.0/tau_fall]),  # for alif testing
				transform=T_bio_bio,
				solver=nengo.solvers.NoSolver(d_bio_out),
				syn_weights=syn_weights,
				seed=conn_seed)
			lif_lif = nengo.Connection(lif, lif,
				synapse=tau,
				transform=T_bio_bio,
				seed=conn_seed)

			if learning_rate > 0.0:
				learning_node = LearningNode(conn=bio_bio,
											 n_syn=n_syn,
											 dim=dim,
											 d_pre=d_pre_bio,
											 e_bio=e_bio,
											 g_bio=g_bio,
											 d_bio=d_bio_out,
											 learning_rate=learning_rate,
											 learning_seed=learning_seed)
				bio_bio.learning_node = learning_node
				bio_node = nengo.Connection(bio, learning_node[:dim],
					synapse=build_filter([], [-1.0/tau_rise, -1.0/tau_fall]),
					solver=nengo.solvers.NoSolver(d_bio_out))
				target_node = nengo.Connection(target, learning_node[dim:2*dim],
					synapse=tau)
				network.learning_node = learning_node

			network.p_stim = nengo.Probe(stim)
			network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=tau)
			network.p_pre = nengo.Probe(pre, synapse=tau)
			network.p_bio_act = nengo.Probe(bio.neurons, 'spikes',
				synapse=build_filter([], [-1.0/tau_rise, -1.0/tau_fall]))
			network.p_bio = nengo.Probe(bio,
				synapse=build_filter([], [-1.0/tau_rise, -1.0/tau_fall]),
				solver=nengo.solvers.NoSolver(d_bio_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=tau)
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_target = nengo.Probe(target, synapse=tau)
			network.pre_lif = pre_lif
			network.bio_bio = bio_bio
			network.p_bio_spikes = nengo.Probe(bio.neurons, 'spikes', synapse=None)

		return network

	'''
	pass #1: compute d_bio_out using activities and filtered target
	since bio will be adapting, we need to evolve readout filters and decoders,
	which will later be used on the synapses of the bio_bio connection
	'''
	learning_rate = 0.0
	h_stim_target = tau
	try:
		zeros_bio_evo = np.load(data_dir+filter_dir+'.npz')['zeros']
		poles_bio_evo = np.load(data_dir+filter_dir+'.npz')['poles']
		d_bio_evo = np.load(data_dir+filter_dir+'.npz')['decoders']
	except IOError:
		d_bio_out = np.zeros((bio_neurons, dim))
		tau_rise = tau
		tau_fall = tau
		T_pre_bio = 1.0
		T_bio_bio = 0.0

		network = make_network(
			d_bio_out,
			tau_rise,
			tau_fall,
			T_pre_bio,
			T_bio_bio,
			h_stim_target,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			learning_seed,
			learning_rate,
			pass1_sig,
			pass1_freq,
			pass1_seed,
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
			data_dir,
			filter_dir)

	# Test the accuracy of the feedforward decode
	try:
		s_bio = np.load(data_dir+ff_dir+'.npz')['s_bio']
		a_bio = np.load(data_dir+ff_dir+'.npz')['a_bio']
		x_bio = np.load(data_dir+ff_dir+'.npz')['x_bio']
		x_target = np.load(data_dir+ff_dir+'.npz')['x_target']
		err_bio = np.load(data_dir+ff_dir+'.npz')['err_bio']
		times = np.load(data_dir+ff_dir+'.npz')['times']	
		d_pre_bio = np.load(data_dir+ff_dir+'.npz')['d_pre_bio']	
	except IOError:
		d_bio_out = d_bio_evo
		tau_rise = -1.0 / poles_bio_evo[0]
		tau_fall = -1.0 / poles_bio_evo[1]
		T_pre_bio = 1.0
		T_bio_bio = 0.0

		network = make_network(
			d_bio_out,
			tau_rise,
			tau_fall,
			T_pre_bio,
			T_bio_bio,
			h_stim_target,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			learning_seed,
			learning_rate,
			pass1_sig,
			pass1_freq,
			pass1_seed,
			bio_type)

		with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
			sim.run(t_train)
		times = sim.trange()
		s_bio = sim.data[network.p_bio_spikes]
		a_bio = sim.data[network.p_bio_act]
		x_bio = sim.data[network.p_bio]
		x_target = sim.data[network.p_target]
		err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
		d_pre_bio = sim.data[network.pre_lif].weights.T  # readout decoders of pre

		np.savez(data_dir+ff_dir+'.npz',
			s_bio=s_bio,
			a_bio=a_bio,
			x_bio=x_bio,
			x_target=x_target,
			err_bio=err_bio,
			times=times,
			d_pre_bio=d_pre_bio)

	fig, ax = plt.subplots(1, 1)
	ax.plot(times, x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(times, x_target, label='target')
	ax.legend()
	fig.savefig('plots/test_learning_feedback_ff_bio')


	'''
	pass #2: using the d_bio_out and h_bio_out to decode bioneurons' spikes
	and to initialize the bio_bio connection, using the learning_node to
	update the bio_bio synaptic weights 
	'''
	learning_rate = 5e-5
	h_stim_target = 1/s
	# Test the accuracy of the feedforward decode
	try:
		s_bio = np.load(data_dir+fb_dir+'.npz')['s_bio']
		a_bio = np.load(data_dir+fb_dir+'.npz')['a_bio']
		x_bio = np.load(data_dir+fb_dir+'.npz')['x_bio']
		x_lif = np.load(data_dir+fb_dir+'.npz')['x_lif']
		x_target = np.load(data_dir+fb_dir+'.npz')['x_target']
		err_bio = np.load(data_dir+fb_dir+'.npz')['err_bio']
		err_lif = np.load(data_dir+fb_dir+'.npz')['err_lif']
		times = np.load(data_dir+fb_dir+'.npz')['times']
		syn_weights = np.load(data_dir+fb_dir+'.npz')['syn_weights']
	except IOError:
		d_bio_out = d_bio_evo
		tau_rise = -1.0 / poles_bio_evo[0]
		tau_fall = -1.0 / poles_bio_evo[1]
		T_pre_bio = tau
		T_bio_bio = 1.0

		network = make_network(
			d_bio_out,
			tau_rise,
			tau_fall,
			T_pre_bio,
			T_bio_bio,
			h_stim_target,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			learning_seed,
			learning_rate,
			pass2_sig,
			pass2_freq,
			pass2_seed,
			bio_type,
			d_pre_bio=d_pre_bio)

		with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
			sim.run(t_learn)
		times = sim.trange()
		s_bio = sim.data[network.p_bio_spikes]
		a_bio = sim.data[network.p_bio_act]
		x_bio = sim.data[network.p_bio]
		x_lif = sim.data[network.p_lif]
		x_target = sim.data[network.p_target]
		err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
		err_lif = nengo.utils.numpy.rmse(x_lif, x_target)
		syn_weights = sim.data[network.bio_bio].weights

		np.savez(data_dir+fb_dir+'.npz',
			s_bio=s_bio,
			a_bio=a_bio,
			x_bio=x_bio,
			x_lif=x_lif,
			x_target=x_target,
			err_bio=err_bio,
			err_lif=err_lif,
			times=times,
			syn_weights=syn_weights)

	fig, ax = plt.subplots(1, 1)
	ax.plot(times, x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(times, x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(times, x_target, label='target')
	ax.legend()
	fig.savefig('plots/test_learning_feedback_fb_bio')

	'''
	pass #3: test the trained weights on a novel signal
	'''
	learning_rate = 0.0
	h_stim_target = 1/s
	# Test the accuracy of the feedforward decode
	try:
		s_bio = np.load(data_dir+test_dir+'.npz')['s_bio']
		a_bio = np.load(data_dir+test_dir+'.npz')['a_bio']
		x_bio = np.load(data_dir+test_dir+'.npz')['x_bio']
		x_lif = np.load(data_dir+test_dir+'.npz')['x_lif']
		x_target = np.load(data_dir+test_dir+'.npz')['x_target']
		err_bio = np.load(data_dir+test_dir+'.npz')['err_bio']
		times = np.load(data_dir+test_dir+'.npz')['times']
	except IOError:
		d_bio_out = d_bio_evo
		tau_rise = -1.0 / poles_bio_evo[0]
		tau_fall = -1.0 / poles_bio_evo[1]
		T_pre_bio = tau
		T_bio_bio = 1.0

		network = make_network(
			d_bio_out,
			tau_rise,
			tau_fall,
			T_pre_bio,
			T_bio_bio,
			h_stim_target,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			learning_seed,
			learning_rate,
			pass3_sig,
			pass3_freq,
			pass3_seed,
			bio_type,
			d_pre_bio=d_pre_bio,
			syn_weights=syn_weights)

		with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
			sim.run(t_test)
		times = sim.trange()
		s_bio = sim.data[network.p_bio_spikes]
		a_bio = sim.data[network.p_bio_act]
		x_bio = sim.data[network.p_bio]
		x_lif = sim.data[network.p_lif]
		x_target = sim.data[network.p_target]
		err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
		err_lif = nengo.utils.numpy.rmse(x_lif, x_target)
		syn_weights = sim.data[network.bio_bio].weights

		encoders = None
		a_bio = sim.data[network.p_bio_act]
		x_pre = sim.data[network.p_pre]
		for ens in network.ensembles:
			if ens.label == 'bio':
				encoders = sim.data[ens].encoders
				break
		plot_tuning_curves(encoders, x_pre, a_bio, n_neurons=20)

		np.savez(data_dir+test_dir+'.npz',
			s_bio=s_bio,
			a_bio=a_bio,
			x_bio=x_bio,
			x_lif=x_lif,
			x_target=x_target,
			err_bio=err_bio,
			err_lif=err_lif,
			times=times)

	fig, ax = plt.subplots(1, 1)
	ax.plot(times, x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(times, x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(times, x_target, label='target')
	ax.legend()
	fig.savefig('plots/test_learning_feedback_test_bio')

	assert err_bio < 0.1