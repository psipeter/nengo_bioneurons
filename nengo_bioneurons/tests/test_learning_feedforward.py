import numpy as np
import nengo
from nengo_bioneurons import BahlNeuron, LearningNode, evolve_h_d_out, build_filter
import matplotlib.pyplot as plt
from seaborn import kdeplot
from nengo.utils.matplotlib import rasterplot

def test_learning_feedforward(Simulator):
	pre_neurons = 100
	bio_neurons = 10
	dt = 0.001
	tau = 0.05
	tau_rise = 0.01
	radius = 1
	n_syn = 5
	t_train = 10.0
	t_learn = 30.0
	t_test = 10.0
	dim = 1
	sig_freq = 1  # 2 * np.pi
	white_noise_freq = 5
	d_out = np.zeros((bio_neurons, dim))
	learning_rate = 5e-7

	network_seed = 1
	sim_seed = 2
	ens_seed = 3
	conn_seed = 4
	sig_seed = 5
	learning_seed = 6
	evo_seed = 7

	t_evo = 10.0
	n_threads = 10
	evo_popsize = 10
	evo_gen = 5
	zeros_init = []
	poles_init = [-1e2, -1e2]
	zeros_delta = []
	poles_delta = [1e1, 1e1]
	training_dir = '/home/pduggins/nengo_bioneurons/nengo_bioneurons/tests/filters/'
	training_file = 'test_evolved_readout_learning_rule' + \
		'_%.3f_bioneurons_%.3f_t_evo_%.3f_evo_popsize_%.3f_evo_gen'\
		%(bio_neurons, t_evo, evo_popsize, evo_gen)

	def make_network(
		d_out,
		h_out,
		syn_weights,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		sig_type,
		sig_freq,
		sig_seed,
		d_pre=np.zeros((pre_neurons, dim)),
		e_bio=None,
		g_bio=None,
		learning_rate=0,
		learning_seed=6):

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
				# encoders=nengo.dists.Uniform(-1e0,1e0),
				# gain=nengo.dists.Uniform(-2e2,2e2),
				# bias=nengo.dists.Uniform(-0e1,0e1),
				# bias=nengo.dists.Uniform(-5e-3, 5e-3),
				neuron_type=BahlNeuron(),
				seed=ens_seed,
				label='bio')
			lif = nengo.Ensemble(
				n_neurons=bio_neurons,
				dimensions=dim,
				radius=radius,
				neuron_type=nengo.LIF(),
				seed=ens_seed)
			target = nengo.Node(size_in=dim)


			stim_pre = nengo.Connection(stim, pre,
				synapse=None,
				seed=conn_seed)
			pre_bio = nengo.Connection(pre, bio,
				sec='tuft',
				n_syn=n_syn,
				syn_type='Exp2Syn',
				tau_list=[tau_rise, tau],
				seed=conn_seed,
				syn_weights=syn_weights)
			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				seed=conn_seed)
			stim_target = nengo.Connection(stim, target,
				synapse=tau)
			if learning_rate > 0.0:
				learning_node = LearningNode(conn=pre_bio,
											 n_syn=n_syn,
											 dim=dim,
											 d_pre=d_pre,
											 e_bio=e_bio,
											 g_bio=g_bio,
											 d_bio=d_out,
											 learning_rate=learning_rate,
											 learning_seed=learning_seed)
				pre_bio.learning_node = learning_node
				bio_node = nengo.Connection(bio, learning_node[:dim],
					synapse=h_out,
					solver=nengo.solvers.NoSolver(d_out))
				target_node = nengo.Connection(target, learning_node[dim:2*dim],
					synapse=tau)
				network.learning_node = learning_node

			network.p_stim = nengo.Probe(stim)
			network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=tau)
			network.p_pre = nengo.Probe(pre, synapse=tau)
			network.p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=tau)
			network.p_bio = nengo.Probe(bio, synapse=h_out, solver=nengo.solvers.NoSolver(d_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=tau)
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_target = nengo.Probe(target, synapse=tau)
			network.pre_bio = pre_bio
			network.pre_lif = pre_lif

		return network


	''' train d_bio_out when pre_bio syn_weights are w_ij = d_pre . e_bio'''
	syn_weights_init = None
	try:
		zeros_evo = np.load(training_dir+training_file+'.npz')['zeros']
		poles_evo = np.load(training_dir+training_file+'.npz')['poles']
		d_bio_evo = np.load(training_dir+training_file+'.npz')['decoders']
	except IOError:
		h_out = nengo.Lowpass(tau)
		network = make_network(
			d_out,
			h_out,
			syn_weights_init,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			'sin',
			sig_freq,
			sig_seed)
		zeros_evo, poles_evo, d_bio_evo = evolve_h_d_out(
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
			training_file)

	try:
		zeros_evo = np.load(training_dir+'test_standard_connection.npz')['zeros_evo']
		poles_evo = np.load(training_dir+'test_standard_connection.npz')['poles_evo']
		h_out_new = build_filter(zeros_evo, poles_evo)
		d_out_new = np.load(training_dir+'test_standard_connection.npz')['d_out_new']
		a_bio = np.load(training_dir+'test_standard_connection.npz')['a_bio']
		x_bio = np.load(training_dir+'test_standard_connection.npz')['x_bio']
		x_lif = np.load(training_dir+'test_standard_connection.npz')['x_lif']
		x_target = np.load(training_dir+'test_standard_connection.npz')['x_target']
		err_bio = np.load(training_dir+'test_standard_connection.npz')['err_bio']
		err_lif = np.load(training_dir+'test_standard_connection.npz')['err_lif']
		d_pre = np.load(training_dir+'test_standard_connection.npz')['d_pre']
		e_bio = np.load(training_dir+'test_standard_connection.npz')['e_bio']
		g_bio = np.load(training_dir+'test_standard_connection.npz')['g_bio']
		syn_weights_set = np.load(training_dir+'test_standard_connection.npz')['syn_weights_set']
		times = np.load(training_dir+'test_standard_connection.npz')['times']
	except IOError:
		h_out_new = build_filter(zeros_evo, poles_evo)
		d_out_new = d_bio_evo

		network = make_network(
			d_out_new,
			h_out_new,
			syn_weights_init,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			'sin',
			sig_freq,
			sig_seed)

		with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
			sim.run(t_train)

		a_bio = sim.data[network.p_bio_act]
		x_target = sim.data[network.p_target]
		x_bio = sim.data[network.p_bio]
		x_lif = sim.data[network.p_lif]
		err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
		err_lif = nengo.utils.numpy.rmse(x_lif, x_target)
		times = sim.trange()

		d_pre = sim.data[network.pre_lif].weights.T
		for ens in network.ensembles:
			if ens.label == 'bio':
				e_bio = sim.data[ens].encoders
				g_bio = sim.data[ens].gain
				break
		syn_weights_set = sim.data[network.pre_bio].weights

		fig, ax = plt.subplots(1, 1)
		ax.plot(times, a_bio, label='bio')
		fig.savefig('plots/learning_rule_standard_activities')

		fig, ax = plt.subplots(1, 1)
		ax.plot(times, x_bio, label='bio, e=%.5f' %err_bio)
		ax.plot(times, x_lif, label='lif, e=%.5f' %err_lif)
		ax.plot(times, x_target, label='target')
		ax.legend()
		fig.savefig('plots/learning_rule_standard_estimate')

		np.savez(training_dir+'test_standard_connection.npz',
			zeros_evo=zeros_evo,
			poles_evo=poles_evo,
			d_out_new=d_out_new,
			a_bio=a_bio,
			x_bio=x_bio,
			x_lif=x_lif,
			x_target=x_target,
			err_bio=err_bio,
			err_lif=err_lif,
			d_pre=d_pre,
			e_bio=e_bio,
			g_bio=g_bio,
			syn_weights_set=syn_weights_set,
			times=times)

	''' test accuracy of a random initial connection '''
	syn_weights_init = np.random.RandomState(seed=network_seed).uniform(
		-1e-4, 1e-4, size=(bio_neurons, pre_neurons, n_syn))
	try:
		a_bio = np.load(training_dir+'test_random_connection.npz')['a_bio']
		x_bio = np.load(training_dir+'test_random_connection.npz')['x_bio']
		x_lif = np.load(training_dir+'test_random_connection.npz')['x_lif']
		x_target = np.load(training_dir+'test_random_connection.npz')['x_target']
		err_bio = np.load(training_dir+'test_random_connection.npz')['err_bio']
		err_lif = np.load(training_dir+'test_random_connection.npz')['err_lif']
		times = np.load(training_dir+'test_random_connection.npz')['times']
	except IOError:
		network = make_network(
			d_out_new,
			h_out_new,
			syn_weights_init,
			network_seed,
			sim_seed,
			ens_seed,
			conn_seed,
			'sin',
			sig_freq,
			sig_seed,
			d_pre=d_pre,
			e_bio=e_bio,
			g_bio=g_bio)

		with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
			sim.run(t_learn)
		a_bio = sim.data[network.p_bio_act]
		x_bio = sim.data[network.p_bio]
		x_lif = sim.data[network.p_lif]
		x_target = sim.data[network.p_target]
		err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
		err_lif = nengo.utils.numpy.rmse(x_lif, x_target)
		times = sim.trange()

		np.savez(training_dir+'test_random_connection.npz',
			a_bio=a_bio,
			x_bio=x_bio,
			x_lif=x_lif,
			x_target=x_target,
			err_bio=err_bio,
			err_lif=err_lif,
			times=times)

	fig, ax = plt.subplots(1, 1)
	ax.plot(times, a_bio, label='bio')
	fig.savefig('plots/learning_rule_initial_activities')

	fig, ax = plt.subplots(1, 1)
	ax.plot(times, x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(times, x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(times, x_target, label='target')
	ax.legend()
	fig.savefig('plots/learning_rule_initial_estimate')


	''' learn from a random initial connection '''
	# syn_weights_init = syn_weights_set
	syn_weights_init = np.random.RandomState(seed=network_seed).uniform(
		-1e-4, 1e-4, size=(bio_neurons, pre_neurons, n_syn))
	network = make_network(
		d_out_new,
		h_out_new,
		syn_weights_init,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		'sin',
		sig_freq,
		sig_seed,
		d_pre=d_pre,
		e_bio=e_bio,
		g_bio=g_bio,
		learning_rate=learning_rate,
		learning_seed=learning_seed)

	with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
		sim.run(t_learn)
	a_bio = sim.data[network.p_bio_act]
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	x_target = sim.data[network.p_target]
	err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	err_lif = nengo.utils.numpy.rmse(x_lif, x_target)
	syn_weights_learned = sim.data[network.pre_bio].weights

	# syn_weights_neuron = np.zeros_like(syn_weights_learned)
	# for ens in network.ensembles:
	# 	if ens.label == 'bio':
	# 		bioneurons = sim.data[ens.neurons]
	# for n, nrn in enumerate(bioneurons):
	# 	for ens_pre in nrn.synapses:
	# 		if ens_pre.label == 'pre':
	# 			for pre in range(nrn.synapses[ens_pre].shape[0]):
	# 				for syn in range(nrn.synapses[ens_pre].shape[1]):
	# 					syn_weights_neuron[n, pre, syn] = nrn.synapses[ens_pre][pre][syn].weight
	# print 'learned', syn_weights_learned
	# print 'neuron', syn_weights_neuron
	# assert False

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), a_bio, label='bio')
	fig.savefig('plots/learning_rule_train_activities')

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/learned_rule_train_estimate')

	np.savez(training_dir+'learned_rule_train.npz',
		a_bio=a_bio,
		x_bio=x_bio,
		x_lif=x_lif,
		x_target=x_target,
		err_bio=err_bio,
		err_lif=err_lif,
		times=times,
		syn_weights_learned=syn_weights_learned)


	''' test learned synaptic weights on a novel signal '''
	network = make_network(
		d_out_new,
		h_out_new,
		syn_weights_learned,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		'white_noise',
		white_noise_freq,
		sig_seed)

	with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
		sim.run(t_test)
	a_bio = sim.data[network.p_bio_act]
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	x_target = sim.data[network.p_target]
	err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	err_lif = nengo.utils.numpy.rmse(x_lif, x_target)
	syn_weights_post = sim.data[network.pre_bio].weights

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), a_bio, label='bio')
	fig.savefig('plots/learning_rule_test_activities')

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/learning_rule_test_estimate')

	np.savez(training_dir+'learned_rule_test.npz',
		a_bio=a_bio,
		x_bio=x_bio,
		x_lif=x_lif,
		x_target=x_target,
		err_bio=err_bio,
		err_lif=err_lif,
		times=times)

	assert err_bio < 0.1