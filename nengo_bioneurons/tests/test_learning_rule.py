import numpy as np
import nengo
from nengo_bioneurons import BahlNeuron
import matplotlib.pyplot as plt
from seaborn import kdeplot
from nengo.utils.matplotlib import rasterplot

class LearningNode(nengo.Node):
	def __init__(self, n_syn, dim, d_pre, e_bio, g_bio, d_bio, learning_rate, syn_weights=None):
		self.n_syn = n_syn
		self.dim = dim
		self.d_pre = d_pre
		self.e_bio = e_bio
		self.g_bio = g_bio
		self.d_bio = d_bio
		self.learning_rate = learning_rate
		self.syn_weights = syn_weights  # linked to syn_weights in builder
		super(LearningNode, self).__init__(self.update, 
			                               size_in=2*self.dim,
			                               size_out=None)
			                               # size_out=self.e_bio.shape[0])
	def update(self, t, x):  # assume 1D for now
		if self.syn_weights is None:
			return
		# x[:dim] is bio, x[dim:2*dim] is target
		# self.syn_weights[i, pre, syn] = np.dot(d_pre.T[pre], g_bio[i] * e_bio[i])  # keeps weights the same
		for i in range(self.d_bio.shape[0]):
			if x[:self.dim] < x[self.dim: 2*self.dim]:
				if self.d_bio[i] < 0.0:
					# bioneuron_i is overactive, must reduce input current by lowering weight
					for pre in range(self.d_pre.shape[0]):
						for syn in range(self.n_syn):
							self.syn_weights[i, pre, syn] -= self.learning_rate
				elif self.d_bio[i] > 0.0:
					# bioneuron_i is underactive, must increase input current by increasing weight
					for pre in range(self.d_pre.shape[0]):
						for syn in range(self.n_syn):
							self.syn_weights[i, pre, syn] += self.learning_rate
			elif x[:self.dim] > x[self.dim: 2*self.dim]:
				if self.d_bio[i] < 0.0:
					# bioneuron_i is underactive, must increase input current by increasing weight
					for pre in range(self.d_pre.shape[0]):
						for syn in range(self.n_syn):
							self.syn_weights[i, pre, syn] += self.learning_rate
				elif self.d_bio[i] > 0.0:
					# bioneuron_i is overactive, must decrease input current by lowering weight
					for pre in range(self.d_pre.shape[0]):
						for syn in range(self.n_syn):
							self.syn_weights[i, pre, syn] -= self.learning_rate
		return

def test_learning_rule(Simulator):
	pre_neurons = 100
	bio_neurons = 10
	dt = 0.001
	tau = 0.05
	tau_rise = 0.01
	radius = 1
	n_syn = 3
	t_train = 10.0
	t_learn = 10.0
	dim = 1
	sig_freq = 1  # 2 * np.pi
	white_noise_freq = 5
	d_out = np.zeros((bio_neurons, dim))
	learning_rate = 1e-7

	network_seed = 1
	sim_seed = 2
	ens_seed = 3
	conn_seed = 4
	sig_seed = 5

	def make_network(
		d_out,
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
		learning_rate=0):

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
			learning_node = LearningNode(n_syn=n_syn,
										 dim=dim,
										 d_pre=d_pre,
										 e_bio=e_bio,
										 g_bio=g_bio,
										 d_bio=d_out,
										 learning_rate=learning_rate,
										 syn_weights=syn_weights)
			error = nengo.Ensemble(1, 2*dim, neuron_type=nengo.Direct())

			stim_pre = nengo.Connection(stim, pre,
				synapse=None,
				seed=conn_seed)
			pre_bio = nengo.Connection(pre, bio,
				sec='tuft',
				n_syn=n_syn,
				syn_type='Exp2Syn',
				tau_list=[tau_rise, tau],
				seed=conn_seed,
				syn_weights=learning_node.syn_weights)
			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				seed=conn_seed)
			stim_target = nengo.Connection(stim, target,
				synapse=tau)
			bio_node = nengo.Connection(bio, learning_node[:dim],
				synapse=tau,
				solver=nengo.solvers.NoSolver(d_out))
			target_node = nengo.Connection(target, learning_node[dim:2*dim],
				synapse=tau)

			network.p_stim = nengo.Probe(stim)
			network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=tau)
			network.p_pre = nengo.Probe(pre, synapse=tau)
			network.p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=tau)
			network.p_bio = nengo.Probe(bio, synapse=tau, solver=nengo.solvers.NoSolver(d_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=tau)
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_target = nengo.Probe(target, synapse=tau)
			network.pre_bio = pre_bio
			network.pre_lif = pre_lif
			network.learning_node = learning_node

		return network

	''' train d_bio_out when pre_bio syn_weights are w_ij = d_pre . e_bio'''
	syn_weights_init = None
	network = make_network(
		d_out,
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
	d_out_new = nengo.solvers.LstsqL2()(a_bio, x_target)[0]
	x_bio = np.dot(a_bio, d_out_new)
	x_lif = sim.data[network.p_lif]
	err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	err_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	d_pre = sim.data[network.pre_lif].weights.T
	for ens in network.ensembles:
		if ens.label == 'bio':
			e_bio = sim.data[ens].encoders
			g_bio = sim.data[ens].gain
			break
	syn_weights_set = sim.data[network.pre_bio].weights

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), a_bio, label='bio')
	fig.savefig('plots/learning_rule_standard_activities')

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/learning_rule_standard_estimate')

	''' test accuracy of a random initial connection '''
	syn_weights_init = np.random.RandomState(seed=network_seed).uniform(
		-1e-4, 1e-4, size=(bio_neurons, pre_neurons, n_syn))
	network = make_network(
		d_out_new,
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
		learning_rate=0)

	with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
		sim.run(t_learn)
	with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
		sim.run(t_learn)
	a_bio = sim.data[network.p_bio_act]
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	x_target = sim.data[network.p_target]
	err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	err_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), a_bio, label='bio')
	fig.savefig('plots/learning_rule_initial_activities')

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/learning_rule_initial_estimate')


	''' learn from a random initial connection '''
	# syn_weights_init = syn_weights_set
	syn_weights_init = np.random.RandomState(seed=network_seed).uniform(
		-1e-4, 1e-4, size=(bio_neurons, pre_neurons, n_syn))
	network = make_network(
		d_out_new,
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
		learning_rate=learning_rate)

	with Simulator(network, seed=sim_seed, dt=dt, optimize=False) as sim:
		sim.run(t_learn)
	a_bio = sim.data[network.p_bio_act]
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	x_target = sim.data[network.p_target]
	err_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	err_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), a_bio, label='bio')
	fig.savefig('plots/learning_rule_learned_activities')

	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %err_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %err_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/learning_rule_learned_estimate')

	# print 'set', syn_weights_set
	# print 'learning node', network.learning_node.syn_weights
	# print 'conn', sim.data[network.pre_bio].weights
	# assert False