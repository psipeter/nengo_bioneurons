import numpy as np
import nengo
from nengo_bioneurons import BahlNeuron, build_filter, evolve_h_d_out

def test_basic_readout(Simulator, plt):
	pre_neurons = 100
	bio_neurons = 10
	dt = 0.001
	tau = 0.05
	radius = 1
	n_syn = 1
	t_train = 3.0
	t_test = 3.0
	dim = 1
	sig_freq = 2 * np.pi
	d_out = np.zeros((bio_neurons, dim))

	network_seed = 1
	sim_seed = 2
	ens_seed = 3
	conn_seed = 4
	sig_seed = 5

	def make_network(
		d_out,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		sig_type,
		sig_freq,
		sig_seed):

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
				# bias=nengo.dists.Uniform(-1e0,1e0),
				neuron_type=BahlNeuron(),
				seed=ens_seed)
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
				synapse=tau,
				n_syn=n_syn,
				seed=conn_seed)
			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				seed=conn_seed)
			stim_target = nengo.Connection(stim, target,
				synapse=tau)

			network.p_stim = nengo.Probe(stim)
			network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=nengo.Lowpass(tau))
			network.p_pre = nengo.Probe(pre, synapse=tau)
			network.p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=nengo.Lowpass(tau))
			network.p_bio = nengo.Probe(bio, synapse=tau, solver=nengo.solvers.NoSolver(d_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=nengo.Lowpass(tau))
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_target = nengo.Probe(target, synapse=tau)

		return network

	''' TRAIN '''
	network = make_network(
		d_out,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		'sin',
		sig_freq,
		sig_seed)
	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_train)
	a_bio = sim.data[network.p_bio_act]
	x_target = sim.data[network.p_target]
	d_out_new = nengo.solvers.LstsqL2()(a_bio, x_target)[0]
	x_bio = np.dot(a_bio, d_out_new)
	x_lif = sim.data[network.p_lif]
	e_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	e_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %e_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/basic_feedforward_train')

	''' TEST '''
	network = make_network(
		d_out_new,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		'cos',
		sig_freq,
		sig_seed)
	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_test)
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	x_target = sim.data[network.p_target]
	e_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	e_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %e_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/basic_feedforward_test')


def test_evolved_feedforward(Simulator, plt):
	pre_neurons = 100
	bio_neurons = 10
	dt = 0.001
	tau = 0.05
	radius = 1
	n_syn = 1
	t_train = 5.0
	t_test = 5.0
	dim = 1
	sig_freq = np.pi
	d_out = np.zeros((bio_neurons, dim))

	network_seed = 1
	sim_seed = 2
	ens_seed = 3
	conn_seed = 4
	sig_seed = 5
	evo_seed = 6

	t_evo = 5.0
	n_threads = 10
	evo_popsize = 10
	evo_gen = 10
	zeros_init = []
	poles_init = [-1e2, -1e2]
	zeros_delta = []
	poles_delta = [1e1, 1e1]
	training_dir = '/home/pduggins/nengo_bioneurons/nengo_bioneurons/tests/filters/'
	training_file = 'test_evolved_readout_feedforward' + \
		'_%.3f_bioneurons_%.3f_t_evo_%.3f_evo_popsize_%.3f_evo_gen'\
		%(bio_neurons, t_evo, evo_popsize, evo_gen)

	def make_network(
		d_out,
		h_out,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		sig_type,
		sig_freq,
		sig_seed):

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
				# bias=nengo.dists.Uniform(-1e0,1e0),
				neuron_type=BahlNeuron(),
				seed=ens_seed)
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
				synapse=tau,
				n_syn=n_syn,
				seed=conn_seed)
			pre_lif = nengo.Connection(pre, lif,
				synapse=tau,
				seed=conn_seed)
			stim_target = nengo.Connection(stim, target,
				synapse=tau)

			network.p_stim = nengo.Probe(stim)
			network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=tau)
			network.p_pre = nengo.Probe(pre, synapse=tau)
			network.p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=h_out)
			network.p_bio = nengo.Probe(bio, synapse=h_out, solver=nengo.solvers.NoSolver(d_out))
			network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=tau)
			network.p_lif = nengo.Probe(lif, synapse=tau)
			network.p_target = nengo.Probe(target, synapse=tau)

		return network

	''' TRAIN '''
	h_out = nengo.Lowpass(tau)
	network = make_network(
		d_out,
		h_out,
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

	h_out_new = build_filter(zeros_evo, poles_evo)
	d_out_new = d_bio_evo
	network = make_network(
		d_out_new,
		h_out_new,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		'sin',
		sig_freq,
		sig_seed)
	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_train)
	x_target = sim.data[network.p_target]
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	e_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	e_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %e_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/evolved_feedforward_train')

	''' TEST '''
	sig_freq = np.pi
	network = make_network(
		d_out_new,
		h_out_new,
		network_seed,
		sim_seed,
		ens_seed,
		conn_seed,
		'sin',
		sig_freq,
		sig_seed)
	with Simulator(network, seed=sim_seed, dt=dt) as sim:
		sim.run(t_test)
	x_bio = sim.data[network.p_bio]
	x_lif = sim.data[network.p_lif]
	x_target = sim.data[network.p_target]
	e_bio = nengo.utils.numpy.rmse(x_bio, x_target)
	e_lif = nengo.utils.numpy.rmse(x_lif, x_target)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 1)
	ax.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
	ax.plot(sim.trange(), x_lif, label='lif, e=%.5f' %e_lif)
	ax.plot(sim.trange(), x_target, label='target')
	ax.legend()
	fig.savefig('plots/evolved_feedforward_test')