import numpy as np
import nengo
from nengo_bioneurons import BahlNeuron, make_tuning_curves

def test_tuning_curves(Simulator, plt):
	pre_neurons = 100
	bio_neurons = 10
	tau = 0.01
	radius = 1
	n_syn = 1
	t_test = 10.0
	dim = 1
	freq = 0.5 * np.pi
	d_out = np.zeros((bio_neurons, dim))

	network_seed = 1
	sim_seed = 2
	ens_seed = 3
	conn_seed = 4
	sig_seed = 5

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
			gain=nengo.dists.Uniform(-2e2,2e2),
			bias=nengo.dists.Uniform(-3e-4,3e-4),
			neuron_type=BahlNeuron(),
			seed=ens_seed,
			label='bio')
		lif = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim,
			radius=radius,
			neuron_type=nengo.LIF(),
			seed=ens_seed)

		stim_pre = nengo.Connection(stim, pre,
			synapse=tau)
		pre_bio = nengo.Connection(pre, bio,
			sec='tuft',
			n_syn=n_syn,
			syn_type='ExpSyn',
			tau_list=[tau],
			seed=conn_seed)
		pre_lif = nengo.Connection(pre, lif,
			synapse=tau)

		network.p_stim = nengo.Probe(stim)
		network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		network.p_pre = nengo.Probe(pre, synapse=tau)
		network.p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		network.p_bio = nengo.Probe(bio, synapse=tau, solver=nengo.solvers.NoSolver(d_out))
		network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		network.p_lif = nengo.Probe(lif, synapse=tau)

	make_tuning_curves(
		network,
		Simulator,
		sim_seed,
		'bio',
		network.p_pre,
		network.p_bio_act,
		t_test)


def test_multi_synapse_section(Simulator, plt):
	pre_neurons = 100
	bio_neurons = 10
	tau = 0.05
	radius = 1
	n_syn = 3
	n_syn2 = 5
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
			seed=bio_seed,
			label='bio')
		lif = nengo.Ensemble(
			n_neurons=bio_neurons,
			dimensions=dim,
			radius=radius,
			neuron_type=nengo.LIF(),
			seed=bio_seed)

		stim_pre = nengo.Connection(stim, pre,
			synapse=tau)
		pre_bio_apical = nengo.Connection(pre, bio,
			sec='apical',
			n_syn=3,
			syn_type='Exp2Syn',
			tau_list=[0.02, 0.05])
		pre_bio_tuft = nengo.Connection(pre, bio,
			sec='tuft',
			n_syn=5,
			syn_type='Exp2Syn',
			tau_list=[0.03, 0.04])
		pre_bio_basal = nengo.Connection(pre, bio,
			sec='basal',
			n_syn=7,
			syn_type='Exp2Syn',
			tau_list=[0.01, 0.06])
		pre_lif = nengo.Connection(pre, lif,
			synapse=tau)

		network.p_stim = nengo.Probe(stim)
		network.p_pre_act = nengo.Probe(pre.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		network.p_pre = nengo.Probe(pre, synapse=tau)
		network.p_bio_act = nengo.Probe(bio.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		network.p_bio = nengo.Probe(bio, synapse=tau, solver=nengo.solvers.NoSolver(d_out))
		network.p_lif_act = nengo.Probe(lif.neurons, 'spikes', synapse=nengo.Lowpass(tau))
		network.p_lif = nengo.Probe(lif, synapse=tau)

	make_tuning_curves(
		network,
		Simulator,
		sim_seed,
		'bio',
		network.p_pre,
		network.p_bio_act,
		t_test)