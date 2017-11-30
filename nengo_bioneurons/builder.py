import numpy as np

import neuron

import nengo
from nengo.base import ObjView
from nengo.builder import Builder, Operator, Signal
from nengo.builder.connection import build_decoders, BuiltConnection
from nengo.builder.ensemble import get_activities
from nengo.exceptions import BuildError
from nengo.utils.builder import full_transform
from nengo.solvers import NoSolver
from nengo.builder.operator import Copy

from nengo_bioneurons.bahl_neuron import BahlNeuron

__all__ = ['test_rates']


class Bahl(object):
    """Adaptor between step_math and NEURON."""

    def __init__(self):
        super(Bahl, self).__init__()
        self.synapses = {}
        self.cell = neuron.h.Bahl()
        self.v_record = neuron.h.Vector()
        self.v_record.record(self.cell.soma(0.5)._ref_v)
        self.ap_counter = neuron.h.APCount(self.cell.soma(0.5))
        self.t_record = neuron.h.Vector()
        self.t_record.record(neuron.h._ref_t)
        self.spikes = neuron.h.Vector()
        self.ap_counter.record(neuron.h.ref(self.spikes))
        self.num_spikes_last = 0
        self._clean = False

    def update(self):
        if self._clean:
            raise RuntimeError("cannot update() after cleanup()")
        count = len(self.spikes) - self.num_spikes_last
        self.num_spikes_last = len(self.spikes)
        return count, np.asarray(self.v_record)[-1]

    def cleanup(self):
        if self._clean:
            raise RuntimeError("cleanup() may only be called once")
        self.v_record.play_remove()
        self.t_record.play_remove()
        self.spikes.play_remove()
        self._clean = True


class ExpSyn(object):
    """
    Conductance-based synapses with exponential decay time tau.
    There are two types, excitatory and inhibitory,
    with different reversal potentials.
    If the synaptic weight is above zero, initialize an excitatory synapse,
    else initialize an inhibitory syanpse with abs(weight).
    """

    def __init__(self, sec, weight, tau, loc, e_exc=0.0, e_inh=-80.0):
        self.tau = tau
        self.loc = loc
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.ExpSyn(sec)
        self.syn.tau = 1000 * self.tau
        self.weight = weight
        if self.weight >= 0.0:
            self.syn.e = self.e_exc
        else:
            self.syn.e = self.e_inh
        # time of spike arrival assigned in nengo step
        self.spike_in = neuron.h.NetCon(None, self.syn)        
        self.spike_in.weight[0] = abs(self.weight)


class Exp2Syn(object):
    """
    Conductance-based synapses with exponential rise time tau and decay time tau2.
    There are two types, excitatory and inhibitory,
    with different reversal potentials.
    If the synaptic weight is above zero, initialize an excitatory synapse,
    else initialize an inhibitory syanpse with abs(weight).
    """

    def __init__(self, sec, weight, tau1, tau2, loc, e_exc=0.0, e_inh=-80.0):
        self.tau1 = tau1
        self.tau2 = tau2
        self.loc = loc
        self.e_exc = e_exc
        self.e_inh = e_inh
        self.syn = neuron.h.Exp2Syn(sec)
        self.syn.tau1 = 1000 * self.tau1
        self.syn.tau2 = 1000 * self.tau2
        self.weight = weight
        if self.weight >= 0.0:
            self.syn.e = self.e_exc
        else:
            self.syn.e = self.e_inh
        # time of spike arrival assigned in nengo step
        self.spike_in = neuron.h.NetCon(None, self.syn)        
        self.spike_in.weight[0] = abs(self.weight)


class SimBioneuron(Operator):
    """
    Operator to simulate the states of a bioensemble through time.
    """

    def __init__(self, neuron_type, neurons, output, voltage, states):
        super(SimBioneuron, self).__init__()
        self.neuron_type = neuron_type
        self.neurons = neurons

        self.reads = [states[0]]
        self.sets = [output, voltage]
        self.updates = []
        self.incs = []

    def make_step(self, signals, dt, rng):
        output = signals[self.output]
        voltage = signals[self.voltage]
        time = signals[self.time]

        def step_nrn():
            self.neuron_type.step_math(dt, output, self.neurons, voltage, time)
        return step_nrn

    @property
    def time(self):
        return self.reads[0]

    @property
    def output(self):
        return self.sets[0]

    @property
    def voltage(self):
        return self.sets[1]


class TransmitSpikes(Operator):
    """
    Operator to deliver spikes from the presynaptic population
    into a bioensemble.
    """

    def __init__(self, ens_pre, ens_post, neurons, spikes, states):
        super(TransmitSpikes, self).__init__()
        self.ens_pre = ens_pre
        self.ens_post = ens_post
        self.neurons = neurons
        self.time = states[0]
        self.reads = [spikes, states[0]]
        self.updates = []
        self.sets = []
        self.incs = []

    @property
    def spikes(self):
        return self.reads[0]

    def make_step(self, signals, dt, rng):
        spikes = signals[self.spikes]
        time = signals[self.time]

        def step():
            t_neuron = (time.item()-dt)*1000
            for n in range(spikes.shape[0]):
                num_spikes = int(spikes[n]*dt + 1e-9)
                for _ in range(num_spikes):
                    for nrn in self.neurons:
                        for syn in nrn.synapses[self.ens_pre][n]:
                            syn.spike_in.event(t_neuron)
        return step

def deref_objview(o):
    return o.obj if isinstance(o, ObjView) else o

@Builder.register(BahlNeuron)
def build_bioneurons(model, neuron_type, neurons):
    ens = neurons.ensemble
    # todo: generalize to new NEURON models specified by neuron_type
    bioneurons = [Bahl() for _ in range(ens.n_neurons)]
    # todo: call user-defined function that introduces variance into specific
    # NEURON parameters in each bioneuron, to encourage heterogeneity
    neuron.init()

    model.sig[neurons]['voltage'] = Signal(
        np.zeros(ens.n_neurons),
        name='%s.voltage' % ens.label)
    op = SimBioneuron(neuron_type=neuron_type,
                       neurons=bioneurons,
                       output=model.sig[neurons]['out'],
                       voltage=model.sig[neurons]['voltage'],
                       states=[model.time])

    # Initialize encoders, gains, and biases according to some heuristics,
    # unless the user has specified them already.
    # Note: setting encoders/gains/biases in this way doesn't really
    # respect the high-level ordering of the nengo build process.
    # This can generate hard-to-track problems related to these attributes.
    # However, setting them like 'neurons' are set below may not be possible
    # because these attributes are used in more places in the build process.
    rng = np.random.RandomState(seed=ens.seed)
    if (hasattr(ens, 'encoders')
            and ens.encoders is not None
            and not isinstance(ens.encoders, np.ndarray)):
        ens.encoders = nengo.dists.get_samples(ens.encoders, ens.n_neurons, ens.dimensions, rng)
    else:
        ens.encoders = gen_encoders(
            ens.n_neurons,
            ens.dimensions,
            ens.radius,
            rng)
    if (hasattr(ens, 'gain')
            and ens.gain is not None
            and not isinstance(ens.gain, np.ndarray)):
        ens.gain = nengo.dists.get_samples(ens.gain, ens.n_neurons, 1, rng)[:,0]
    else:
        ens.gain = gen_gains(
            ens.n_neurons,
            ens.dimensions,
            ens.radius,
            rng)
    if (hasattr(ens, 'bias')
            and ens.bias is not None
            and not isinstance(ens.bias, np.ndarray)):
        ens.bias = nengo.dists.get_samples(ens.bias, ens.n_neurons, 1, rng)[:,0]
    else:
        ens.bias = gen_biases(
            ens.n_neurons,
            ens.dimensions,
            ens.radius,
            rng)

    model.add_op(op)

    assert neurons not in model.params
    model.params[neurons] = bioneurons

    # Build a bias-emulating connection
    build_bias(model, ens, ens.bias)


def build_bias(model, bioensemble, biases):
    rng = np.random.RandomState(bioensemble.seed)
    neurons_lif = 100
    neurons_bio = bioensemble.n_neurons
    tau = 0.01

    lif = nengo.Ensemble(
            neuron_type=nengo.LIF(),
            dimensions=1,
            n_neurons=neurons_lif,
            # seed=bioensemble.seed,
            add_to_container=False)
    model.seeds[lif] = bioensemble.seed  # seeds normally set early in builder
    model.build(lif)  # add to the model
    model.add_op(Copy(Signal(1), model.sig[lif]['in'], inc=True))  # connect input(t)=1
    A = get_activities(model.params[lif],  # grab tuning curve activities
        lif,
        model.params[lif].eval_points)

    # Desired output function Y -- just repeat "bias" m times
    Y = np.tile(biases, (A.shape[0], 1))
    bias_decoders = nengo.solvers.LstsqL2()(A, Y)[0]

    # initialize synaptic locations
    syn_loc = get_synaptic_locations(
        rng,
        neurons_lif,
        neurons_bio,
        n_syn=1)
    syn_weights = np.zeros((
        neurons_bio,
        neurons_lif,
        syn_loc.shape[2]))

    # unit test that synapse and weight arrays are compatible shapes
    if not syn_loc.shape[:-1] == bias_decoders.T.shape:
        raise BuildError("Shape mismatch: syn_loc=%s, bias_decoders=%s"
                         % (syn_loc.shape[:-1], bias_decoders))

    # add synapses to the bioneurons with weights = bias_decoders
    neurons = model.params[bioensemble.neurons]
    for j, bahl in enumerate(neurons):
        assert isinstance(bahl, Bahl)
        loc = syn_loc[j]
        bahl.synapses[lif] = np.empty(
            (loc.shape[0], loc.shape[1]), dtype=object)
        for pre in range(loc.shape[0]):
            for syn in range(loc.shape[1]):
                section = bahl.cell.apical(loc[pre, syn])
                # w_ij = np.dot(decoders[pre], gain * encoder)
                w_ij = bias_decoders[pre, j]
                syn_weights[j, pre, syn] = w_ij
                synapse = ExpSyn(section, w_ij, tau, loc[pre, syn])
                bahl.synapses[lif][pre][syn] = synapse
    neuron.init()

    model.add_op(TransmitSpikes(
        lif, bioensemble, neurons,
        model.sig[lif]['out'], states=[model.time]))
    # todo: update model.params
    # model.params['bias'] = BuiltConnection(eval_points=eval_points,
    #                                      solver_info=solver_info,
    #                                      transform=transform,
    #                                      weights=syn_weights)

@Builder.register(nengo.Connection)
def build_connection(model, conn):
    """
    Method to build connections into bioensembles.
    Calculates the optimal decoders for this conneciton as though
    the presynaptic ensemble was connecting to a hypothetical LIF ensemble.
    These decoders are used to calculate the synaptic weights
    in init_connection().
    Adds a transmit_spike operator for this connection to the model
    """

    conn_pre = deref_objview(conn.pre)
    conn_post = deref_objview(conn.post)

    if isinstance(conn_pre, nengo.Ensemble) and \
            isinstance(conn_pre.neuron_type, BahlNeuron):
        # todo: generalize to custom online solvers
        if not isinstance(conn.solver, NoSolver):
            raise BuildError("Connections from bioneurons must provide a NoSolver"
                            " (got %s from %s to %s)" 
                            % (conn.solver, conn_pre, conn_post))

    if isinstance(conn_post, nengo.Ensemble) and \
       isinstance(conn_post.neuron_type, BahlNeuron):
        # TODO: other error handling?
        # TODO: detect this earlier inside BioConnection __init__
        if not isinstance(conn_pre, nengo.Ensemble) or \
                'spikes' not in conn_pre.neuron_type.probeable:
            raise BuildError("May only connect spiking neurons (pre=%s) to "
                             "bioneurons (post=%s)" % (conn_pre, conn_post))

        rng = np.random.RandomState(model.seeds[conn])
        model.sig[conn]['in'] = model.sig[conn_pre]['out']
        transform = full_transform(conn, slice_pre=False)

        """
        Given a parcicular connection, labeled by conn.pre,
        repeat the following procedure for each section that the user
        has specified will be synapsed onto (e.g. apical, tuft, basal dend.)
            Grab the initial decoders
            Generate locations for synapses,
            Create synapses with weight equal to
            w_ij=np.dot(d_i,alpha_j*e_j)/n_syn, where
                - d_i is the initial decoder,
                - e_j is the single bioneuron encoder
                - alpha_j is the single bioneuron gain
                - n_syn normalizes total input current for multiple-synapse conns
            Add synapses to bioneuron.synapses
        Finally call neuron.init().
        """
        for sec in conn.syn_sec:
            # todo: warn users about how to specify tau properly
            # initialize synaptic locations and weights for this section
            syn_loc = get_synaptic_locations(
                rng,
                conn_pre.n_neurons,
                conn_post.n_neurons,
                conn.syn_sec[sec]['n_syn'])
            syn_weights = np.zeros((
                conn_post.n_neurons,
                conn_pre.n_neurons,
                syn_loc.shape[2]))

            # Grab decoders from the specified solver (usually nengo.solvers.NoSolver(d))
            eval_points, decoders, solver_info = build_decoders(
                    model, conn, rng, transform)

            # normalize the area under the ExpSyn curve to compensate for effect of tau
            times = np.arange(0, 1.0, 0.001)
            k_norm = np.linalg.norm(np.exp((-times/conn.syn_sec[sec]['tau'][0])),1)

            # todo: synaptic gains and encoders
            neurons = model.params[conn_post.neurons]  # set in build_bioneurons
            for j, bahl in enumerate(neurons):
                assert isinstance(bahl, Bahl)
                loc = syn_loc[j]
                encoder = conn_post.encoders[j]
                gain = conn_post.gain[j]
                bahl.synapses[conn_pre] = np.empty(
                    (loc.shape[0], loc.shape[1]), dtype=object)
                for pre in range(loc.shape[0]):
                    for syn in range(loc.shape[1]):
                        if sec == 'apical':
                            section = bahl.cell.apical(loc[pre, syn])
                        elif sec == 'tuft':
                            section = bahl.cell.tuft(loc[pre, syn])
                        elif sec == 'basal':
                            section = bahl.cell.basal(loc[pre, syn])
                        w_ij = np.dot(decoders.T[pre], gain * encoder)
                        w_ij = w_ij / conn.syn_sec[sec]['n_syn'] / k_norm
                        syn_weights[j, pre, syn] = w_ij
                        if conn.syn_sec[sec]['syn_type'] == 'ExpSyn':
                            tau = conn.syn_sec[sec]['tau'][0]
                            synapse = ExpSyn(section, w_ij, tau, loc[pre, syn])
                        elif conn.syn_sec[sec]['syn_type'] == 'Exp2Syn':
                            tau1 = conn.syn_sec[sec]['tau'][0]
                            tau2 = conn.syn_sec[sec]['tau'][1]
                            synapse = Exp2Syn(section, w_ij, tau1, tau2, loc[pre, syn])
                        bahl.synapses[conn_pre][pre][syn] = synapse
        neuron.init()

        model.add_op(TransmitSpikes(
            conn_pre, conn_post, neurons,
            model.sig[conn_pre]['out'], states=[model.time]))
        model.params[conn] = BuiltConnection(eval_points=eval_points,
                                             solver_info=solver_info,
                                             transform=transform,
                                             weights=syn_weights)
        # todo: test if computed weights produce
        # heterogeneous tuning curves with plausible firing rates

    else:  # normal connection
        return nengo.builder.connection.build_connection(model, conn)

def gen_encoders(n_neurons, dimensions, radius, rng):
    enc_mag = 1.0 * radius
    encoders = rng.uniform(-enc_mag, enc_mag, size=(n_neurons, dimensions))
    return encoders.astype(float)

def gen_gains(n_neurons, dimensions, radius, rng):
    gain_mag = 1e2 * radius
    gains = rng.uniform(-gain_mag, gain_mag, size=n_neurons)
    return gains

def gen_biases(n_neurons, dimensions, radius, rng):
    bias_mag = 3e0 * radius
    biases = rng.uniform(-bias_mag, bias_mag, size=n_neurons)
    return biases

def get_synaptic_locations(rng, pre_neurons, n_neurons, n_syn):
    """Choose one:"""
    # todo: make syn_distribution an optional parameters of nengo.Connection
    # unique locations per connection and per bioneuron (uses conn's rng)
    syn_locations = rng.uniform(0, 1, size=(n_neurons, pre_neurons, n_syn))
    return syn_locations

def test_rates(network, Simulator, sim_seed, label, p_pre, p_bio_act, t_test):
    import warnings
    min_rate = 10
    max_rate = 100

    with Simulator(network, seed=sim_seed) as sim:
        sim.run(t_test)

    for ens in network.ensembles:
        if ens.label == label:
            encoders = sim.data[ens].encoders
            break

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)

    n_eval_points = 20
    xhat_pre = sim.data[p_pre]
    act_bio = sim.data[p_bio_act]
    for i in range(sim.data[p_bio_act].shape[1]):
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

        bioplot = ax.plot(x_dot_e_vals[:-2], Hz_mean[:-2], label='%s' %i)
        ax.fill_between(x_dot_e_vals[:-2],
            Hz_mean[:-2]+Hz_stddev[:-2],
            Hz_mean[:-2]-Hz_stddev[:-2],
            alpha=0.5)

        if np.any(Hz_mean[:-2]+Hz_stddev[:-2] > max_rate):
            warnings.warn('warning: neuron %s over max_rate' %i)
        if np.all(Hz_mean[:-2]+Hz_stddev[:-2] < min_rate):
            warnings.warn('warning: neuron %s under min_rate' %i)

    ax.legend()
    fig.savefig('plots/test_rates.png')