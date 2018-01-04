from nengo import Connection as NengoConnection
from nengo import Node
import numpy as np

__all__ = ['BioConnection', 'LearningNode']

class BioConnection(NengoConnection):
    """
    Extends nengo.Connection to take additional parameters
    and support oracle decoder updating
    """

    def __init__(
        self,
        pre,
        post,
        sec='tuft',
        n_syn=1,
        syn_type='ExpSyn',
        tau_list=[0.01],
        syn_locs=None,
        syn_weights=None,
        learning_node=None,
        **kwargs):
        """
        sec: the section(s) of the NEURON model on which to distribute synapses
        n_syn: the number of synapses to create in this section
        syn_type: ExpSyn for lowpass conductance, Exp2Syn for risetime tau, falltime tau2
        tau_list: list of tau's for syn_type, [tau] or [tau_rise, tau_fall]
        loc: set the locations of the synapses instead of randomly generating them
        weights: set the initial weights instead of randomly generating them
        """
        self.sec = sec
        self.n_syn = n_syn
        self.syn_type = syn_type
        self.tau_list = tau_list
        self.syn_locs = syn_locs
        self.syn_weights = syn_weights
        self.learning_node = learning_node

        super(BioConnection, self).__init__(pre, post, **kwargs)


class LearningNode(Node):
    def __init__(self,
                 conn,
                 n_syn,
                 dim,
                 d_pre,
                 e_bio,
                 g_bio,
                 d_bio,
                 learning_rate,
                 learning_seed=6):
        self.conn = conn
        self.n_syn = n_syn
        self.dim = dim
        self.d_pre = d_pre
        self.e_bio = e_bio
        self.g_bio = g_bio
        self.d_bio = d_bio
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed=learning_seed)
        # weight update retrieved in builder's transmit_spikes 
        self.delta_weights = np.zeros((
            self.d_bio.shape[0], self.d_pre.shape[0], self.n_syn))
        super(LearningNode, self).__init__(self.update, 
                                           size_in=2*self.dim,
                                           size_out=None)
                                           # size_out=self.e_bio.shape[0])
    def update(self, t, x):  # assume 1D for now
        if not self.learning_rate > 0.0:
            return
        # x[:dim] is bio, x[dim:2*dim] is target
        # self.syn_weights[i, pre, syn] = np.dot(d_pre.T[pre], g_bio[i] * e_bio[i])  # keeps weights the same
        error = x[:self.dim] - x[self.dim: 2*self.dim]
        for i in range(self.d_bio.shape[0]):
            if error < 0.0:
                if self.d_bio[i] < 0.0:
                    # bioneuron_i is overactive, must reduce input current by lowering weight
                    for pre in range(self.d_pre.shape[0]):
                        for syn in range(self.n_syn):
                            self.delta_weights[i, pre, syn] = error * self.rng.uniform(0, self.learning_rate)
                elif self.d_bio[i] > 0.0:
                    # bioneuron_i is underactive, must increase input current by increasing weight
                    for pre in range(self.d_pre.shape[0]):
                        for syn in range(self.n_syn):
                            self.delta_weights[i, pre, syn] = -error * self.rng.uniform(0, self.learning_rate)
            elif error > 0.0:
                if self.d_bio[i] < 0.0:
                    # bioneuron_i is underactive, must increase input current by increasing weight
                    for pre in range(self.d_pre.shape[0]):
                        for syn in range(self.n_syn):
                            self.delta_weights[i, pre, syn] = error * self.rng.uniform(0, self.learning_rate)
                elif self.d_bio[i] > 0.0:
                    # bioneuron_i is overactive, must decrease input current by lowering weight
                    for pre in range(self.d_pre.shape[0]):
                        for syn in range(self.n_syn):
                            self.delta_weights[i, pre, syn] = -error * self.rng.uniform(0, self.learning_rate)
        return