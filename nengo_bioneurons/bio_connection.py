from nengo import Connection as NengoConnection

__all__ = ['BioConnection']


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

        super(BioConnection, self).__init__(pre, post, **kwargs)
