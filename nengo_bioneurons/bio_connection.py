from nengo import Connection as NengoConnection

__all__ = ['BioConnection']


class BioConnection(NengoConnection):
    """
    Extends nengo.Connection to take additional parameters
    and support oracle decoder updating
    """

    def __init__(self, pre, post,
        syn_sec={'apical': {'n_syn': 1, 'syn_type': 'ExpSyn', 'tau': [0.01]}},
        **kwargs):
        """
        syn_sec: the section(s) of the NEURON model on which
                    to distribute n_syn synapses
        """
        self.syn_sec = syn_sec

        super(BioConnection, self).__init__(pre, post, **kwargs)
