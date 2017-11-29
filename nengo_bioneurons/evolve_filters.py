import numpy as np
import nengo
import nengolib
from pathos import multiprocessing as mp
import copy

__all__ = ['build_filter', 'evolve_h_d_out']

def build_filter(zeros, poles):
	"""
	create the transfer function from the passed constants to serve as the filter
	"""
	built_filter = nengolib.signal.LinearSystem((zeros, poles, 1.0))
	built_filter /= built_filter.dcgain
	return built_filter

def evolve_h_d_out(
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
	p_bio_act,
	p_target,
	training_dir,
	training_file):

	def evaluate(inputs):
		network=inputs[0]
		Simulator=inputs[1]
		zeros = inputs[2][0]
		poles = inputs[2][1]
		p_bio_act = inputs[3][0]
		p_target = inputs[3][1]
		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters, as well as other filters that have been
		trained already (these can't be fed into pool.evaluate() without _paramdict errors)
		"""
		filt = build_filter(zeros, poles)
		with network:
			p_bio_act.synapse = filt
		"""
		run the simulation, collect filtered activites,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_evo)
		a_bio = sim.data[p_bio_act]
		x_target = sim.data[p_target]
		if np.sum(a_bio) > 0:
			d_bio = nengo.solvers.LstsqL2()(a_bio, x_target)[0]
		else:
			d_bio = np.zeros((a_bio.shape[1], x_target.shape[1]))
		x_bio = np.dot(a_bio, d_bio)
		e_bio = nengo.utils.numpy.rmse(x_target, x_bio)
		return e_bio

	def get_decoders(inputs, plot=False):
		network=inputs[0]
		Simulator=inputs[1]
		zeros = inputs[2]
		poles = inputs[3]
		p_bio_act = inputs[4]
		p_target = inputs[5]
		"""
		ensure stim outputs the training signal and the bio/alif are assigned
		their particular readout filters
		"""
		filt = build_filter(zeros, poles)
		with network:
			p_bio_act.synapse = filt
		"""
		run the simulation, collect filtered activites,
		and apply the oracle method to calculate readout decoders
		"""
		with Simulator(network, dt=dt, progress_bar=False, seed=sim_seed) as sim:
			sim.run(t_evo)
		a_bio = sim.data[p_bio_act]
		x_target = sim.data[p_target]
		if np.sum(a_bio) > 0:
			d_bio = nengo.solvers.LstsqL2()(a_bio, x_target)[0]
		else:
			d_bio = np.zeros((a_bio.shape[1], x_target.shape[1]))
		x_bio = np.dot(a_bio, d_bio)
		e_bio = nengo.utils.numpy.rmse(x_target, x_bio)

		import matplotlib.pyplot as plt
		figure, ax1 = plt.subplots(1,1)
		ax1.plot(sim.trange(), x_bio, label='bio, e=%.5f' %e_bio)
		ax1.plot(sim.trange(), x_target, label='target')
		ax1.set(xlabel='time (s)', ylabel='activity',
			title='zeros: %s \npoles: %s' %(zeros, poles))
		ax1.legend()
		figure.savefig('plots/evolution/evo_decodes.png')  # %id(p_bio_act)
		figure, ax1 = plt.subplots(1,1)
		ax1.plot(sim.trange(), a_bio, label='bio')
		ax1.set(xlabel='time (s)', ylabel='activity',
			title='zeros: %s \npoles: %s' %(zeros, poles))
		ax1.legend()
		figure.savefig('plots/evolution/evo_activities.png')  # %id(p_bio_act)

		return d_bio

	pool = mp.ProcessingPool(nodes=n_threads)
	rng = np.random.RandomState(seed=evo_seed)

	""" Initialize evolutionary population """
	filter_pop = []
	for p in range(evo_popsize):
		my_zeros= []
		my_poles = []
		for z in zeros_init:
			my_zeros.append(rng.uniform(-z, z))
		for p in poles_init:
			my_poles.append(rng.uniform(0, p))  # poles must be negative
		filter_pop.append([my_zeros, my_poles])


	""" Run evolutionary strategy """
	fit_vs_gen = []
	for g in range(evo_gen):
		probes = [p_bio_act, p_target]
		# reconfigure nengolib synapses to have propper attributes to be passed to pool.map()
		# these synapses are restored to the evolved filter inside evaluate()
		for probe in network.probes:
			if isinstance(probe.synapse, nengolib.signal.LinearSystem):
				try:
					probe.synapse._paramdict = nengo.Lowpass(tau)._paramdict
					probe.synapse.tau = tau
					probe.synapse.default_size_in = 1
					probe.synapse.default_size_out = 1
				except:
					continue
		for conn in network.connections:
			if isinstance(conn.synapse, nengolib.signal.LinearSystem):
				try:
					conn.synapse._paramdict = nengo.Lowpass(tau)._paramdict
					conn.synapse.tau = tau
					conn.synapse.default_size_in = 1
					conn.synapse.default_size_out = 1
				except:
					continue
		inputs = [[network, Simulator, filter_pop[p], probes] for p in range(evo_popsize)]
		# fitnesses = np.array([evaluate(inputs[0]), evaluate(inputs[1]), evaluate(inputs[2])])  # debugging
		fitnesses = np.array(pool.map(evaluate, inputs))
		best_filter = filter_pop[np.argmin(fitnesses)]
		best_fitness = fitnesses[np.argmin(fitnesses)]
		fit_vs_gen.append([best_fitness])
		decay = np.exp(-g / 5.0)
		# decay = 1.0  # off
		""" repopulate filter pops with mutated copies of the best individual """
		filter_pop_new = []
		for p in range(evo_popsize):
			my_zeros = []
			my_poles = []
			for term in range(len(best_filter[0])):
				my_zeros.append(best_filter[0][term] + rng.normal(0, zeros_delta[term]) * decay)  # mutate
			for term in range(len(best_filter[1])):
				my_poles.append(best_filter[1][term] + rng.normal(0, poles_delta[term]) * decay)  # mutate	
			filter_pop_new.append([my_zeros, my_poles])
		filter_pop = filter_pop_new

	""" Grab the best filters and decoders and plot fitness vs generation """
	best_zeros = best_filter[0]
	best_poles = best_filter[1]
	best_d_bio = get_decoders([network, Simulator, best_zeros, best_poles, p_bio_act, p_target], plot=True)

	import matplotlib.pyplot as plt
	figure, ax1 = plt.subplots(1,1)
	ax1.plot(np.arange(0, evo_gen), np.array(fit_vs_gen))
	ax1.set(xlabel='generation', ylabel='fitness')
	ax1.legend()
	figure.savefig('plots/evolution/evo_fit.png')  #  % id(p_bio_act)
	figure, ax1 = plt.subplots(1,1)
	times = np.arange(0, 1e0, 1e-3)
	ax1.plot(times, build_filter(best_zeros, best_poles).impulse(len(times)), label='evolved')
	ax1.plot(times, nengolib.Lowpass(tau).impulse(len(times)), label='lowpass')
	ax1.set(xlabel='time', ylabel='amplitude')
	ax1.legend()
	figure.savefig('plots/evolution/evo_filt.png')  #  % id(p_bio_act)

	np.savez(training_dir+training_file,
		zeros=best_zeros,
		poles=best_poles,
		decoders=best_d_bio)

	return best_zeros, best_poles, best_d_bio