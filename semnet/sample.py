"""
These utility functions were used for generating negative examples that help denoise LSR's for the :mod:`semnet.lsr` module.
"""

import numpy as np
import pickle
import gzip
import random

""" Load dictionaries that help keep track of names and types """
with gzip.open('../semnet/data/name2type.pkl.gz', 'rb') as file:
   	name2type = pickle.load(file)
with gzip.open('../semnet/data/type2names.pkl.gz', 'rb') as file:
	type2names = pickle.load(file)


def get_negative_examples(examples, hold=None, seed=None):
	"""
	Randomly replaces either the source or the target node of the training example with another node of the same type in the network.

	.. warning:: This function didn't end up working very well because the random replacement was typically not well connected with the example node that wasn't replaced. 

	Parameters
	----------
		examples: list of str pairs
			A list of lenth-2 lists of CUI strings. These example pairs are chosen manually by the user.

		hold: str, 's' or 't'
			A character that determines whether the source or the target is held constant.

		seed: int
			The random seed for choosing the example pairs.

	Returns
	-------
		neg_ex: list of str pairs
			A list of length-2 lists of strings. Each of these example pairs has one node from the example set and one node chosen randomly from the graph.
	"""

	if seed:
		np.random.seed(seed)

	neg_ex = [None]*len(examples)

	for i, ex in enumerate(examples):
		s_type = name2type[ex[0]]
		t_type = name2type[ex[1]]

		s_neg_ix = np.random.randint(0, len(type2names[s_type]))
		t_neg_ix = np.random.randint(0, len(type2names[t_type]))

		s_neg_ex = type2names[s_type][s_neg_ix]
		t_neg_ex = type2names[t_type][t_neg_ix]

		# Hold the source or the target constant
		if hold == 's':
			s_neg_ex = ex[0]
		elif hold =='t':
			t_neg_ex = ex[1]

		neg_ex[i] = [s_neg_ex, t_neg_ex]

	return neg_ex

def generate_negative_examples(examples, hold=None, seed=None):
	"""
	Randomly shuffles example pairs. 

	.. note:: This function is more recent.
	
	Parameters
	----------
		examples: list of str pairs
			A list of lenth-2 lists of CUI strings. These example pairs are chosen manually by the user.

		hold: str, 's' or 't'
			A character that determines whether the source or the target is held constant.

		seed: int
			The random seed for choosing the example pairs.

	Returns
	-------
		neg_ex: list of str pairs
			A list of length-2 lists of strings. Each of these example nodes has been re-paired with another node from the example set.=
	"""

	if seed:
		random.seed(seed)

	# Transpose and shuffle the sources and/or targets
	neg_ex = list(map(list, zip(*examples))).copy()

	if hold == 's':
		random.shuffle(neg_ex[1])
	elif hold == 't':
		random.shuffle(neg_ex[0])
	else:
		random.shuffle(neg_ex[0])
		random.shuffle(neg_ex[1])

	return list(map(list, zip(*neg_ex)))
