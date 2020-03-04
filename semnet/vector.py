"""
These utility functions turn sets of counters and weighted counts into vectors. They were primarily used with the :mod:`semnet.lsr` module.
"""

import numpy as np

def vectorize_example(counts, mp2ix):
	"""
	Turns a Counter of metapaths from an example query into a unit vector, indexed by a dictionary.

	Parameters
	----------
		counts: dict
			A dictionary that maps metapath strings to their counts.

		mp2ix: dict
			A dictionary that maps metapath strings to their desired indices in a vector.

	Returns
	-------
		vec: np.array
			A normalized vector of metapath counts.
	"""

	vec = np.zeros(len(mp2ix))
	for mp, ct in counts.items():
		ix = mp2ix[mp]
		vec[ix] = ct

	norm = np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else 1

	return vec / norm


def vectorize_results(counters_list):
	"""
	Turns a list of metapath counters into a ``numpy`` array.

	.. note:: This is the main function in this module.
	
	Parameters
	----------

		counters_list: list of Counter
			A list of metapath counters that describe pairwise relationships between source and target nodes.

	Returns
	-------
		array: np.array
			A row-normalized matrix, where the rows represent metapaths between pairwise latent semantic relaionship examples and the columns represent metapaths counts.

		mp2ix: dict
			A dictionary that maps metapaths to column indices.
	"""

	mp_count_mat = [None]*len(counters_list)
	mp_vocab = get_mp_vocab(counters_list)
	mp2ix = {mp:ix for ix, mp in enumerate(mp_vocab)}
	for i, ex in enumerate(counters_list):
		mp_count_mat[i] = vectorize_example(ex, mp2ix)

	return np.array(mp_count_mat), mp2ix


def get_mp_vocab(query_data):
	"""
	Returns an ordered list of metapaths that occurred in all queries.

	Parameters
	----------
		query_data: list of Counter
			A list of metapath counters.
		
	Returns
	-------
		sorted_mps: list
			A sorted set of all metapaths found between all source nodes and all target nodes.
	"""

	mp_vocab = {mp for ex in query_data for mp in list(ex.keys())}

	return sorted(list(mp_vocab))
