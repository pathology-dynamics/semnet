"""
:term:`Latent Semantic Relationships` (LSRs) are underlying semantics between
nodes that are not directly apparent in the graph edges [1]_. They may be
represented by weighted sets of metaedges spanning multiple nodes.

How to Use

1. Manually create a set of example pairs that exemplify the relationship of interest.
2. Start a Neo4j instance containing the semnet database at `localhost:7474`.
3. Generate negative examples based on the pairs using :mod:`semnet.sample`.
4. Submit queries to the database to compute metapath-based count metrics using ``py2neo``.
5. Use metapath counters between pairs to compute the latent semantic relationship using :func:`semnet.lsr.get_lsr()`.

.. [1] Wang, Chenguang, et al. "Relsim: relation similarity search in schema-rich heterogeneous information networks." Proceedings of the 2016 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2016.
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

from semnet.vector import vectorize_results

def compute_mp_weights(mp_count_mat, c):
	"""
	Computes the optimal LSR weights by solving the linear programming problem 
	formulated by Wang et al. 2016.

	Parameters
	----------
		mp_count_mat: np.array
			A ``numpy`` array of metapath counts, where the rows are training examples and the columns are metapath counts.
		
		c: float
			A slack variable tuning parameter where :math:`c\\in(0,1]`. When :math:`c=1`, "this model will essentially maximize the weights of metapaths that have the biggest difference between positive and negative examples. If :math:`c<1`, then the model will consider the accident that positive and negative examples share the important meta-paths, and that some of the important metapaths are missing in some positive examples" [1]_.

	Returns
	-------
		result: scipy.optimize.OptimizeResult
			The result of the optimization, which contains the latent semantic relationship vector in ``x``.
	"""

	half = int(len(mp_count_mat) / 2)
	pos_count_mat = mp_count_mat[:half]
	neg_count_mat = mp_count_mat[half:]

	K = len(pos_count_mat)
	M = len(pos_count_mat[0])

	# Compute the C matrix
	C = np.array([0]*M + [1]*K)

	# Build up the upper bound matrix
	tmp1 = np.concatenate([-np.eye(M), np.zeros((M,K))], axis=1)
	tmp2 = np.concatenate([np.zeros((K,M)), -np.eye(K)], axis=1)
	tmp = np.array([neg - pos for neg, pos in zip(neg_count_mat, pos_count_mat)])
	tmp3 = np.concatenate([tmp, -np.eye(K)], axis=1)
	A_ub = np.concatenate([tmp1, tmp2, tmp3])

	# Build up the upper bound vector
	b_ub = np.array([0]*(M+K)+[-c]*K)

	# Build up the equality matrix
	A_eq = np.array([[1]*M + [0]*K])

	# Build up the equality vector
	b_eq=np.array(1)

	# Optimize for the slack variables
	result = linprog(C, A_ub, b_ub, A_eq, b_eq, method='interior-point')

	return result



def get_lsr(counters_list, c, top_num=10):
	"""
	Solves for LSR weights and returns the top metapaths.

	.. note:: This is the function that will typically be called by the user.
	
	Parameters
	----------
		counters_list: list of Counter()
			A list of counters for positive and negative example pairs.
		c: float
			See :func:`semnet.compute_mp_weights()`.
		top_num: int
			The number of representative metapaths to select for the LSR.
	
	Returns
	-------
		top_mp_names: list of str
			The top metapath string representations.

		top_mp_wt: list of float
			The top-weighted metapaths of the latent semantic relationship.
	"""

	# Vectorize the results
	mp_count_mat, mp2ix = vectorize_results(counters_list)

	# Compute the top latent semantic relationships and remove slack variables
	results = compute_mp_weights(mp_count_mat, c)
	num_slack_var = int(len(counters_list) / 2)
	lsr = results.x[:-num_slack_var]

	# Dict to convert top metapaths indices back to metapaths
	ix2mp = dict({ix:mp for mp, ix in mp2ix.items()})

	# Find the top metapaths and their weights
	top_mp_ix = lsr.argsort()[::-1][:top_num]
	top_mp_wt = np.sort(lsr)[::-1][:top_num]
	top_mp_names = [ix2mp[ix] for ix in top_mp_ix]

	return top_mp_names, top_mp_wt



def plot_lsr(top_mp_names, top_mp_wt, filename=None):
	""" 
	Generates a bar plot of the most relevant metapaths. 

	.. note:: This is the function that will typically be called by the user.

	Parameters
	----------
		top_mp_names: list of str
			The top metapath string representations.

		top_mp_wt: list of float
			The top-weighted metapaths of the latent semantic relationship.

		filename: str
			The filename for saving the plot.
	"""
	plt.figure(figsize=(int(len(top_mp_names)*0.6), 5))
	y_pos = np.arange(len(top_mp_names))
	plt.bar(y_pos, top_mp_wt, align='center', alpha=0.5)
	plt.xticks(y_pos, top_mp_names, horizontalalignment='right', rotation=30)
	plt.ylabel('Weight')
	plt.title('Top Latent Semantic Relationship Metapaths')
	if filename:
		plt.savefig(filename, bbox_inches='tight')

	return plt.show()