'''
This module is intended to generate rankers for pairs of nodes that exemplify the relationships extracted using :mod:`semnet.lsr`. We implement a metapath-based ranking model that selects a feature subset and computes weights to rank a set of entities. It is based on work by Yu et al. 2012 [1]_.

.. [1] Yu, Xiao, et al. "User guided entity similarity search using meta-path selection in heterogeneous information networks." Proceedings of the 21st ACM international conference on Information and knowledge management. ACM, 2012.
'''

from collections import defaultdict
import math
import numpy as np
from scipy.optimize import minimize

from semnet.vector import vectorize_results


def get_ranking_correlation(pos_feature_dicts, neg_feature_dicts):
	'''
	This function computes the concordance/discordance ratio of each feature
	in the dataset. Kendall's tau coefficient was used instead of Yu's 
	ranking coefficient because it provided more intiuitive results and 
	showed a nicer distibution.


	The code assumes that query_pairs contains an array of tuples, the first 
	half of which contains positive pairs and the second half of which 
	contains negative pairs.

	Parameters
	----------
		pos_feature_dicts
		neg_feature_dicts
	Returns
	-------
	'''

	# Use defaultdicts to allow comparison
	pos = [defaultdict(lambda: 0, feature_dict) for feature_dict in pos_feature_dicts]
	neg = [defaultdict(lambda: 0, feature_dict) for feature_dict in neg_feature_dicts]

	# Compute whether each feature is concordant or discordant for each query
	query_concordance = []
	for pos_feats, neg_feats in zip(pos, neg):
		all_metapaths = set().union(*(pos_feats.keys(), neg_feats.keys()))
		concordance = defaultdict(lambda: 0)
		for metapath in all_metapaths:
			if pos_feats[metapath] > neg_feats[metapath]:
				concordance[metapath] = 1
			elif pos_feats[metapath] < neg_feats[metapath]:
				concordance[metapath] = -1
		query_concordance.append(concordance)

	# Calculate the ranking correlation of each feature across queries
	all_metapaths = set().union(*(d.keys() for d in pos_feature_dicts + neg_feature_dicts))
	rank_corr = dict()
	for metapath in all_metapaths:
		concordant = 0
		discordant = 0
		for q_conc in query_concordance:
			if q_conc[metapath] == 1:
				concordant += 1
			elif q_conc[metapath] == -1:
				discordant += 1
		# Computes Kendall's tau coefficient
		rank_corr[metapath] = (concordant - discordant) / len(query_concordance)
		# Computes Yu's ranking correlation
		# rank_corr[metapath] = (concordant + 1) / (discordant + 1)

	return rank_corr


def select_features(rank_corr_dict):
	'''
	Selects all features that predict with Kendall's tau > 0.2. This function must be used first 

	.. todo:: In the future, may want to select features for the ranking model using an entropy-based histogram thresholding method.
	'''

	return {mp: tau for mp, tau in rank_corr_dict.items() if tau > 0.2}

def learn_ranking_model(top_features, feature_dicts, seed=1):
	'''
	This function maximizes a sum of per-query objective functions and returns
	a set of weighted metapaths.

	We can then:
	This function ranks a set of query pairs based on their agreement with the
	learned ranking coefficients.

	.. note:: This is the main function in this module.

	'''
	np.random.seed(seed)

	# Split the counters back up into positive and negative examples
	pos_feature_dicts = feature_dicts[:len(feature_dicts)//2]
	neg_feature_dicts = feature_dicts[len(feature_dicts)//2:]

	# Remove anything not in top_features
	top_pos_feat_dicts = []
	for feat_dict in pos_feature_dicts:
		top_pos_feat_dicts.append({mp: count for mp, count in feat_dict.items() if mp in top_features})
	top_neg_feat_dicts = []
	for feat_dict in neg_feature_dicts:
		top_neg_feat_dicts.append({mp: count for mp, count in feat_dict.items() if mp in top_features})

	# Vectorize the counts
	count_mat, mp2ix = vectorize_results(top_pos_feat_dicts + top_neg_feat_dicts)

	pos_mat = count_mat[:len(count_mat)//2]
	neg_mat = count_mat[len(count_mat)//2:]

	def sigmoid(x):
		try:
			return 1 / (1 + math.exp(-x))
		except OverflowError:
			if x > 0: return 1
			if x < 0: return 0

	def p_i(theta, feats):
		return sigmoid(np.dot(theta, feats))

	def per_query_obj(theta, pos_ex, neg_ex):
		pos_contrib = np.log(p_i(theta, pos_ex))
		neg_contrib = np.log(1-p_i(theta, neg_ex))
		return pos_contrib + neg_contrib

	def obj_func(theta, pos_mat, neg_mat):
		return -sum([per_query_obj(theta, pos_vect, neg_vect) 
					for pos_vect, neg_vect in zip(pos_mat, neg_mat)])

	result = minimize(obj_func, 
					np.random.randn(1, len(pos_mat[0])), 
					(pos_mat, neg_mat), 
					method='BFGS', options={'disp': True})

	ranking_model = {mp:weight for mp, weight in zip(sorted(mp2ix.keys()), result.x)}

	return ranking_model
