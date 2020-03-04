"""
This module implements a metapath-based similarity metric known as
:term:`HeteSim`. It is commonly used for computing similarity between nodes in
heterogeneous information networks. Please see Shi et al. 2014 [#]_ for more details.

.. [#] Shi, Chuan, et al. "HeteSim: A General Framework for Relevance Measure in Heterogeneous Networks." IEEE Trans. Knowl. Data Eng. 6.10 (2014): 2479-2492.
"""

import gzip
import pickle
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics.pairwise import cosine_similarity

#import hetio.neo4j
from hetio import neo4j
#import hetio.readwrite
from hetio import readwrite
import hetio.hetnet #added 7/8/19

from semnet.neo4j import execute_multithread_query

# Load the metagraph and fetch the metapath
# path = '/usr/local/anaconda3/lib/python3.6/site-packages/semnet/data/sem-net-mg_hetiofmt.json.gz' #change this back if below fails
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(_ROOT,'data/sem-net-mg_hetiofmt.json.gz')
metagraph = hetio.readwrite.read_metagraph(path)


def get_neighbors(sources, edge, graph):
	"""
	Generates a Cypher query and uses it to find the neighbors of a set of
	sources node along a given edge. 
	
	Parameters
	----------
	sources: list of str
		A list of source CUI's.

	edge: str
		A string representation of the edge predicate. See the :ref:`Data` 
		section for an example of the ``predicate`` edge field.

	graph: py2neo.Graph
		A Graph object connected to a locally hosted instance of the Neo4j 
		database.

	Returns
	-------
		neighbors_list: list of lists of str
			A list of the CUI's of neighbors of the source along the given edge.
	"""

	assert isinstance(sources, list)

	with gzip.open(os.path.join(_ROOT, 'data/cui2type.pkl.gz'), 'rb') as file:
		convert2type = pickle.load(file)

	neighbors_list = []

	for source in sources:
		s_type = convert2type[source]
		format_dict = {'source': source, 
					's_type': s_type,
					'edge': hetio.neo4j.as_type(edge)}
		q = """
			MATCH (m:{s_type} {{identifier: '{source}'}})-[:{edge}]-(n) 
			RETURN n.identifier ORDER BY n.identifier
			"""
		query = q.format(**format_dict).replace('\n', '').replace('\t', '')
		cursor = graph.run(query)
		query_results = cursor.data()
		cursor.close()

		neighbors_list.append([list(r.values())[0] for r in query_results])

	return neighbors_list


def neighbors_to_tp_df(source_list, target_list):
	"""
	Turns a list of results into a sorted transition probability dataframe.
	
	Parameters
	----------
		source_list: list of str
			A list of source nodes.

		target_list: list of list of str
			A list of the neighbors of each node in ``source_list``.

	Returns
	-------
		df: pd.DataFrame
			A dataframe of transition probabilities from source node to 
			each target node.
	"""

	# Compute appropriate indices for data matrix and DataFrame
	source_nodes = sorted(source_list)
	source2ix = {source:i for i, source in enumerate(source_nodes)}
	target_nodes = sorted(set(cui for result_set in target_list for cui in result_set))
	target2ix = {target:i for i, target in enumerate(target_nodes)}

	num_source = len(source_nodes)
	num_target = len(target_nodes)

	# Create the data matrix
	adj_mat = np.zeros((num_source, num_target))
	for source, targets in zip(source_list, target_list):
		for target in targets:
			adj_mat[source2ix[source]][target2ix[target]] = 1

	df = pd.DataFrame(data=adj_mat, index=source_nodes, columns=target_nodes)

	# Normalize each row of the dataframe by its sum
	df = df.div(df.sum(axis=1), axis=0)
	# Replace NaN's with 0
	df = df.fillna(0)

	return df


def decompose_along_edges(u_xy, revers=False):
	""" 
	Moves each nonzero entry in u_xy to its own column, labeled with 'SOURCE_TARGET' 
	(in direction of metapath).

	Parameters
	----------
		u_xy: pd.DataFrame
			An input dataframe.
		
		revers: boolean
			Indicates the direction of metapath traversal, which allows indices 
			to match up when multiplying dataframes.

	Returns
	-------
		u_xe: pd.DataFrame
			The input dataframe with each nonzero entry in its own column.
	"""

	u_xe = pd.DataFrame()

	# Iterate through the columns using the number of nonzeros in each column
	for column, nnz in u_xy.astype(bool).sum(axis=0).iteritems():
		# Get the indices of the nonzero elements
		##nz_int_ix = u_xy[column].nonzero()[0]
		nz_int_ix = u_xy[column].to_numpy().nonzero()[0]
		
		# If there are more than one nonzero in the column, create multiple 
		# columns in the edge df
		if nnz > 1:
			for int_ix in nz_int_ix:
				data = np.zeros(len(u_xy[column]))
				data[int_ix] = u_xy[column].iloc[int_ix]
				new_col = pd.Series(data, u_xy.index)
				if revers:
					u_xe['{}_{}'.format(column, u_xy.index[int_ix])] = new_col
				else:
					u_xe['{}_{}'.format(u_xy.index[int_ix], column)] = new_col
							
		# If there is only one nonzero in the column, copy it directly to the edge df
		else:
			if revers:
				u_xe['{}_{}'.format(column, u_xy.index[nz_int_ix[0]])] = u_xy[column]
			else:
				u_xe['{}_{}'.format(u_xy.index[nz_int_ix[0]], column)] = u_xy[column]

	return u_xe



def reconcile_inner_dims(u1, u2):
	""" 
	Add the missing columns to each DataFrame and re-sort them so their 
	new columns are the sorted union of previous two sets of columns. 

	Parameters
	----------
		u1: pd.DataFrame
			An input dataframe.
		
		u2: pd.DataFrame
			A second input dataframe.

	Returns
	-------
		u1: pd.DataFrame
			The input dataframe with reconciled columns.

		u2: pd.DataFrame
			The input dataframe with reconciled columns.
	
	"""

	missing_cols = u2.columns[[not i for i in u2.columns.isin(u1.columns)]]
	for col in missing_cols:
		u1[col] = pd.Series(data=np.zeros(len(u1)), index=u1.index)
	u1 = u1.sort_index(axis=1)
		
	missing_cols = u1.columns[[not i for i in u1.columns.isin(u2.columns)]]
	for col in missing_cols:
		u2[col] = pd.Series(data=np.zeros(len(u2)), index=u2.index)
	u2 = u2.sort_index(axis=1)

	return u1, u2
	

def compute_mp_hetesim(source_list, target_list, metapath, graph):
	"""
	A helper function that computes normalized HeteSim scores between a 
	list of sources and a list of targets along a given metapath. 
	
	Parameters
	----------
		source_list: list of str
			A list of source CUI's.

		target_list: list of str
			A list of target CUI's.

		metapath: str
			A string representation of the metapath of interest.

		graph: py2neo.Graph
		A Graph object connected to a locally hosted instance of the Neo4j 
		database.

	Returns
	-------
		hetesim_mat: np.array
			An array of metapath-normalized HeteSim scores between sets 
			of sources and targets for a given metapath.
	"""

	mp_len = len(metapath)
	num_trans_probs = mp_len if not mp_len % 2 else mp_len + 1
	# A list to be filled with transition probability matrices
	tp = [None]*num_trans_probs

	sources = source_list
	targets = target_list
	for layer in range(num_trans_probs // 2):
		# Compute forward transitions
		edge = metapath[layer]
		neighbors_list = get_neighbors(sources, edge, graph)
		tp[layer] = neighbors_to_tp_df(sources, neighbors_list)

		# Compute reverse transitions
		edge = metapath[-1-layer].inverse
		neighbors_list = get_neighbors(targets, edge, graph)
		tp[-1-layer] = neighbors_to_tp_df(targets, neighbors_list)

		# Set up next sources and targets
		sources = list(tp[layer].columns)
		targets = list(tp[-1-layer].columns)

	# Compute indices where source and target computations will mesh together
	s_join_ix = (mp_len-1) // 2
	t_join_ix = (mp_len-1) // 2 + 1

	# If this is an odd metapath, decompose edges
	if mp_len % 2:
		tp[s_join_ix] = decompose_along_edges(tp[s_join_ix])
		tp[t_join_ix] = decompose_along_edges(tp[t_join_ix], revers=True)

	# Reconcile the inner dimensions to mesh source and target computations
	tp[s_join_ix], tp[t_join_ix] = reconcile_inner_dims(tp[s_join_ix], tp[t_join_ix])

	# Multiply out the transition probabilities and reachable probability matrices
	pm_left = np.eye(len(tp[0]))
	pm_right = np.eye(len(tp[-1]))
	for i in range(len(tp) // 2):
		pm_left = np.matmul(pm_left, tp[i])
		pm_right = np.matmul(pm_right, tp[-1-i])

	hetesim_mat = cosine_similarity(pm_left, pm_right)
	
	return hetesim_mat

def compute_all_hetesim(source_list, target_list, metapath_list, graph, workers=40):
	"""
	Compute all HeteSim metrics along a set of metapaths for a set of source and
	target entities.

	Distributes lists of sources, targets, and metapaths to Cypher queries to
	count metapaths for all examples in parallel. Returns a structured
	representation of the results. While the feature extractors in
	:mod:`semnet.feature_extraction` count node pairs in parallel, this
	function computes scores along a given metapath in parallel.

	The details of the complete algorithm are best found in Shi et al. 2014 (above). 
	Essentially, this function and its helpers work by:

	1. Splitting the metapath list amongst parallel queries
	2. For each metapath, computing forward and reverse transition probability matrices.
	3. Reconciling indices of the innermost transition probability matrices.
	4. Multiplying everything out to end up with a matrix of HeteSim scores.

	.. note:: This is the function that will typically be called by the user.

	Parameters
	----------
		source_list: list of str
			A list of source CUI's.

		target_list: list of str
			A list of target CUI's.

		metapath_list: list of str
			A list of metapaths between the source and target nodes.

		graph: py2neo.Graph
			A Graph object connected to a locally hosted instance of the Neo4j 
			database.

		workers: int
			The number of workers desired for parallel computation.

	Returns
	-------
		data: xarray.DataArray
			A 3-D data structure containing the metapath strings and HeteSim scores 
			for each source-target pair.
	"""

	params = []
	source_list = sorted(source_list)
	target_list = sorted(target_list)
	metapath_list = sorted(metapath_list)
	for mp_string in metapath_list:
		metapath = metagraph.get_metapath(mp_string)
		params.append({'source_list': source_list,
						'target_list': target_list,
						'metapath': metapath,
						'graph': graph})

	result = execute_multithread_query(compute_mp_hetesim, params=params, workers=workers)

	# ADDEDED to debug 7/31/19	
	print("result: ")
	print("rows: " , len(result))
	print("cols: " , len(result[0]))
	print("dims: " , len(result[0][0]))
	#print(result[0:2])
	#dataa = np.array(result)
	#print("dataa shape: ")
	#print(dataa.shape)
	####

	#return hetesim_results_to_xr(result, source_list, target_list, metapath_list)
	return hetesim_results_to_xr(source_list, target_list, result, metapath_list)
	
def hetesim_results_to_xr(source_list, target_list, results, metapath_list):
	""" 
    Converts the results array of dicts into the structured ``xr.DataArray``
    format.

    Uses the sources, targets, metapaths, and feature type to construct a 
    labeled, multi-dimensional data structure that supports named axes.

    Parameters
    ----------
        source_list: list of str
          A list of strings containing the CUI's of source nodes.

        target_list: list of str
          A list of strings containing the CUI's of target nodes.

        results: list of dict
          A list of dicts containing the results of the Cypher query.

        metapath_list: list of str
          The list of metapaths corresponding to the features calculated.

    Returns
    -------
      data: xarray.DataArray
        A multi-dimensional, labeled array that contains the feature data.
    """

	### ADDED to debug 7/31/2019
	data = np.array(results)
	print("data shape: ")
	print(data.shape)
	#print(data)
	### 

	data = np.swapaxes(np.swapaxes(data, 0, 2), 0, 1)

	s_type = metapath_list[0][:4]
	t_type = metapath_list[0][-4:]

	return xr.DataArray(data, 
						coords=[source_list, target_list, metapath_list],
						dims=['source', 'target', 'metapath'],
						attrs={'feature': 'hetesim', 's_type':s_type, 
						't_type':t_type})
