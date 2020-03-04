"""
Constructs queries compatible with Neo4j and submits multithreaded jobs using ``concurrent.futures``. These functions will not typically be called directly by the user, but are used by the :mod:`semnet.feature_extraction` classes.
"""

import pickle
import threading
import concurrent.futures
import gzip
import os
from tqdm import tqdm_notebook
# Avoid set size change warning
tqdm_notebook.monitor_interval=0
import copy

#Added on 7/3/19 to fix file path
from semnet.conversion import get_metapath_abbrev
_ROOT = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(_ROOT, 'data/cui2type.pkl.gz')

def build_metapath_query(source, target, d):
	""" 
	Generates a Cypher query string for all metapaths with :math:`length\\leq d`.
	
	Parameters
	----------
		source: str
			The CUI of the source node.

		target: str
			The CUI of the target node.
		
		d: int
			The length of the metapaths of interest.

	Returns
	-------
		query: str
			A Cypher query string that tells Neo4j to return all metapaths between ``source`` and ``target`` nodes.
	"""

	q = """
		MATCH path = (m:`{s_type}` {{identifier: '{source}'}})-
		[*..{d}]-(n:`{t_type}` {{identifier: '{target}'}}) 
		RETURN extract(a in nodes(path) | a.kind) as nodes, 
		extract(b in relationships(path) | b.predicate ) as edges 
		"""
#        import os
#        _ROOT = os.path.abspath(os.path.dirname(__file__))
#        path = os.path.join(_ROOT, 'data/cui2type.pkl.gz')
	with gzip.open(path, 'rb') as file: #changed from '../semnet/data/cui2type.pkl.gz' to path
		convert2type = pickle.load(file)

	s_type = convert2type[source]
	t_type = convert2type[target]

	format_dict = {'source': source, 
					'target': target,
					's_type': s_type,
					't_type': t_type,
					'd': d}

	return q.format(**format_dict).replace('\n', '').replace('\t', '')


def execute_multithread_query(func, params, workers=40):
	"""
	Executes a large number of Cypher queries simultaneously. Displays a `tqdm_notebook` progress bar in Jupyter.

	Parameters
	----------
		func: function
			The function to be performed on many different CPU cores.

		params: list of dict
			A list of different parameter sets to be applied to the function.

		workers: int
			The number of workers desired for parallel computation.

	Returns
	-------
		results: list
			A list of the func's output for the different parameter sets.
	
	"""

	# Transform params for mapping
	transformed_params = [list(col) for col in zip(*[param.values() for param in params])]

	# Submit jobs with ThreadPoolExecutor
	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
		results = list(tqdm_notebook(executor.map(func, *transformed_params), total=len(params)))
	return results
