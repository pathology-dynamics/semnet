"""
Converts data from the Neo4j database into a format compatible with manipulation
in hetio. Hetio provides simple operations on the metagraph.
"""

import re
import hetio.readwrite
#import hetio.hetnet #added import hetio.hetnet in an attempt to get metagraph.get_metaedge to work
from hetio.hetnet import MetaGraph

""" Load the metagraph. """
import os
import sys
path = os.path.abspath('data/sem-net-mg_hetiofmt.json.gz')

curr_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(curr_dir, 'data/sem-net-mg_hetiofmt.json.gz')

metagraph = hetio.readwrite.read_metagraph(path)


def get_metapath_abbrev(query_result):
	""" 
	Creates a string abbreviation to label query results.

	Converts a dictionary of an individual Neo4j query results into a simplified
	and unique string representation.

	Parameters
	----------
	query_result: dict
		The result of a query for metapaths between a source and target node.

	Returns
	-------
	metapath: str
		A string representation of the metapath, which we use to refer to it 
		throughout semnet.
	"""

	node_types = query_result['nodes']
	edge_types = query_result['edges']
	mp = neo4j_rels_as_metapath(edge_types, node_types)

	return str(mp)

def neo4j_rels_as_metapath(edge_types, node_types):
	""" 
	Converts lists of edge and node types to a ``hetio.hetnet.MetaPath`` object.

	Uses regular expressions to capture the appreviation at the end of the edge
	type and insert directionality symbol. Using this method, a list of strings
	formatted as ``TREATS_ORCHtreatsDSYN`` in Neo4j become MetaPath objects.
	Note, directionality may become compromised when two sequential nodes have
	the same type.

	Parameters
	----------
	edge_types: array_like 
		A sequence of edge type strings. 
	
	node_types: array_like 
		A sequence of node type strings.

	Returns
	-------
	metapath: hetio.hetnet.MetaPath 
		A MetaPath object that represents the sequence of nodes and edges.
	"""

	# Capture the abbreviation at the end of the edge type
	abbrevs = [re.split('_(?=[A-Z])', e)[-1] for e in edge_types]
	# Insert the directionality symbol
	all_matches = [re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|$)', a) for a in abbrevs]
	std_metaedge_abbrevs = ('>'.join([m.group(0) for m in matches]) for matches in all_matches)
	# Convert to MetaEdges
	metaedges = [metagraph.get_metaedge(e) for e in std_metaedge_abbrevs]

	# Check if the relationships need to be inverted - note that 
	# relationships are passed through by neo4j in the correct order,
	# but sometimes subject may need to be swapped with predicate 
	# for the metapath.
	for i, metaedge in enumerate(metaedges):
		if i==0:
			# Match node type to database formatted string
			metaedge_source = str(metaedge.source).title().replace(' ', '')
			if metaedge_source != node_types[0]: 
				metaedges[i] = metaedge.inverse
		else:
			if metaedge.source != metaedges[i-1].target:
				metaedges[i] = metaedge.inverse

	return metagraph.get_metapath(tuple(metaedges))
