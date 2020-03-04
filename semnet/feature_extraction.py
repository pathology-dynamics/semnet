"""
This module implements two feature extractors that wrap the ``py2neo.Graph``
class. When objects are constructed, they initiate a connection to the locally
hosted Neo4j instance. Given pairs of nodes, feature extractors compute a
representation of the metapaths between them.
"""

import xarray as xr
import numpy as np
from collections import Counter

#import hetio.readwrite
from hetio import readwrite
import hetio.neo4j

from py2neo import Graph

from semnet.neo4j import build_metapath_query, execute_multithread_query
from semnet.conversion import get_metapath_abbrev

class BaseFeatureExtractor(object):
  """ 
  Defines the basic functions of a feature extractor. You'll have to customize
  the password in the ``__init__`` function to your own database.
  """

  def __init__(self):
    # Credentials to neo4j graph
    self.graph = Graph(password='')
	
    
  def results_to_dataarray(self, sources, targets, results, metric):
    """ 
    Converts the results array of dicts into the structured ``xr.DataArray``
    format.

    Uses the sources, targets, metapaths, and feature type to construct a 
    labeled, multi-dimensional data structure that supports named axes.

    Parameters
    ----------
        sources: list of str
          A list of strings containing the CUI's of source nodes.

        targets: list of str
          A list of strings containing the CUI's of target nodes.

        results: list of dict
          A list of dicts containing the results of the Cypher query.

        metric: 'counts' or 'dwpc'
          The name of the metric that was computed by the query.
    Returns
    -------
      data: xarray.DataArray
        A multi-dimensional, labeled array that contains the feature data.
    """
    
    s2ix = {s:ix for ix, s in enumerate(sorted(set(sources)))}
    t2ix = {t:ix for ix, t in enumerate(sorted(set(targets)))}
    mp2ix = {mp:ix for ix, mp in enumerate(sorted({metapath for r in results for metapath in r.keys()}))}
    
    data = np.zeros((len(s2ix), len(t2ix), len(mp2ix), 1))
    
    for s, t, mps in zip(sources, targets, results):
      for mp, value in mps.items():
        data[s2ix[s], t2ix[t], mp2ix[mp], 0] = value
    
    s_type = list(results[0].keys())[0][:4]
    t_type = list(results[0].keys())[0][-4:]
    
    data = xr.DataArray(data,
              coords=[sorted(s2ix.keys()), sorted(t2ix.keys()), sorted(mp2ix.keys()), [metric]],
              dims=['source', 'target', 'metapath', 'metric'],
              attrs={'s_type':s_type, 't_type':t_type})

    return data
    

class CountExtractor(BaseFeatureExtractor):
  """ Extracts metapath counts between pairs of nodes. """

  def get_metapath_counts(self, source, target, d):
    """ 
    Gets metapath counts from a source node to a target node.

    Generates and sends a Cypher query to the Neo4j instance to return all
    metapaths between a given pair. The set of metapaths is converted into a
    string abbreviation and these are counted.

    Parameters
    ----------
      source: str
        The CUI of the source node.

      target: str
        The CUI of the target node.
      
      d: int
        The maximum length of the metapaths to be fetched.

    Returns
    -------
      ctr: Counter
        A counter holding string representations of the metapaths and their counts.
    """

    query = build_metapath_query(source, target, d)
    cursor = self.graph.run(query)
    query_results = cursor.data()
    cursor.close()

    return Counter([get_metapath_abbrev(r) for r in query_results])


  def get_all_metapath_counts(self, sources, targets, d, workers=40):
    """ 
    Computes metapath counts of length less than ``d`` across a list of sources
    and a list of targets.

    Distributes a list of sources and targets to Cypher queries to count
    metapaths for all examples in parallel. Returns a structured representation
    of the results.

    .. note:: This is the function that will typically be called by the user.

    Parameters
    ----------
      sources: list of str
        A list of source CUI's.
      
      targets: list of str
        A list of target CUI's.

      d: int
        The maximum lenth of the metapaths to be fetched.
      
      workers: int
        The number of workers desired for parallel computation.

    Returns
    -------
      data: xarray.DataArray
        A 3-D data structure containing the metapath strings and counts for each source-target pair.
    """
    
    # Retrieve the results from Neo4j
    params = []
    for s, t in zip(sources, targets):
      params.append({'source': s, 'target': t, 'd': d})
      
    result = execute_multithread_query(self.get_metapath_counts, params=params, workers=workers)
    
    # Remembering which metapaths are nonzero helps with computational efficiency for dwpc
    self.metapath_counts = result
    
    return self.results_to_dataarray(sources, targets, result, 'count')
  
  
  
class DwpcExtractor(BaseFeatureExtractor):
  """
  Extracts :term:`degree-weighted path counts` (DWPC) between pairs of nodes.
  DWPC are an alternative to simple metapath counts that downweight paths with
  highly connected nodes.
  """
  
  def __init__(self):
    """ Load the metagraph and connect to Neo4j """
    
#    path = '../semnet/data/sem-net-mg_hetiofmt.json.gz'
    import os
    _ROOT = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(_ROOT,'data/sem-net-mg_hetiofmt.json.gz')
    self.metagraph = hetio.readwrite.read_metagraph(path)
    super(DwpcExtractor, self).__init__()
  
  def compute_dwpc(self, source, target, metapath, damping):
    """ 
    Performs a DWPC calculation for a source/target pair along a
    single metapath.

    Parameters
    ----------
      source: str
        The CUI of a source node.

      target: str
        The CUI of a target node.

      metapath: str
        The string representation of a metapath from 
        :func:`semnet.conversion.neo4j_rels_as_metapath()`.

      damping: float
        The damping coefficient of the calculation (ranges from 0 to 1, but 
        0.4 usually works well). A higher coefficient more strongly downweights 
        highly connected nodes.

    Returns
    -------
      query_results: list of dicts
        A single-element list containing the dictionary of query results under 
        the ``DWPC`` key.
    """

    metapath = self.metagraph.get_metapath(metapath)
    query = hetio.neo4j.construct_dwpc_query(metapath, 'identifier')

    params = {
      'source': source,
      'target': target,
      'w': damping
    }

    cursor = self.graph.run(query, params)
    query_results = cursor.data()
    cursor.close()

    return query_results


  def  compute_example_dwpc(self, source, target, metapaths, damping):
    """
    Performs DWPC calculations on all metapaths between a given pair of nodes. 

    Parameters
    ----------
    source: str
      The CUI of a source node.

    target: str
      The CUI of a target node.

    metapaths: list of str
      A list of string representations of unique metapaths, previously collected 
      from the metapath counting step.

    damping: float
      The damping coefficient of the calculation (ranges from 0 to 1, but 
      0.4 usually works well). A higher coefficient more strongly downweights 
      highly connected nodes.

    Returns
    -------
    dwpcs: dict
      A dictionary containing the metapaths and DWPC scores for a given source-target pair.
    """

    dwpcs = dict()
    for mp in metapaths:
      result = self.compute_dwpc(source, target, mp, damping)
      dwpcs[mp] = result[0]['DWPC']

    return dwpcs



  def get_all_dwpc(self, sources, targets, d, damping, metapath_counts=None, workers=40):
    """
    Performs all DWPC calculations for all example pairs in parallel.

    Distributes a list of sources and targets to Cypher queries to count
    metapaths for all examples in parallel. Returns a structured representation
    of the results.

    .. note:: This is the function that will typically be called by the user.

    Parameters
    ----------
    sources: list of str
      A list of source CUI's.
    
    targets: list of str
      A list of target CUI's.

    d: int
        The maximum lenth of the metapaths to be fetched.

    damping: float
      The damping coefficient of the calculation (ranges from 0 to 1, but 
      0.4 usually works well). A higher coefficient more strongly downweights 
      highly connected nodes.

    metapath_counts: xarray.DataArray
      The data array returned from a previous call to :func:`semnet.feature_extraction.CountExtractor.get_all_metapath_counts`.

    workers: int
      The number of workers desired for parallel computation.

    Returns
    -------
    data: xarray.DataArray
      A 3-D data structure containing the metapath strings and DWPCs for each source-target pair.
    """
    
    if not metapath_counts.any():
      metapath_counts = CountExtractor().get_all_metapath_counts(sources, targets, d, workers)
    assert isinstance(metapath_counts, xr.DataArray)

    params = []
    for s, t in zip(sources, targets):
      nz_metapath_ix = metapath_counts.loc[s, t, :, 'count'].values.nonzero()
      nz_metapaths = metapath_counts.metapath.values[nz_metapath_ix]
      params.append({'source': s, 'target': t, 'metapaths': nz_metapaths, 'damping': damping})

    result = execute_multithread_query(self.compute_example_dwpc, params=params, workers=workers)

    return self.results_to_dataarray(sources, targets, result, 'dwpc')
