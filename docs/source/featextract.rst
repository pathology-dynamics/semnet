.. _FeatExtract:

Feature Extraction
==================

Overview
--------

We can represent each pair of nodes in the biomedical concept graph in terms of the many types of paths between them. For example, there might be hundreds of genes and proteins between the amyloid-:math:`\beta` (:term:`source`) and Alzheimer's disease (:term:`target`) nodes. Additionally, the specific relationships that connect these genes and proteins with amyloid-:math:`\beta` and Alzheimer's disease are likely to vary. Each gene or protein could even have multiple relationships with both the source and target nodes. We use the term :term:`paths` to refer to the *specific sequences of nodes and relationships* that lead from the source node to the target node. More generally, we can use the term :term:`metapaths` to refer to the *sequences of node and relationship types* between source and target nodes.

.. note:: Since each node has both a unique identity and one of a set of 133 types, each metapath can represent multiple paths.

When characterizing the relationship between a pair of nodes in the graph, we first find the set of metapaths between the two nodes. Then, for each of these metapaths, we compute a feature value. The simplest example of a metapath-based feature is simply the total number of paths associated with that metapath. We have created classes and functions for extracting three types of metapath-based features from the graph.

.. image:: _static/feature_extraction.png

In addition to counts, we can compute two additional metrics - :term:`degree-weighted path counts` (DWPC) and :term:`HeteSim`. Both of these features are designed to counteract the bias that highly connected nodes cause in the network.

Usage
-----

Count Features
^^^^^^^^^^^^^^

The :class:`semnet.feature_extraction.CountExtractor` class is designed to find and count all metapaths between pairs of nodes. All you need to do is specify Python ``lists`` of the UMLS CUIs of source and target nodes. Notice that we repeat ``sources`` and ``targets`` in ``s`` and ``t`` to get all possible combinations.

.. code-block:: python

    import numpy as np
    from semnet.feature_extraction import CountExtractor

    cex = CountExtractor()
    s = sorted(sources) * len(targets)
    t = np.repeat(sorted(targets), len(sources))
    count_data = cex.get_all_metapath_counts(s, t, 2)

Extracting count-based features is usually fairly quick - typically only 3-5 seconds per pair. In this case, ``count_data`` contains an ``xarray.DataArray`` object. This is analogous to a multidimensional ``pandas.Series``. More details on the data format can be found in the :ref:`DataArray` section. Typically, we will also want to save these results so they don't have to be computed again later. More info can be found in the :ref:`DataPersistence` section.

.. note:: When performing feature extraction, you will always start by collecting count-based features. This is because the process of collecting unique metapaths is integrated into this part of SemNet, and counting the paths doesn't add a substantial amount of time to the process.

DWPC Features
^^^^^^^^^^^^^

Another class, :class:`semnet.feature_extraction.DwpcExtractor`, uses a more sophisticated feature computation algorithm that damps the effect of highly weighted nodes [#]_. It takes the same parameters as the ``CountExtractor`` class above, but adds a damping hyperparameter and an optional count ``DataArray`` over the same sources and targets. 

.. code-block:: python

    import numpy as np
    from semnet.feature_extraction import DwpcExtractor
    
    dex = DwpcExtractor()
    s = sorted(sources) * len(targets)
    t = np.repeat(sorted(targets), len(sources))
    dwpc_data = dex.get_all_dwpc(s, t, 2, 0.4, count_data)

.. warning:: The metapaths in the counts ``DataArray`` are required by the feature extractor. If it is not passed, the function will call it by default and add more execution time.

We haven't experimented much with the hyperparameter, but 0.4 was used by the creator of the feature, and we have found it to work well. This might be a good area for exploration.

HeteSim Features
^^^^^^^^^^^^^^^^

Finally, we use the function :func:`semnet.hetesim.compute_all_hetesim` to calculate the HeteSim features. HeteSim is a state-of-the-art measure of similarity in heterogeneous information networks that is based on the probability of random walkers starting at source and target nodes, travelling along the metapath and meeting in the middle [#]_. It is implemented slightly differently than the previous two feature extractors for algorithmic reasons. It takes as parameters lists of (unique) source and target nodes, a list of metapaths, and a ``py2neo.Graph`` object.

.. code-block:: python

    from semnet.hetesim import compute_all_hetesim

    hs_data = compute_all_hetesim(count_data.source.values, 
                                  count_data.target.values, 
                                  count_data.metapath.values, graph)

.. warning:: Be sure to consider the differences in the parameters before calling this function.

.. _DataArray:

DataArray Formatting
^^^^^^^^^^^^^^^^^^^^

To keep our data organized, we use handy multidimensional data structures called ``DataArrays``. They are similar to multidimensional ``numpy.arrays``, with the addition of labeled axes and coordinates. In SemNet, they have four dimensions: ``source``, ``target``, ``metapath``, and ``metric``. The ``source`` and ``target`` axes are labeled with their respective CUIs, the ``metapath`` axis is labeled with a string representation of the metapath, and the ``metric`` axis is labeled with the feature calculated, so that count, DWPC, and HeteSim features can be combined into one ``DataArray``. You can find plenty of documentation for how to manipulate these data structures on the `xarray website`_.

.. _DataPersistence:

Storing Data
^^^^^^^^^^^^

Since these feature extraction algorithms can take several hours to run, we want to avoid duplicate runs. Fortunately, Python objects like ``xarray.DataArrays`` can be serialized, converted to a binary format, and saved to the hard drive. The ``pickle`` package makes it easy to perform these functions:

.. code-block:: python
    
    import pickle

    # Saving the data object
    with open('data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)

    # Loading the data object
    with open('data.pkl', 'rb') as outfile:
        data = pickle.load(outfile)

.. _xarray website: http://xarray.pydata.org/en/stable/

.. [#] Himmelstein, Daniel S., and Sergio E. Baranzini. "Heterogeneous network edge prediction: a data integration approach to prioritize disease-associated genes." PLoS computational biology 11.7 (2015): e1004259.
.. [#] Shi, Chuan, et al. "HeteSim: A General Framework for Relevance Measure in Heterogeneous Networks." IEEE Trans. Knowl. Data Eng. 6.10 (2014): 2479-2492.
