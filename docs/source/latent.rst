Latent Relationships
====================

Overview
--------

Characterizing the complex and noisy relationships between nodes can be very difficult. However, it can become greatly simplified if we provide training examples. :mod:`semnet.lsr` can characterize a relationship type using a weighted metapath profile. This gives an intuitive representation of the types of paths that are important for a given relationship. For example, we learned the following latent semantic relationship profile that was common to many drug-disease pairs:

.. image:: _static/dwpc_lsr_dd_pairs.png

Usage
-----

To compute the latent semantic relationship between pairs of nodes, we first need to curate a set of :math:`N` training examples. In past work, these have been drug-disease pairs, but any pairs that share similar relationships will work fine. These must be mapped to UMLS CUI's. These strings should be stored in a :math:`2\times N` numpy array, where the rows are the pairs and the columns are sets of similar entities. Note that all nodes should have the same type on a per-column basis.

We then generate negative exaples by shuffling the example pairs using :mod:`semnet.sample`. These negative examples will help us to eliminate metapaths that are only present because of noise.

Both the positive and negative examples can then be passed to the :mod:`semnet.feature_extraction` module to return an ``xarray.DataArray`` of feature values. Note that this data must be converted to a list of ``dicts`` in order to be used with this module.

A sample implementation is provided here:

.. code-block:: python

    from semnet.lsr import generate_negative_examples, get_lsr, plot_lsr
    from feature_extraction import CountExtractor

    # Generate negative examples and convert pairs into source/target arrays
    # Note: pos_ex contains positive example CUI pairs
    neg_ex = generate_negative_examples(pos_ex, hold='s', seed=1)
    all_ex = pos_ex + neg_ex
    sources = all_ex[:,0]
    targets = all_ex[:,1]

    # Extract features
    cex = CountExtractor()
    data = cex.get_all_metapath_counts(sources, targets, 2)

    # Reformatting the DataArray
    features_dict_list = list()
    for s, t in zip(data.source.values, data.target.values):
        nonzero_mps = (data.loc[s, t, :, 'count'] != 0).values
        num_nz = sum(nonzero_mps)
        if num_nz != 0:
            metapaths = data.metapath.values[nonzero_mps]
            scores = data.loc[s, t, nonzero_mps, 'count'].values
            feat_dict = {mp: score for mp, score in zip(metapaths , scores)}
            features_dict_list.append(feat_dict)

    # Compute the LSR
    top_mp_names, top_mp_wt = get_lsr(feature_dict_list, 0.02, 20)
