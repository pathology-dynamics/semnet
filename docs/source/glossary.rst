Glossary
========

.. glossary::

    target
        A node that **you define**. It is related to the key concepts that you are interested in and is typically highly connected in the graph. A typical analysis can have anywhere from 1-5 target nodes.

    source
        A node that is **defined by some criteria that you choose**. Often, you will define source nodes as the nodes of a given type that are neighbors of your target node. It is usually reasonable to have on the order of 100 source nodes for a given analysis.

    paths
        Sequences of nodes and relationships between source and target nodes.

    metapaths
        Sequences of node and relationship types between source and target nodes.

    degree-weighted path counts
        A metapath-based feature that downweights highly connected nodes. It is derived from Himmelstein et. al. 2015 [#]_.

    HeteSim
        A commonly used metapath-based similarity measure from Shi et. al. 2014 [#]_. The score represents the probability that two walkers traveling along the metapath from source and target nodes will meet in the middle.
    
    Latent Semantic Relationships
        Learned groups of metapaths that tend to commonly occur between nodes that have a specific relationship. These are defined more thoroughly in Wang et. al. 2016 [#]_.

.. [#] Himmelstein, Daniel S., and Sergio E. Baranzini. "Heterogeneous network edge prediction: a data integration approach to prioritize disease-associated genes." PLoS computational biology 11.7 (2015): e1004259.
.. [#] Shi, Chuan, et al. "HeteSim: A General Framework for Relevance Measure in Heterogeneous Networks." IEEE Trans. Knowl. Data Eng. 6.10 (2014): 2479-2492.
.. [#] Wang, Chenguang, et al. “Relsim: relation similarity search in schema-rich heterogeneous information networks.” Proceedings of the 2016 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2016.