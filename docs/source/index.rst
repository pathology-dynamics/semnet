SemNet: Semantic Networks in Python
===================================

What is SemNet?
^^^^^^^^^^^^^^^

SemNet is a series of modules for working with semantic networks. More
specifically, we implemented algorithms that would help us gather features and
rank nodes in terms of their connections to other nodes. The software was
designed to work with a `network`_ of biomedical concepts stored in a locally
hosted `Neo4j`_ database (`node types`_, `edge types`_)

.. note::
    To install, clone `the repository`_ to your machine, change to the
    ``semnet`` directory, and use ``pip install .``. You should then be able to
    ``import semnet`` and use it.

Features
^^^^^^^^

The following features have been implemented in SemNet:

* **Feature Extraction**

    * Metapath counts between node pairs
    * Degree-weighted path counts between node pairs [1]_
    * HeteSim similarity scores between node pairs [2]_

* **Computation**

    * Latent semantic relationship extraction [3]_
    * User-guided similarity search [4]_
    * Unsupervised ranker aggregation [5]_

* **Utilities**

    * Neo4j database interaction
    * Query parallelization
    * Vectorization
    * Negative example generation

.. toctree::
    :hidden:

    about_us
    about_data
    setup
    featextract
    ranking
    latent
    apidoc/modules
    updating_docs
    glossary

References
^^^^^^^^^^

.. [1] Himmelstein, Daniel S., and Sergio E. Baranzini. "Heterogeneous network edge prediction: a data integration approach to prioritize disease-associated genes." PLoS Computational Biology 11.7 (2015): e1004259.
.. [2] Shi, Chuan, et al. "HeteSim: A General Framework for Relevance Measure in Heterogeneous Networks." IEEE Trans. Knowl. Data Eng. 6.10 (2014): 2479-2492.
.. [3] Wang, Chenguang, et al. "RelSim: relation similarity search in schema-rich heterogeneous information networks." Proceedings of the 2016 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2016.
.. [4] Yu, Xiao, et al. "User guided entity similarity search using meta-path selection in heterogeneous information networks." Proceedings of the 21st ACM International Conference on Information and Knowledge Management. Acm, 2012.
.. [5] Klementiev, Alexandre, Dan Roth, and Kevin Small. "An unsupervised learning algorithm for rank aggregation." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2007.

.. _network: https://skr3.nlm.nih.gov/SemMedDB/index.html
.. _Neo4j: https://neo4j.com/
.. _node types: https://www.nlm.nih.gov/research/umls/META3_current_semantic_types.html
.. _edge types: https://www.nlm.nih.gov/research/umls/META3_current_relations.html
.. _the repository: https://github.gatech.edu/pathology-dynamics/semnet
