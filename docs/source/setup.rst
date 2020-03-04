Setup and Installation
======================

The only external resource that SemNet expects is a locally hosted Neo4j database (``https://localhost:7474``) containing the Semantic Medline data from SemMedDB, in the format described in the :ref:`Data` section. Neo4j is freely available software, and the documentation for installing and importing data is available on their `website`_.

If you haven't installed Anaconda on your machine, we recommend doing this first. `Anaconda`_ is a free tool that makes installing and managing Python packages and environments much easier. The best way to install Anaconda for most users is through either the graphical or command line installer available on their website.

After installing Anaconda, we recommend setting up an environment for working with SemNet. You can do this by opening up the Anaconda prompt and entering ``conda create -name semnet-env python``. More information on managing environments can be found in the `conda environments documentation`_.

Next, you will need to download a copy of SemNet from GitHub. You can do this by cloning the repository to whatever location you prefer (we recommend ``Downloads``, unless you are planning on modifying the code). Then, you'll want to switch to your newly created ``conda`` environment and install ``semnet`` using ``pip``. Dependencies should install automatically. We summarize the Anaconda prompt input below:

.. code-block:: bash

    conda create -name semnet-env python
    cd Downloads
    git clone https://github.gatech.edu/pathology-dynamics/semnet.git
    conda activate semnet-env
    cd semnet
    pip install .

You're now ready to use SemNet! Open up the Neo4j database and a Python script and ``import semnet`` to get started.

.. todo:: Make sure to update this when SemNet finds its permanent home - perhaps on a public repository outside of GitHub Enterprise.

.. _`website`: https://neo4j.com/docs/
.. _`Anaconda`: https://www.anaconda.com/distribution/
.. _`conda environments documentation`: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
