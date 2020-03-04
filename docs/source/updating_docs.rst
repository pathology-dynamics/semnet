Updating Docs
=============
Use the following steps to make changes to documentation and post them on the page.

.. code-block:: none

    cd semnet
    git add .
    git commit -m "[message]"
    git push
    cd semnet/docs
    make html
    cd semnet-docs/html
    git add .
    git commit -m "[message]"
    git push origin gh-pages

More information on the setup can be found here:

https://daler.github.io/sphinxdoc-test/includeme.html
