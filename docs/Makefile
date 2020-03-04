# Minimal makefile for Sphinx documentation
# Note that only make.bat runs in Windows. "make latexpdf" doesn't yet work in Windows.

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = ../../semnet-docs
PDFBUILDDIR   = /tmp
PDF           = ../manual.pdf
# sphinx-apidoc
SPHINXAPIDOC  = sphinx-apidoc
APISOURCEDIR  = source/apidoc
SOURCECODE    = ../semnet

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(PDFBUILDDIR)/latex
	#                                          ^^^
	@echo "Running LaTeX files through pdflatex..."
	make -C $(PDFBUILDDIR)/latex all-pdf
	#         ^^^
	cp $(PDFBUILDDIR)/latex/*.pdf $(PDF)
	#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	@echo "pdflatex finished; see $(PDF)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXAPIDOC) -M -H "SemNet Docs" -f -e -o "$(APISOURCEDIR)" $(SOURCECODE)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	