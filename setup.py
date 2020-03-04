from setuptools import setup

setup(name='semnet',
	version='0.1.0',
	description='A package for working with Semantic Medline data',
	url='https://github.gatech.edu/asedler3/semnet',
	author='Andrew Sedler',
	author_email='asedler3@gatech.edu',
	packages=['semnet'],
	install_requires=['hetio==0.2.8', 'xarray==0.10.7', 'numpy==1.15.0','py2neo==3.1.2', 'pandas==0.23.0', 
	'sklearn==0.19.1', 'scipy==1.1.0', 'matplotlib==2.2.2', 'tqdm==4.23.4', 'seaborn==0.8.1'],
	include_package_data=True,
	package_data={
		'semnet': ['data/*']
	}
)
