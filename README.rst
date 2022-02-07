py_neuromodulation
==================

.. image:: https://app.travis-ci.com/neuromodulation/py_neuromodulation.svg?branch=main
    :target: https://app.travis-ci.com/neuromodulation/py_neuromodulation

Click this button to run the "example_BIDS.ipynb":

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/neuromodulation/py_neuromodulation/HEAD

The py_neuromodulation toolbox allows for real time capable processing of multimodal electrophysiological data. The primary use is movement prediction for `adaptive deep brain stimulation <https://pubmed.ncbi.nlm.nih.gov/30607748/>`_.

Find the documentation here https://neuromodulation.github.io/py_neuromodulation/ for example usage and parametrization.

Setup
=====

For running this toolbox first create a new virtual conda environment:

.. code-block::

    conda env create --file=environment.yml

And enable it:

.. code-block::

    conda activate pn_env

Optionally the ipython kernel can be specified to installed for the pyneuromodulation_test conda environment:

.. code-block::

    ipython kernel install --user --name=pn_env

To install py_neuromodulation in an editable development version inside your conda enviroment, type the following inside the py_neuromodulation root directory:

.. code-block::

    conda develop .

Then py_neuromodulation can be imported via:

.. code-block::

    import py_neuromodulation

The main modules include running real time enabled feature preprocessing based on `iEEG BIDS <https://www.nature.com/articles/s41597-019-0105-7>`_ data.

Different features can be enabled/disabled and parametrized in the `https://github.com/neuromodulation/py_neuromodulation/blob/main/pyneuromodulation/nm_settings.json>`_.

The current implementation mainly focuses band power and `sharpwave <https://www.sciencedirect.com/science/article/abs/pii/S1364661316302182>`_ feature estimation.

An example folder with a mock subject and derivate `feature <https://github.com/neuromodulation/py_neuromodulation/tree/main/examples/data>`_ set was estimated.

To run feature estimation given the example BIDS data run in root directory.

.. code-block::

    python main.py

This will write a feature_arr.csv file in the 'examples/data/derivatives' folder.

For further documentation view `ParametrizationDefinition <ParametrizationDefinition.html#>`_ for description of necessary parametrization files.
`FeatureEstimationDemo <FeatureEstimationDemo.html#>`_ walks through an example feature estimation and explains sharpwave estimation.
