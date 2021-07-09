.. image:: https://badge.fury.io/py/pyronn-torch.svg
   :target: https://badge.fury.io/py/pyronn-torch
   :alt: PyPI version

.. image:: https://travis-ci.org/theHamsta/pyronn-torch.svg?branch=master
    :target: https://travis-ci.org/theHamsta/pyronn-torch

============
pyronn-torch
============

This repository provides PyTorch bindings for `PYRO-NN <https://github.com/csyben/PYRO-NN>`_,
a collection of back-propagatable projectors for CT reconstruction.

Feel free to cite our publication:


.. code-block:: bibtex

    @article{PYRONN2019,
    author = {Syben, Christopher and Michen, Markus and Stimpel, Bernhard and Seitz, Stephan and Ploner, Stefan and Maier, Andreas K.},
    title = {Technical Note: PYRO-NN: Python reconstruction operators in neural networks},
    year = {2019},
    journal = {Medical Physics},
    }


Installation
============

From PyPI:

.. code-block:: bash

    pip install pyronn-torch

From this repository:

.. code-block:: bash

    git clone --recurse-submodules --recursive https://github.com/theHamsta/pyronn-torch.git
    cd pyronn-torch
    pip install torch
    pip install -e .
    
You can build a binary wheel using

.. code-block:: bash
    
    python setup.py bdist_wheel


**Important**

If you're using an older CUDA version you might get an error about ``'-allow-unsupported-compiler'`` not being a
valid compiler option. In that case remove that compiler option from this project's ``setup.py``.

Usage
=====

 
.. code-block:: python

    import pyronn_torch
    import numpy as np

    projector = pyronn_torch.ConeBeamProjector(
        (128, 128, 128),  # volume shape
        (2.0, 2.0, 2.0),  # volume spacing in mm
        (-127.5, -127.5, -127.5),  # volume origin in mm
        (2, 480, 620),  # projection_shape (n, width, height)
        [1.0, 1.0],  # projection_spacing in mm
        (0, 0),  # projection_origin 
        np.array([[[-3.10e+2, -1.20e+03,  0.00e+00,  1.86e+5],
                   [-2.40e+2,  0.00e+00,  1.20e+03,  1.44e+5],
                   [-1.00e+00,  0.00e+00,  0.00e+00,  6.00e+2]],
                  [[-2.89009888e+2, -1.20522754e+3, -1.02473585e-13,
                    1.86000000e+5],
                   [-2.39963440e+2, -4.18857765e+0,  1.20000000e+3,
                    1.44000000e+5],
                   [-9.99847710e-01, -1.74524058e-2,  0.00000000e+0,
                    6.00000000e+2]]])  # two projection matrices in shape (n, 3, 4)
                    # optionally: source_isocenter_distance=1, source_detector_distance=1 for a scalar weighting the projections
    )
    projection = projector.new_projection_tensor(requires_grad=True)

    projection = projection + 1.
    result = projector.project_backward(projection, use_texture=True)

    assert projection.requires_grad
    assert result.requires_grad

    loss = result.mean()
    loss.backward()

Or easier with `PyCONRAD <https://pypi.org/project/pyconrad/>`_ (``pip install pyconrad``)

.. code-block:: python

    projector = pyronn_torch.ConeBeamProjector.from_conrad_config()

The configuration can then be done using `CONRAD <https://github.com/akmaier/CONRAD>`_
(startable using ``conrad`` from command line)

