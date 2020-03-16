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


Usage
=====

 
.. code-block:: python

    import pyronn_torch

    #ConeBeamProjector(volume_shape,
    #                  volume_spacing,
    #                  volume_origin,
    #                  projection_shape,
    #                  projection_spacing,
    #                  projection_origin,
    #                  projection_matrices)
    projector = pyronn_torch.ConeBeamProjector(
        (128, 128, 128),
        (2.0, 2.0, 2.0),
        (-127.5, -127.5, -127.5),
        (2, 480, 620),
        [1.0, 1.0],
        (0, 0),
        np.array([[[-3.10e+2, -1.20e+03,  0.00e+00,  1.86e+5],
                   [-2.40e+2,  0.00e+00,  1.20e+03,  1.44e+5],
                   [-1.00e+00,  0.00e+00,  0.00e+00,  6.00e+2]],
                  [[-2.89009888e+2, -1.20522754e+3, -1.02473585e-13,
                    1.86000000e+5],
                   [-2.39963440e+2, -4.18857765e+0,  1.20000000e+3,
                    1.44000000e+5],
                   [-9.99847710e-01, -1.74524058e-2,  0.00000000e+0,
                    6.00000000e+2]]]) # two projection matrices
    )
    projection = projector.new_projection_tensor(requires_grad=True)

    projection += 1.
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

