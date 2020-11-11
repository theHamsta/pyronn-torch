#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""
This tests are unrelated to pyronn_torch itself.
But rather for wrapping PYRO-NN for other frameworks.
"""

import tempfile

import pytest

from pyronn_torch.codegen import generate_shared_object


def test_wrap_walberla():
    import pytest
    pytest.importorskip("walberla_app")

    from walberla_app.kernel_call_nodes import WalberlaModule

    generate_shared_object(tempfile.TemporaryDirectory, None, show_code=True,
                           framework_module_class=WalberlaModule, generate_code_only=True)


@pytest.mark.xfail(reason="allow failure", strict=False)
def test_wrap_tensorflow():
    import pytest
    pytest.importorskip("pystencils_autodiff")

    from pystencils_autodiff.backends.astnodes import TensorflowModule

    generate_shared_object(tempfile.TemporaryDirectory, None, show_code=True,
                           framework_module_class=TensorflowModule, generate_code_only=True)


def test_wrap_torch():
    import pytest
    pytest.importorskip("pystencils_autodiff")

    from pystencils_autodiff.backends.astnodes import TorchModule

    generate_shared_object(tempfile.TemporaryDirectory, None, show_code=True,
                           framework_module_class=TorchModule, generate_code_only=True)
