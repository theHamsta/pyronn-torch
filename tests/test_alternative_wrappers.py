#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""
This tests are unrelated to pyronn_torch itself.
But rather for wrapping PYRO-NN for other frameworks.
"""

import tempfile

from pyronn_torch.codegen import generate_shared_object


def test_wrap_walberla():
    import pytest
    pytest.importorskip("walberla_app")

    from walberla_app.kernel_call_nodes import WalberlaModule

    generate_shared_object(tempfile.TemporaryDirectory, None, show_code=True,
                           framework_module_class=WalberlaModule, generate_code_only=True)


def test_wrap_tensorflow():
    pass
