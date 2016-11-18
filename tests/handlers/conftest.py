from __future__ import absolute_import

import pytest

from ss_logutils import handlers


@pytest.yield_fixture
def reset_locks():
    yield
    handlers._handler_locks.clear()
