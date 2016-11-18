from __future__ import absolute_import

import mock
import pytest

from ss_logutils.handlers import get_lock, register_lock

pytestmark = pytest.mark.usefixtures('reset_locks')


@pytest.fixture
def lock_name():
    return 'rose'


@pytest.fixture
def lock():
    return mock.Mock()


def test_get_unregistered_lock(lock_name):
    with pytest.raises(KeyError):
        get_lock(lock_name)


def test_repeat_register(lock_name, lock):
    register_lock(lock_name, lock)
    with pytest.raises(ValueError):
        register_lock(lock_name, lock)
    assert get_lock(lock_name) is lock


def test_register_get_composition(lock_name, lock):
    register_lock(lock_name, lock)
    assert get_lock(lock_name) is lock
