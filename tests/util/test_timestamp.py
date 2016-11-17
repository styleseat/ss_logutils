from __future__ import absolute_import

import datetime
import io
import random

import mock
import pytest

from ss_logutils import util
from ss_logutils.util import Timestamp


class TestRead(object):
    def test_missing_bytes(self):
        stream = io.BytesIO()
        with pytest.raises(ValueError):
            Timestamp.read(stream)

    def test_too_many_bytes(self):
        stream = io.BytesIO(b'\x00\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF')
        with pytest.raises(ValueError):
            Timestamp.read(stream)

    def test_valid(self):
        stream = io.BytesIO(b'\x7F\xFF\xFF\xFF\xFF\xFF\xFF\xFF')
        assert Timestamp.read(stream) == Timestamp(0x7FFFFFFFFFFFFFFF)


def test_write():
    stream = io.BytesIO()
    ts = Timestamp(0xEDCB)
    assert ts.write(stream) is ts
    assert stream.getvalue() == b'\x00\x00\x00\x00\x00\x00\xED\xCB'


def test_read_write_composition():
    original_ts = Timestamp(random.randint(0, 2**63-1))
    stream = io.BytesIO()
    original_ts.write(stream)
    stream.seek(0)
    read_ts = original_ts.read(stream)
    assert original_ts == read_ts


def test_current(mock_datetime_module, mock_datetime_class):
    mock_datetime_class.utcnow.return_value = datetime.datetime(
        1970, 1, 1, 10)
    with mock.patch.object(util, 'datetime', mock_datetime_module):
        assert Timestamp.current() == Timestamp(10 * 3600)
