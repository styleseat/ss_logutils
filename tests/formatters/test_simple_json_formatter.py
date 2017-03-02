from __future__ import absolute_import

import logging
import json

import mock
import pytest

from ss_logutils import formatters


@pytest.fixture
def formatter_class():
    return formatters.SimpleJSONFormatter


@pytest.fixture
def formatter(formatter_class):
    return formatter_class()


def test_init(formatter_class):
    assert isinstance(formatter_class(), logging.Formatter)


def test_default_serializer(formatter):
    data = dict(a='1', b='2')
    record = mock.Mock(msg=data)
    assert formatter.format(record) == json.dumps(data)


def test_custom_serializer(formatter_class):
    record = mock.Mock()
    serializer = mock.Mock(spec_set=True)
    formatter = formatter_class(json_serializer=serializer)
    assert formatter.format(record) == serializer.return_value
    serializer.assert_called_once_with(record.msg)
