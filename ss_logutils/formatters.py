from __future__ import absolute_import

import json
import logging


class SimpleJSONFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._serialize = kwargs.pop('json_serializer', json.dumps)
        super(SimpleJSONFormatter, self).__init__(*args, **kwargs)

    def format(self, record):
        return self._serialize(record.msg)
