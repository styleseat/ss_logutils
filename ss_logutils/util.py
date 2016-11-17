from __future__ import absolute_import

import datetime
import struct


class Timestamp(int):
    FORMAT = '!q'

    def write(self, fp):
        fp.write(struct.pack(self.FORMAT, self))
        return self

    @classmethod
    def read(cls, fp):
        try:
            value, = struct.unpack(cls.FORMAT, fp.read())
        except struct.error as e:
            raise ValueError(str(e))
        return cls(value)

    @classmethod
    def current(cls):
        now = datetime.datetime.utcnow().replace(microsecond=0)
        return cls(
            (now - datetime.datetime.utcfromtimestamp(0)).total_seconds())
