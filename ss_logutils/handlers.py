from __future__ import absolute_import

import codecs
import datetime
import errno
import logging.handlers
import multiprocessing
import os
import sys
import threading
import uuid

from .util import Timestamp

_lock = threading.Lock()
_handler_locks = {}


def register_lock(name, lock):
    """Register a named lock."""
    with _lock:
        if name in _handler_locks:
            raise ValueError('Lock %s has already been registered' % (name,))
        _handler_locks[name] = lock


def get_lock(name):
    """Retrieve a named lock."""
    return _handler_locks[name]


class ArchivingFileHandler(logging.handlers.BaseRotatingHandler):
    def __init__(
            self, filename, archiveDir, encoding=None, delay=False,
            maxBytes=0, interval=0):
        if maxBytes < 1 and interval < 1:
            raise TypeError(
                'Archival requires either a byte limit or interval')
        self.archiveDir = os.path.abspath(archiveDir)
        if not os.path.isdir(self.archiveDir):
            raise ValueError(
                'Invalid archive directory: %s' % (self.archiveDir,))
        self.maxBytes = maxBytes
        self.interval = interval
        if interval > 0:
            self.tsFilename = '.'.join((os.path.abspath(filename), 'ts'))
        super(ArchivingFileHandler, self).__init__(
            filename, 'a', encoding=encoding, delay=delay)

    def doRollover(self):
        self.__close()
        archiveBasename = '.'.join((
            os.path.basename(self.baseFilename),
            datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'),
            str(uuid.uuid4())))
        archiveFilename = os.path.join(self.archiveDir, archiveBasename)
        os.rename(self.baseFilename, archiveFilename)
        if self.interval > 0:
            self._updateLastRollover()
        if not self.delay:
            self.stream = self._open()

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()
            opened = True
        else:
            opened = False
        if self.maxBytes > 0:
            # The POSIX open O_APPEND specification doesn't require the file
            # position prior to a write to match the file's size, and even if
            # it did, Windows doesn't believe in POSIX.
            # http://pubs.opengroup.org/onlinepubs/009695399/functions/open.html
            # http://stackoverflow.com/questions/31680677
            self.stream.seek(0, 2)
            logBytes = self.stream.tell()
            if logBytes > self.maxBytes:
                # If the log size exceeds the max, either the max has changed,
                # or another thread of execution rolled the log and the log
                # file should be re-opened.
                if opened:
                    return True
                self.__close()
                self.stream = self._open()
                self.stream.seek(0, 2)
                logBytes = self.stream.tell()
                if logBytes > self.maxBytes:
                    return True
            if logBytes > 0:
                msgLength = len(self.format(record)) + len('\n')
                if logBytes + msgLength > self.maxBytes:
                    return True
        if self.interval > 0:
            now = Timestamp.current()
            if now - self.lastRollover >= self.interval:
                # If another thread of execution rolled the log, the local
                # timestamp value will be out of date, so refresh it.
                self._readLastRollover()
                if now - self.lastRollover >= self.interval:
                    return True
        return False

    def _readLastRollover(self):
        with open(self.tsFilename, 'rb') as fp:
            self.lastRollover = Timestamp.read(fp)

    def _updateLastRollover(self):
        with open(self.tsFilename, 'wb') as fp:
            self.lastRollover = Timestamp.current().write(fp)

    def _open(self):
        # Open files in binary mode to ensure tell works in Python 3.
        # See https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files  # noqa
        encoding = (
            sys.getdefaultencoding() if self.encoding is None else
            self.encoding)
        stream = codecs.open(self.baseFilename, self.mode, encoding)
        if self.interval > 0 and not hasattr(self, 'lastRollover'):
            try:
                self._readLastRollover()
            except (IOError, OSError) as e:
                if e.errno != errno.ENOENT:
                    raise
                self._updateLastRollover()
        return stream

    def __close(self):
        # Make this method class-private in case the superclass eventually
        # defines a _close method.
        if self.stream:
            self.stream.close()
            self.stream = None


class ForkSafeArchivingFileHandler(ArchivingFileHandler):
    """Supports concurrent writes in forked processes."""

    def createLock(self):
        self.lock = multiprocessing.RLock()


class NamedLockArchivingFileHandler(ArchivingFileHandler):
    """Handler which uses a named, external lock for concurrency control.

    This class may be useful in situations where child processes configure
    their own logging hierarchies but may share file outputs.
    """

    def __init__(
            self, filename, archiveDir, lockName, encoding=None, delay=False,
            maxBytes=0, interval=0):
        self.lockName = lockName
        super(NamedLockArchivingFileHandler, self).__init__(
            filename, archiveDir, encoding=encoding, delay=delay,
            maxBytes=maxBytes, interval=interval)

    def createLock(self):
        self.lock = get_lock(self.lockName)
