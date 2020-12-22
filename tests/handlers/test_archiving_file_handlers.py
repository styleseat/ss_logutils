from __future__ import absolute_import

import collections
import datetime
import functools
import itertools
import logging
import multiprocessing
import os
import random
import uuid

import mock
import pytest

from ss_logutils import handlers, util
from six.moves import range

try:
    range = xrange
except NameError:
    pass

MESSAGE_ALPHABET = [
    chr(x) for x in range(0, 128) if x not in (ord('\n'), ord('\r'))]


@pytest.fixture
def nonexistent_dir_factory(tmpdir):
    def factory():
        while True:
            path = tmpdir / str(uuid.uuid4())
            if not path.check():
                return path

    return factory


@pytest.fixture
def archive_dir(tmpdir):
    return (tmpdir / 'archive').mkdir()


@pytest.fixture
def log_dir(tmpdir):
    return tmpdir


@pytest.fixture
def log_path(tmpdir):
    return tmpdir / 'example.log'


@pytest.fixture
def formatter(request):
    return logging.Formatter(fmt=getattr(request, 'param', None))


@pytest.fixture
def handler_class():
    return handlers.ArchivingFileHandler


@pytest.fixture
def handler_factory(handler_class, log_path, archive_dir, formatter):
    def factory(kwargs):
        handler = handler_class(str(log_path), str(archive_dir), **kwargs)
        handler.setFormatter(formatter)
        return handler

    return factory


@pytest.fixture
def handler(request, handler_factory):
    return handler_factory(getattr(request, 'param', None) or {})


def make_log_record(msg=None, args=None, level=logging.INFO):
    pathname = lineno = func = 'unknown'
    return logging.LogRecord(
        'test', level, pathname, lineno, msg, args, None, func)


def emit(handler, entry):
    if isinstance(entry, dict):
        record = make_log_record(**entry)
    else:
        record = make_log_record(msg=entry)
    handler.handle(record)


@pytest.fixture
def create_log_entry_factory():
    def factory(handler):
        return functools.partial(emit, handler)

    return factory


@pytest.fixture
def create_log_entry(create_log_entry_factory, handler):
    return create_log_entry_factory(handler)


@pytest.fixture
def create_logs_factory(create_log_entry_factory):
    def factory(handler):
        create_log_entry = create_log_entry_factory(handler)

        def emit_records(logs):
            for entry in itertools.chain.from_iterable(logs):
                create_log_entry(entry)

        return emit_records

    return factory


@pytest.fixture
def create_logs(create_logs_factory, handler):
    return create_logs_factory(handler)


def list_archive_paths(archive_dir):
    return archive_dir.listdir(fil=lambda p: os.path.isfile(str(p)))


def iter_archives(archive_dir):
    archive_paths = list_archive_paths(archive_dir)
    for path in archive_paths:
        yield path, path.read()


def iter_archive_contents(archive_dir):
    for _, content in iter_archives(archive_dir):
        yield content


def parse_log(contents):
    return contents.split('\n')[:-1]


@pytest.fixture
def assert_logs_equal(handler, log_path, archive_dir):
    def assert_logs_equal_(expected):
        if not expected:
            assert not log_path.check()
            assert list_archive_paths(archive_dir) == []
            return
        actual_log = parse_log(log_path.read())
        actual_archives = (
            parse_log(a) for a in iter_archive_contents(archive_dir))
        expected_archives, expected_log = expected[0:-1], expected[-1]
        assert actual_log == expected_log
        assert (
            collections.Counter(tuple(a) for a in actual_archives) ==
            collections.Counter(tuple(a) for a in expected_archives))

    return assert_logs_equal_


@pytest.fixture
def assert_logs_contain(handler, log_path, archive_dir):
    def assert_logs_contain_(expected):
        if not expected:
            raise ValueError('Expected contents cannot be empty')
        missing = set(expected)
        log_contents = [
            [log_path.read()],
            iter_archive_contents(archive_dir)]
        for content in itertools.chain.from_iterable(log_contents):
            log = parse_log(content)
            for entry in log:
                missing.discard(entry)
                if not missing:
                    return
        assert missing == set()

    return assert_logs_contain_


class TestInit(object):
    def test_no_rotation_policy(self, handler_factory):
        with pytest.raises(TypeError):
            handler_factory({})

    def test_archive_dir_does_not_exist(
            self, handler_class, log_path, nonexistent_dir_factory):
        with pytest.raises(ValueError):
            handler_class(
                str(log_path), str(nonexistent_dir_factory()), maxBytes=1)

    def test_archive_dir_is_not_a_directory(self, handler_class, log_path):
        log_file = str(log_path)
        with pytest.raises(ValueError):
            handler_class(log_file, log_file, maxBytes=1)

    def test_file_identification(self, handler_factory):
        handlers = [handler_factory(dict(maxBytes=1)) for _ in range(2)]
        file_id = handlers[0].file_id
        assert file_id is not None
        assert handlers[1].file_id == file_id


class TestSizeRotation(object):
    @staticmethod
    @pytest.fixture
    def delay():
        return False

    @staticmethod
    @pytest.fixture
    def handler(handler_factory, max_bytes, delay):
        return handler_factory(dict(maxBytes=max_bytes, delay=delay))

    @pytest.mark.parametrize('max_bytes', [1])
    @pytest.mark.parametrize('delay', [False, True])
    def test_log_exceeds_max_bytes_before_first_entry(
            self, handler, max_bytes, log_path, create_log_entry,
            assert_logs_equal):
        entries = [
            'x' * max_bytes,
            '123',
        ]
        log_path.write(''.join((entries[0], '\n')))
        create_log_entry(entries[1])
        assert_logs_equal([[e] for e in entries])

    @pytest.mark.parametrize('max_bytes', [1])
    def test_message_exceeds_max_bytes(
            self, handler, create_logs, assert_logs_equal):
        logs = [
            ['ab'],
            ['cd'],
        ]
        create_logs(logs)
        assert_logs_equal(logs)

    @pytest.mark.parametrize('max_bytes', [1, 2, 16])
    @pytest.mark.parametrize('delay', [False, True])
    def test_multiple_rollovers(
            self, handler, max_bytes, create_logs, assert_logs_equal):
        message_length = max_bytes - 1
        logs = []
        for i in range(3):
            message = ''.join(
                MESSAGE_ALPHABET[i*message_length + c]
                for c in range(message_length))
            logs.append([message])
        create_logs(logs)
        assert_logs_equal(logs)

    @pytest.mark.parametrize('max_bytes', [6])
    def test_multiple_messages_per_log(
            self, handler, max_bytes, create_logs, assert_logs_equal):
        logs = [
            ['a', 'b', 'c'],
            ['d', 'e', 'f'],
        ]
        create_logs(logs)
        assert_logs_equal(logs)

    @pytest.mark.parametrize('max_bytes', [16])
    def test_rollover_by_other_handler_with_shared_lock(
            self, handler_factory, handler, create_log_entry_factory,
            create_log_entry, assert_logs_equal):
        """The handler should account for rollovers by external actors.

        If two ArchivingFileHandlers share the same lock but no other state,
        and one of the handlers rolls the log, the other handler should
        automatically switch output log files."""
        logs = [['x'], ['y', 'z']]
        entries = list(itertools.chain.from_iterable(logs))
        external_handler = handler_factory(dict(maxBytes=1))
        original_file_id = external_handler.file_id
        external_create_log_entry = create_log_entry_factory(external_handler)
        for entry in entries[:2]:
            external_create_log_entry(entry)
        updated_file_id = external_handler.file_id
        assert updated_file_id is not None
        assert updated_file_id != original_file_id
        create_log_entry(entries[2])
        assert handler.file_id == updated_file_id
        assert_logs_equal(logs)


@pytest.mark.usefixtures('mock_datetime_class')
class TestTimedRotation(object):
    @staticmethod
    @pytest.yield_fixture(autouse=True)
    def patched_datetime_module(mock_datetime_module):
        with mock.patch.object(util, 'datetime', mock_datetime_module):
            yield mock_datetime_module

    @staticmethod
    @pytest.fixture
    def delay():
        return False

    @staticmethod
    @pytest.fixture
    def interval():
        return 1

    @staticmethod
    @pytest.fixture
    def handler(handler_factory, interval, delay):
        return handler_factory(dict(interval=interval, delay=delay))

    def test_interval_has_not_expired(
            self, handler, interval, create_logs, assert_logs_equal):
        logs = [['a', 'b']]
        create_logs(logs)
        assert_logs_equal(logs)

    @pytest.mark.parametrize('delay', [False, True])
    def test_interval_has_expired(
            self, mock_datetime_class, handler, interval, create_logs,
            assert_logs_equal):
        logs = [['a'], ['b']]
        create_logs(logs[0:1])
        mock_datetime_class.utcnow.return_value += datetime.timedelta(
            seconds=interval)
        create_logs(logs[1:])
        assert_logs_equal(logs)

    def test_last_updated_in_future(
            self, mock_datetime_class, handler, interval, create_log_entry,
            assert_logs_equal):
        """
        Test backwards adjustments to the system clock.

        If the system clock moves backwards, the last update could be in the
        future.
        """
        log = ['a', 'b']
        create_log_entry(log[0])
        mock_datetime_class.utcnow.return_value -= datetime.timedelta(
            seconds=interval+1)
        create_log_entry(log[1])
        assert_logs_equal([log])

    def test_rollover_by_other_handler_with_shared_lock(
            self, mock_datetime_class, handler_factory, handler, interval,
            create_log_entry_factory, create_log_entry, assert_logs_equal):
        """The handler should account for rollovers by external actors.

        If two ArchivingFileHandlers share the same lock but no other state,
        and one of the handlers rolls the log, the other handler should
        automatically switch output log files."""
        logs = [['x'], ['y', 'z']]
        entries = list(itertools.chain.from_iterable(logs))
        external_handler = handler_factory(dict(interval=interval))
        original_file_id = external_handler.file_id
        external_create_log_entry = create_log_entry_factory(external_handler)
        external_create_log_entry(entries[0])
        mock_datetime_class.utcnow.return_value += datetime.timedelta(
            seconds=interval)
        external_create_log_entry(entries[1])
        updated_file_id = external_handler.file_id
        assert updated_file_id is not None
        assert updated_file_id != original_file_id
        create_log_entry(entries[2])
        assert handler.file_id == updated_file_id
        assert_logs_equal(logs)


def generate_message(size):
    return ''.join(random.choice(MESSAGE_ALPHABET) for _ in range(size))


def write_log_entries(logger_name, seed, size, count):
    random.seed(seed)
    logger = logging.getLogger(logger_name)
    for i in range(count):
        msg = generate_message(size)
        logger.info(msg)


@pytest.yield_fixture
def restore_random_state():
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.mark.parametrize('handler_class', [
    handlers.ForkSafeArchivingFileHandler])
@pytest.mark.parametrize('handler', [
    dict(maxBytes=64 * 20**10, interval=1),
], indirect=True)
def test_concurrent_writes(
        restore_random_state, handler, log_path, archive_dir,
        assert_logs_contain):
    nprocs = 8
    entries_per_proc = 1000
    entry_size = 4 * 2**10
    time_per_write = 1e-5 * 2**nprocs
    # Enabling coverage doubles the test duration, and not all
    # test environments have the same resources, so make the timeout a
    # generous multiple of the real expected duration in ideal circumstances.
    timeout = 4 * nprocs * entries_per_proc * time_per_write
    logger_name = '.'.join((
        'ss_logutils', 'tests', 'handlers', 'archiving_file_handlers',
        'concurrency'))
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    procs, seeds = [], []
    for i in range(nprocs):
        seed = i
        proc = multiprocessing.Process(
            target=write_log_entries,
            args=(logger_name, seed, entry_size, entries_per_proc))
        proc.daemon = True
        procs.append(proc)
        seeds.append(seed)
    for proc in procs:
        proc.start()
    unfinished_pids = []
    for proc in procs:
        proc.join(timeout)
        if proc.is_alive():
            unfinished_pids.append(proc.pid)
            proc.terminate()
    assert unfinished_pids == []
    archive_sizes = [p.size() for p in list_archive_paths(archive_dir)]
    assert all(s <= handler.maxBytes for s in archive_sizes), \
        'One or more archives exceeds the maximum size, %s b; sizes: %s' % (
            handler.maxBytes, archive_sizes)
    expected_combined_size = nprocs * entries_per_proc * (entry_size + 1)
    actual_combined_size = log_path.size() + sum(archive_sizes)
    assert expected_combined_size == actual_combined_size
    for seed in seeds:
        expected_entries = set()
        random.seed(seed)
        for _ in range(entries_per_proc):
            expected_entries.add(generate_message(entry_size))
        assert_logs_contain(expected_entries)


@pytest.mark.parametrize('handler_class', [
    handlers.NamedLockArchivingFileHandler])
def test_named_lock_handler(handler_factory, reset_locks):
    lock_name = 'my_lock'
    lock = mock.Mock()
    handlers.register_lock(lock_name, lock)
    handler = handler_factory(dict(lockName=lock_name, maxBytes=1))
    assert handler.lock is lock
