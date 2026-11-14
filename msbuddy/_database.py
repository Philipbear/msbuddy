"""Lossless, indexed numerical database storage with bounded decoding caches."""
from collections import OrderedDict
import hashlib
import json
import operator
import os
from pathlib import Path
import threading
import zipfile

import numpy as np


FORMAT = 'msbuddy-numeric-database'
VERSION = 1
DEFAULT_BLOCK_ROWS = 4096


def _write_member(archive, name, payload):
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, payload, compresslevel=3)


def _byte_predict(array):
    values = np.ascontiguousarray(array)
    planes = values.view(np.uint8).reshape(-1, values.dtype.itemsize).T.copy()
    if planes.shape[1] > 1:
        planes[:, 1:] = np.bitwise_xor(planes[:, 1:], planes[:, :-1])
    return planes.tobytes()


def _byte_restore(payload, dtype, shape):
    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(payload) != size:
        raise ValueError('Invalid database block length.')
    planes = np.frombuffer(payload, dtype=np.uint8).reshape(dtype.itemsize, -1).copy()
    np.bitwise_xor.accumulate(planes, axis=1, out=planes)
    return planes.T.copy().view(dtype).reshape(shape)


def _exact_grid(array):
    """Use fixed point only when conversion restores every original float bit."""
    if array.ndim != 1 or array.dtype.kind != 'f' or array.dtype.itemsize != 8 or not len(array):
        return None
    if not np.isfinite(array).all():
        return None
    scaled = array.astype(np.float64) * 1000000
    if np.any(scaled < -(2. ** 63)) or np.any(scaled >= 2. ** 63):
        return None
    grid = np.rint(scaled).astype('<i8')
    restored = (grid.astype(np.float64) / 1000000).astype(array.dtype)
    return grid if restored.tobytes() == array.tobytes() else None


def write_database(path, arrays, role, block_rows=DEFAULT_BLOCK_ROWS, source_sha256=None):
    """Store every numerical value, dtype, row and duplicate without rounding."""
    if not isinstance(block_rows, int) or isinstance(block_rows, bool) or block_rows < 1:
        raise ValueError('block_rows must be a positive integer.')
    manifest = {'format': FORMAT, 'version': VERSION, 'role': role, 'arrays': {}}
    if source_sha256 is not None:
        manifest['source_sha256'] = source_sha256
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    try:
        with zipfile.ZipFile(temporary, 'w', allowZip64=True) as archive:
            for name, value in arrays.items():
                if not isinstance(name, str) or not name or '/' in name or '\\' in name or name in ('.', '..'):
                    raise ValueError('Invalid database array name.')
                array = np.asarray(value)
                if array.ndim not in (1, 2) or array.dtype.kind not in 'biufc':
                    raise ValueError('Database arrays must be one- or two-dimensional numerical arrays.')
                array = np.ascontiguousarray(array)
                spec = {'shape': list(array.shape), 'dtype': array.dtype.str, 'block_rows': block_rows,
                        'sha256': hashlib.sha256(array.tobytes()).hexdigest(), 'blocks': []}
                grid = _exact_grid(array)
                if grid is not None:
                    spec.update(codec='exact-float-grid', grid_scale=1000000)
                elif array.ndim == 2 and array.dtype.kind in 'iu':
                    columns = []
                    for column in range(array.shape[1]):
                        low = int(array[:, column].min()) if len(array) else 0
                        high = int(array[:, column].max()) if len(array) else 0
                        span = high - low
                        width = 0 if not span else next(w for w in (1, 2, 4, 8) if span < (1 << (8 * w)))
                        columns.append({'minimum': low, 'width': width})
                    spec.update(codec='integer-columns', columns=columns)
                else:
                    spec['codec'] = 'byte-xor'
                for index, start in enumerate(range(0, len(array), block_rows)):
                    block = array[start:start + block_rows]
                    if spec['codec'] == 'exact-float-grid':
                        delta = grid[start:start + block_rows].copy()
                        if len(delta) > 1:
                            delta[1:] -= delta[:-1].copy()
                        payload = _byte_predict(delta)
                    elif spec['codec'] == 'integer-columns':
                        parts = []
                        for column, field in enumerate(spec['columns']):
                            if not field['width']:
                                continue
                            minimum = np.uint64(field['minimum'] % (1 << 64))
                            values = block[:, column].astype(np.uint64) - minimum
                            parts.append(values.astype('<u' + str(field['width'])).tobytes())
                        payload = b''.join(parts)
                    else:
                        payload = _byte_predict(block)
                    member = name + '/' + str(index) + '.bin'
                    _write_member(archive, member, payload)
                    spec['blocks'].append({'member': member, 'rows': len(block), 'bytes': len(payload)})
                manifest['arrays'][name] = spec
            # Padding keeps small valid databases above the historical ready-file threshold.
            payload = json.dumps(manifest, separators=(',', ':'), sort_keys=True).encode()
            _write_member(archive, 'manifest.json', payload + b' ' * max(0, 1200 - len(payload)))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


class Database:
    """Open a local database; live handles and decoded blocks stay process-local."""
    def __init__(self, path, cache_bytes=32 * 1024 * 1024):
        self.path = str(Path(path).resolve())
        self.cache_bytes = operator.index(cache_bytes)
        if self.cache_bytes < 0:
            raise ValueError('Database cache size cannot be negative.')
        self._archive = None
        self._pid = None
        self._cache = OrderedDict()
        self._cached_bytes = 0
        self._lock = threading.RLock()
        with zipfile.ZipFile(self.path) as archive:
            info = archive.getinfo('manifest.json')
            if info.file_size > 8 * 1024 * 1024:
                raise ValueError('Database manifest is too large.')
            self.manifest = json.loads(archive.read(info))
            self._validate(archive)

    def _validate(self, archive):
        manifest = self.manifest
        if manifest.get('format') != FORMAT or manifest.get('version') != VERSION:
            raise ValueError('Unsupported numerical database format.')
        arrays = manifest.get('arrays')
        if not isinstance(arrays, dict) or not arrays:
            raise ValueError('Database contains no arrays.')
        for name, spec in arrays.items():
            shape = spec.get('shape', [])
            dtype = np.dtype(spec['dtype'])
            block_rows = spec['block_rows']
            if (len(shape) not in (1, 2) or any(type(n) is not int or n < 0 for n in shape)
                    or dtype.kind not in 'biufc' or type(block_rows) is not int or block_rows < 1
                    or len(spec['blocks']) != (shape[0] + block_rows - 1) // block_rows):
                raise ValueError('Invalid database array metadata.')
            if spec['codec'] == 'integer-columns':
                columns = spec['columns']
                if len(shape) != 2 or dtype.kind not in 'iu' or len(columns) != shape[1]:
                    raise ValueError('Invalid integer database columns.')
                for field in columns:
                    if type(field['minimum']) is not int or field['width'] not in (0, 1, 2, 4, 8):
                        raise ValueError('Invalid integer database encoding.')
            elif spec['codec'] == 'exact-float-grid':
                if len(shape) != 1 or dtype.kind != 'f' or dtype.itemsize != 8 or spec.get('grid_scale') != 1000000:
                    raise ValueError('Invalid exact floating-point grid metadata.')
            elif spec['codec'] != 'byte-xor':
                raise ValueError('Unsupported database array encoding.')
            for index, block in enumerate(spec['blocks']):
                rows = min(block_rows, shape[0] - index * block_rows)
                expected = rows * (sum(f['width'] for f in spec['columns']) if spec['codec'] == 'integer-columns'
                                   else int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize)
                if (block['member'] != name + '/' + str(index) + '.bin' or block['rows'] != rows
                        or block['bytes'] != expected or archive.getinfo(block['member']).file_size != expected):
                    raise ValueError('Invalid database block metadata.')

    def __getstate__(self):
        return {'path': self.path, 'cache_bytes': self.cache_bytes, 'manifest': self.manifest}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._archive = None
        self._pid = None
        self._cache = OrderedDict()
        self._cached_bytes = 0
        self._lock = threading.RLock()

    def close(self):
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._cache.clear()
        self._cached_bytes = 0

    def _open(self):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.path)
        return self._archive

    def _payload(self, name, index):
        if self._pid is not None and self._pid != os.getpid():
            self._lock = threading.RLock()
        with self._lock:
            archive = self._open()
            key = (name, index)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            item = self.manifest['arrays'][name]['blocks'][index]
            payload = archive.read(item['member'])
            if len(payload) != item['bytes']:
                raise ValueError('Invalid database block length.')
            if len(payload) <= self.cache_bytes:
                while self._cache and self._cached_bytes + len(payload) > self.cache_bytes:
                    _, old = self._cache.popitem(last=False)
                    self._cached_bytes -= len(old)
                self._cache[key] = payload
                self._cached_bytes += len(payload)
            return payload

    def block(self, name, index):
        return self.rows(name, index, 0, self.manifest['arrays'][name]['blocks'][index]['rows'])

    def rows(self, name, index, start, stop):
        spec = self.manifest['arrays'][name]
        item = spec['blocks'][index]
        payload = self._payload(name, index)
        shape = (stop - start, *spec['shape'][1:])
        dtype = np.dtype(spec['dtype'])
        if spec['codec'] == 'integer-columns':
            value = np.empty(shape, dtype=dtype)
            offset = 0
            for column, field in enumerate(spec['columns']):
                width = field['width']
                if width:
                    count = item['rows'] * width
                    encoded = np.frombuffer(payload, dtype='<u' + str(width), count=stop - start, offset=offset + start * width)
                    restored = encoded.astype(np.uint64) + np.uint64(field['minimum'] % (1 << 64))
                    value[:, column] = restored.astype(dtype)
                    offset += count
                else:
                    value[:, column] = field['minimum']
        elif spec['codec'] == 'exact-float-grid':
            delta = _byte_restore(payload, np.dtype('<i8'), (item['rows'],))
            grid = np.cumsum(delta, dtype=np.int64)
            value = (grid[start:stop].astype(np.float64) / spec['grid_scale']).astype(dtype)
        else:
            value = _byte_restore(payload, dtype, (item['rows'], *spec['shape'][1:]))[start:stop]
        value.flags.writeable = False
        return value

    def array(self, name):
        return Array(self, name)

    def materialize(self, name, verify=False):
        result = self.array(name)._range(0, self.manifest['arrays'][name]['shape'][0])
        if verify and hashlib.sha256(result.tobytes()).hexdigest() != self.manifest['arrays'][name]['sha256']:
            raise ValueError('Database array checksum does not match.')
        return result


class Array:
    """Numerical row access backed by a bounded block cache."""
    def __init__(self, database, name):
        self.database = database
        self.name = name
        spec = database.manifest['arrays'][name]
        self.shape = tuple(spec['shape'])
        self.dtype = np.dtype(spec['dtype'])
        self.ndim = len(self.shape)
        self._block_rows = spec['block_rows']

    def __len__(self):
        return self.shape[0]

    def _range(self, start, stop):
        if stop <= start:
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        first, last = start // self._block_rows, (stop - 1) // self._block_rows
        if first == last:
            return self.database.rows(self.name, first, start % self._block_rows, (stop - 1) % self._block_rows + 1)
        value = np.empty((stop - start, *self.shape[1:]), dtype=self.dtype)
        for index in range(first, last + 1):
            left = max(start, index * self._block_rows)
            count = self.database.manifest['arrays'][self.name]['blocks'][index]['rows']
            right = min(stop, index * self._block_rows + count)
            value[left - start:right - start] = self.database.rows(self.name, index, left - index * self._block_rows, right - index * self._block_rows)
        return value

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return self._range(start, stop)
            return self._rows(np.arange(start, stop, step, dtype=np.int64))
        if isinstance(index, tuple):
            value = self[index[0]]
            return value[index[1:]] if isinstance(index[0], (int, np.integer)) else value[(slice(None), *index[1:])]
        if isinstance(index, (list, np.ndarray)):
            return self._rows(np.asarray(index))
        row = operator.index(index)
        if row < 0:
            row += len(self)
        if not 0 <= row < len(self):
            raise IndexError('Database row index out of bounds.')
        start = row % self._block_rows
        return self.database.rows(self.name, row // self._block_rows, start, start + 1)[0]

    def _rows(self, indices):
        if indices.dtype.kind == 'b':
            if indices.shape != (len(self),):
                raise IndexError('Invalid database boolean index shape.')
            indices = np.flatnonzero(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu':
            raise IndexError('Database indices must be a one-dimensional integer array.')
        if indices.dtype.kind == 'u' and np.any(indices >= len(self)):
            raise IndexError('Database row index out of bounds.')
        indices = indices.astype(np.int64, copy=True)
        indices[indices < 0] += len(self)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError('Database row index out of bounds.')
        result = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)
        block_numbers = indices // self._block_rows
        for block_number in np.unique(block_numbers):
            mask = block_numbers == block_number
            positions = indices[mask] % self._block_rows
            left, right = int(positions.min()), int(positions.max()) + 1
            result[mask] = self.database.rows(self.name, int(block_number), left, right)[positions - left]
        return result

    def __array__(self, dtype=None, copy=None):
        value = self._range(0, len(self))
        if dtype is not None:
            value = value.astype(dtype, copy=False)
        return value.copy() if copy else value


def convert_legacy(source, target, role, block_rows=DEFAULT_BLOCK_ROWS):
    """Convert a trusted legacy joblib database; inference never needs pickle here."""
    import joblib
    original = joblib.load(source)
    if role == 'formula':
        names = ('mass', 'formula', 'index')
        arrays = {prefix + '_' + name: value for prefix, group in zip(('basic', 'halogen'), original)
                  for name, value in zip(names, group)}
    elif role == 'common':
        arrays = dict(zip(('common_loss', 'common_fragment'), original))
    else:
        raise ValueError('Database role must be formula or common.')
    return write_database(target, arrays, role, block_rows,
                          source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())


def load_database(path):
    """Return the historical numerical groups, with lazy formula rows."""
    database = Database(path)
    try:
        role = database.manifest.get('role')
        if role == 'common':
            expected = {'common_loss', 'common_fragment'}
            if set(database.manifest['arrays']) != expected:
                raise ValueError('Invalid common database arrays.')
            values = [database.materialize(name, verify=True) for name in ('common_loss', 'common_fragment')]
            if any(a.ndim != 2 or a.shape[1] != 12 or a.dtype.kind not in 'iu' for a in values):
                raise ValueError('Invalid common formula arrays.')
            database.close()
            return values
        if role != 'formula':
            raise ValueError('Unsupported database role.')
        expected = {prefix + '_' + field for prefix in ('basic', 'halogen') for field in ('mass', 'formula', 'index')}
        if set(database.manifest['arrays']) != expected:
            raise ValueError('Invalid formula database arrays.')
        groups = []
        for prefix in ('basic', 'halogen'):
            mass = database.materialize(prefix + '_mass', verify=True)
            index = database.materialize(prefix + '_index', verify=True)
            formula = database.array(prefix + '_formula')
            if (mass.ndim != 1 or mass.dtype.kind != 'f' or not np.isfinite(mass).all()
                    or np.any(mass[1:] < mass[:-1]) or formula.shape != (len(mass), 12)
                    or formula.dtype.kind not in 'iu' or index.ndim != 1 or index.dtype.kind not in 'iu'
                    or np.any(index < 0) or np.any(index > len(mass)) or np.any(index[1:] < index[:-1])):
                raise ValueError('Invalid formula database dimensions or search index.')
            mass.flags.writeable = False
            index.flags.writeable = False
            groups.append((mass, formula, index))
        database._cache.clear()
        database._cached_bytes = 0
        return groups
    except Exception:
        database.close()
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Losslessly convert a trusted legacy msbuddy database.')
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--role', choices=['formula', 'common'], required=True)
    arguments = parser.parse_args()
    convert_legacy(arguments.source, arguments.target, arguments.role)
