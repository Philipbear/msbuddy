"""Protect exact numerical storage, array access and process-safe lazy decoding."""
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np

from msbuddy._database import Database, write_database
from msbuddy.query import query_neutral_mass, query_precursor_mass, query_fragnl_mass
from msbuddy.base import Adduct, Formula


def _spawn_rows(array):
    return array[[1, 4, -1]].tobytes()


class DatabaseFormatTests(unittest.TestCase):
    def test_exact_float_grid_is_selected_only_for_bit_identical_values(self):
        arrays = {'grid': np.array([1.007825, 12., 18.998403, 1999.998900]),
                  'negative_grid': np.array([-1.007825, 12., -18.998403, 1999.998900], dtype='>f8'),
                  'off_grid': np.array([1. / 3., 2.]),
                  'signed_zero': np.array([0., -0.])}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grid.msbdb'
            manifest = write_database(path, arrays, 'test', block_rows=2)
            self.assertEqual(manifest['arrays']['grid']['codec'], 'exact-float-grid')
            self.assertEqual(manifest['arrays']['negative_grid']['codec'], 'exact-float-grid')
            self.assertEqual(manifest['arrays']['off_grid']['codec'], 'byte-xor')
            self.assertEqual(manifest['arrays']['signed_zero']['codec'], 'byte-xor')
            database = Database(path)
            for name, values in arrays.items():
                decoded = database.materialize(name, verify=True)
                self.assertEqual(decoded.dtype, values.dtype)
                self.assertEqual(decoded.tobytes(), values.tobytes())
            database.close()

    def test_threaded_and_spawned_access(self):
        values = np.arange(1200, dtype=np.int64).reshape(100, 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parallel.msbdb'
            write_database(path, {'values': values}, 'test', block_rows=8)
            database = Database(path, cache_bytes=128)
            array = database.array('values')
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda i: array[i:i+3].tobytes(), range(90)))
            self.assertEqual(results, [values[i:i+3].tobytes() for i in range(90)])
            with mp.get_context('spawn').Pool(2) as pool:
                results = pool.map(_spawn_rows, [array, array])
            self.assertEqual(results, [values[[1, 4, -1]].tobytes()] * 2)
            self.assertLessEqual(database._cached_bytes, database.cache_bytes)
            database.close()

    def test_integer_and_float_bits_roundtrip_including_extremes(self):
        values = np.array([[0, -4, 294, np.iinfo(np.int64).min],
                           [255, -4, 293, np.iinfo(np.int64).max],
                           [166, -4, 0, 0]], dtype=np.int64)
        float_bits = np.array([0, 1 << 63, 0x7ff8000000001234, 0x7ff0000000000000,
                               0xfff0000000000000, 0x3ff0000000000001], dtype=np.uint64)
        arrays = {'integers': values, 'floats': float_bits.view(np.float64),
                  'big_endian': values.astype('>i8'), 'empty': np.empty((0, 12), dtype=np.int32),
                  'unsigned': np.array([[0], [2**64-1]], dtype=np.uint64)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'database.msbdb'
            write_database(path, arrays, 'test', block_rows=2)
            database = Database(path)
            for name, source in arrays.items():
                result = database.materialize(name, verify=True)
                self.assertEqual(source.dtype, result.dtype)
                self.assertEqual(source.shape, result.shape)
                self.assertEqual(source.tobytes(), result.tobytes())
            database.close()

    def test_lazy_slices_fancy_indices_cache_bounds_and_pickle(self):
        source = np.arange(1200, dtype=np.int64).reshape(100,12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'database.msbdb'
            write_database(path, {'table':source}, 'test', block_rows=7)
            database = Database(path,cache_bytes=1500)
            array = database.array('table')
            for index in [0, -1, slice(2,19), slice(None,None,-3), slice(3,3), [0,8,-2],
                          (slice(1,14),slice(2,5)), ([0,8],2)]:
                np.testing.assert_array_equal(array[index],source[index])
                self.assertLessEqual(database._cached_bytes,1500)
            with self.assertRaises(IndexError):array[100]
            with self.assertRaises(IndexError):array[-101]
            with self.assertRaises(IndexError):array[np.array([2**64-1], dtype=np.uint64)]
            np.testing.assert_array_equal(array[np.array([0, 99], dtype=np.uint64)], source[[0, 99]])
            clone = pickle.loads(pickle.dumps(array))
            self.assertIsNone(clone.database._archive)
            self.assertFalse(clone.database._cache)
            np.testing.assert_array_equal(clone[3:18],source[3:18])
            database.close();clone.database.close()

    def test_storage_is_deterministic_and_preserves_duplicates(self):
        arrays={'mass':np.array([12.,12.,25.2]),'formula':np.array([[1,0],[1,0],[2,1]],dtype=np.int64)}
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/name for name in ['first.msbdb','second.msbdb']]
            for path in paths:write_database(path,arrays,'test',block_rows=2)
            self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())
            database=Database(paths[0]);np.testing.assert_array_equal(database.array('formula')[:],arrays['formula']);database.close()

    def test_invalid_format_and_corrupt_payload_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'database.msbdb'
            with zipfile.ZipFile(path,'w') as z:z.writestr('manifest.json',json.dumps({'format':'other','version':1}))
            with self.assertRaisesRegex(ValueError,'Unsupported'):Database(path)
            with self.assertRaisesRegex(ValueError,'numerical'):write_database(path,{'bad':np.array([object()],dtype=object)},'test')

    def test_query_parity_with_lazy_formula_rows_and_boundary_windows(self):
        names=['H2','C','CH2','CH2','C6H12O6','C20H40','C100H100O20']
        forms=[Formula.from_string(n) for n in names]
        order=np.argsort([f.mass for f in forms],kind='stable')
        masses=np.array([forms[i].mass for i in order],dtype=np.float64)
        matrix=np.array([forms[i].array for i in order],dtype=np.int64)
        index=np.searchsorted(masses,np.arange(15000)/10)
        old={'basic_db_mass':masses,'basic_db_formula':matrix,'basic_db_idx':index,
             'halogen_db_mass':np.empty(0),'halogen_db_formula':np.empty((0,12),dtype=np.int64),
             'halogen_db_idx':np.zeros(15000,dtype=np.int64)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'database.msbdb';write_database(path,{'formula':matrix},'test',block_rows=2)
            database=Database(path);new=dict(old,basic_db_formula=database.array('formula'))
            def equal(a,b):
                self.assertEqual([(str(f),float(f.mass),f.charge) for f in a],[(str(f),float(f.mass),f.charge) for f in b])
            for target in list(masses)+[0.,1499.9,1500.,1900.,2000.]:
                for ppm in [False,True]:
                    equal(query_neutral_mass(target,.001,ppm,True,old),query_neutral_mass(target,.001,ppm,True,new))
                    for text in ['[M+H]+','[M-H]-','[2M+H]+']:
                        ion=Adduct(text,text.endswith('+'))
                        a=query_precursor_mass(target,ion,.001,ppm,1,old);b=query_precursor_mass(target,ion,.001,ppm,1,new)
                        equal(a[0],b[0]);equal(a[1],b[1])
                    for fragment in [False,True]:
                        equal(query_fragnl_mass(target,fragment,True,True,True,.001,ppm,1,old),query_fragnl_mass(target,fragment,True,True,True,.001,ppm,1,new))
            database.close()


if __name__=='__main__':unittest.main()
