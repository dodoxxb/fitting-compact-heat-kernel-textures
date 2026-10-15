"""Bitstream decode, unquantized-field integrity, and input rejection checks."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/round21'))
from joint_codec import encode, decode, modes


class CodecTests(unittest.TestCase):
    def test_roundtrip_and_preserved_fields(self):
        rng = np.random.default_rng(30)
        n = 64
        raw = dict(_kernel_colours=rng.uniform(-1,1,(n,3)).astype('float16'),
                   _kernel_bary_coords=rng.dirichlet([2,2,2],n)[:,:2].astype('float16'),
                   _angles=rng.uniform(0,3.14,(n,)).astype('float16'),
                   _anisotropies=rng.uniform(1,8,n).astype('float16'),
                   _thresholds=rng.uniform(.1,.8,n).astype('float16'),
                   _sharpnesses=rng.uniform(.1,2,n).astype('float16'),
                   _mean_colour=np.array([.5,.5,.5],dtype='float16'),
                   _kernel_face_ids=np.arange(n,dtype='int32'))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder/'source.npz'
            target = folder/'encoded.jhk'
            np.savez_compressed(source, **raw)
            record = encode(source, target, modes()['C8B12S8'])
            decoded = decode(target)
            self.assertEqual(record['bytes'], target.stat().st_size)
            for key in raw:
                self.assertTrue(np.isfinite(decoded[key]).all())
                self.assertEqual(raw[key].shape, decoded[key].shape)
                if key not in modes()['C8B12S8']:
                    self.assertEqual(raw[key].tobytes(), decoded[key].tobytes())
                    self.assertEqual(raw[key].dtype, decoded[key].dtype)
            raw['_kernel_colours'][0,0]=np.nan
            np.savez_compressed(source, **raw)
            with self.assertRaises(AssertionError):
                encode(source, target, modes()['C8B12S8'])


if __name__ == '__main__':
    unittest.main()
