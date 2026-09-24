import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from deepEMhancer.applyProcessVol.processVol import AutoProcessVol


class _ResourceExhaustedError(Exception):
  pass


class _OutOfMemoryModel:
  def predict_on_batch(self, batch):
    raise _ResourceExhaustedError("out of memory")


class ProcessVolumeTests(unittest.TestCase):
  def test_out_of_memory_error_recommends_a_smaller_batch(self):
    tensorflow = types.ModuleType("tensorflow")
    tensorflow_errors = types.ModuleType("tensorflow.errors")
    tensorflow_errors.ResourceExhaustedError = _ResourceExhaustedError
    tensorflow.errors = tensorflow_errors

    processor = AutoProcessVol.__new__(AutoProcessVol)
    processor.model = _OutOfMemoryModel()
    processor.batch_size_per_gpu = 8

    with patch.dict(sys.modules, {
      "tensorflow": tensorflow,
      "tensorflow.errors": tensorflow_errors,
    }):
      with self.assertRaisesRegex(
        RuntimeError,
        r"--batch_size 8.*--batch_size 4",
      ):
        processor._predictOnBatch(np.zeros((8, 2, 2, 2), dtype=np.float32))


if __name__ == "__main__":
  unittest.main()
