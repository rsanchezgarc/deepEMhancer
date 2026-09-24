import os
import tempfile
import unittest

from deepEMhancer.utils.loadModel import (
    _keras_compatible_model_path,
    getInputCubeSize,
)


class _ModelWithShape:
  input_shape = (None, 64, 64, 64, 1)


class _ModelWithMultipleInputs:
  input_shape = [(None, 48, 48, 48, 1), (None, 1)]


class LoadModelCompatibilityTests(unittest.TestCase):
  def test_hd5_checkpoint_is_exposed_as_h5_symlink(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      checkpoint = os.path.join(tmp_dir, "checkpoint.hd5")
      with open(checkpoint, "wb"):
        pass

      with _keras_compatible_model_path(checkpoint) as compatible_path:
        self.assertTrue(compatible_path.endswith(".h5"))
        self.assertTrue(os.path.islink(compatible_path))
        self.assertEqual(os.path.realpath(compatible_path), checkpoint)

  def test_h5_checkpoint_is_used_directly(self):
    checkpoint = "/models/checkpoint.h5"
    with _keras_compatible_model_path(checkpoint) as compatible_path:
      self.assertEqual(compatible_path, checkpoint)

  def test_input_cube_size_uses_model_input_shape(self):
    self.assertEqual(getInputCubeSize(_ModelWithShape()), 64)
    self.assertEqual(getInputCubeSize(_ModelWithMultipleInputs()), 48)


if __name__ == "__main__":
  unittest.main()
