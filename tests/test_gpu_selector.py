import os
import unittest
from unittest.mock import patch

from deepEMhancer.utils.gpuSelector import (
  configureGpuEnvironment,
  resolveDesiredGpus,
)


class GpuSelectorTests(unittest.TestCase):
  def setUp(self):
    self.original_visible_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)

  def tearDown(self):
    if self.original_visible_devices is None:
      os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
      os.environ["CUDA_VISIBLE_DEVICES"] = self.original_visible_devices

  def test_first_gpu_uses_zero_based_cuda_index(self):
    gpu_ids, number_of_gpus = configureGpuEnvironment("0")
    self.assertEqual(gpu_ids, [0])
    self.assertEqual(number_of_gpus, 1)
    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

  def test_cpu_selection_masks_all_gpus(self):
    gpu_ids, number_of_gpus = configureGpuEnvironment("-1")
    self.assertEqual(gpu_ids, [None])
    self.assertEqual(number_of_gpus, 1)
    self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")

  @patch("deepEMhancer.utils.gpuSelector.check_output")
  @patch("deepEMhancer.utils.gpuSelector.shutil.which")
  def test_all_uses_nvidia_smi_exe_when_needed(self, which, check_output_mock):
    which.side_effect = lambda command: (
      "/mnt/c/Windows/System32/nvidia-smi.exe" if command == "nvidia-smi.exe" else None
    )
    check_output_mock.return_value = "GPU 0: NVIDIA RTX 5000\nGPU 1: NVIDIA RTX 5000\n"

    gpu_ids, number_of_gpus = resolveDesiredGpus("all")

    self.assertEqual(gpu_ids, [0, 1])
    self.assertEqual(number_of_gpus, 2)
    check_output_mock.assert_called_once_with(
      ["/mnt/c/Windows/System32/nvidia-smi.exe", "-L"], text=True
    )

  @patch("deepEMhancer.utils.gpuSelector.check_output")
  @patch("deepEMhancer.utils.gpuSelector.shutil.which")
  def test_all_retries_with_nvidia_smi_exe_if_linux_command_fails(self, which, check_output_mock):
    executables = {
      "nvidia-smi": "/usr/bin/nvidia-smi",
      "nvidia-smi.exe": "/mnt/c/Windows/System32/nvidia-smi.exe",
    }
    which.side_effect = executables.get
    check_output_mock.side_effect = [
      OSError("Linux nvidia-smi is unavailable"),
      "GPU 0: NVIDIA RTX 5000\n",
    ]

    gpu_ids, number_of_gpus = resolveDesiredGpus("all")

    self.assertEqual(gpu_ids, [0])
    self.assertEqual(number_of_gpus, 1)
    self.assertEqual(check_output_mock.call_count, 2)

  @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False)
  def test_all_respects_existing_visible_devices(self):
    gpu_ids, number_of_gpus = resolveDesiredGpus("all")
    self.assertEqual(gpu_ids, ["2", "3"])
    self.assertEqual(number_of_gpus, 2)


if __name__ == "__main__":
  unittest.main()
