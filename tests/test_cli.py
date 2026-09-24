import subprocess
import sys
import unittest


class CommandLineTests(unittest.TestCase):
  def test_importing_cli_does_not_import_tensorflow(self):
    command = (
      "import sys; "
      "import deepEMhancer.exeDeepEMhancer; "
      "assert 'tensorflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)

  def test_importing_process_volume_does_not_import_tensorflow(self):
    command = (
      "import sys; "
      "import deepEMhancer.applyProcessVol.processVol; "
      "assert 'tensorflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)

  def test_help_does_not_initialize_tensorflow(self):
    result = subprocess.run(
      [sys.executable, "-m", "deepEMhancer.exeDeepEMhancer", "--help"],
      check=True,
      capture_output=True,
      text=True,
    )
    self.assertIn("DeepEMHancer. Deep post-processing", result.stdout)
    self.assertNotIn("oneDNN custom operations", result.stderr)
    self.assertNotIn("absl::InitializeLog", result.stderr)


if __name__ == "__main__":
  unittest.main()
