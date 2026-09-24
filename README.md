# Deep cryo-EM Map Enhancer (DeepEMhancer)
**DeepEMhancer** is a Python package for post-processing cryo-EM maps, as described in
[DeepEMhancer: a deep learning solution for cryo-EM volume post-processing](https://doi.org/10.1038/s42003-021-02399-1)
by Sanchez-Garcia et al. (2021).

DeepEMhancer was trained on pairs of experimental volumes and atomic model-corrected volumes. Its preferred inputs are
unprocessed half-maps. Post-translational modifications and ligands were not included in the training set, so results for
these features may be inaccurate.

DeepEMhancer performs nonlinear post-processing with two main effects:

- Local sharpening-like post-processing.
- Automatic masking and denoising.

## Table of contents

- [Installation](#installation)
- [Usage guide](#usage-guide)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

Run `deepemhancer -h` for the complete command-line reference.

## Installation

- [Requirements](#requirements)
- [Recommended pip installation](#recommended-pip-installation)
- [Reproducible Conda environment](#reproducible-conda-environment)
- [Install from a source checkout](#install-from-a-source-checkout)

### Requirements

DeepEMhancer has been tested on Linux systems (including WSL2) with Python 3.10-3.13 and TensorFlow 2.21.
The default installation uses TensorFlow's official `and-cuda` extra, which installs compatible CUDA 12 runtime
libraries. A sufficiently recent NVIDIA driver is still required. CPU-only execution is supported but considerably
slower.

### Recommended pip installation

Create and activate a Python 3.10-3.13 virtual environment, then install DeepEMhancer directly
from GitHub. GPU support and its CUDA runtime libraries are installed by default.

```bash
python3 -m venv deepEMhancer_env
source deepEMhancer_env/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/rsanchezgarc/deepEMhancer.git"
deepemhancer --download
deepemhancer -h
```

For a CPU-only installation, set `DEEPEMHANCER_CPU_ONLY` while installing:

```bash
DEEPEMHANCER_CPU_ONLY=1 python -m pip install "git+https://github.com/rsanchezgarc/deepEMhancer.git"
```

### Reproducible Conda environment

The pinned environment is useful for reproducing the dependency versions tested for this release:

```bash
git clone https://github.com/rsanchezgarc/deepEMhancer.git
cd deepEMhancer
conda env create -f deepEMhancer_env.yml -n deepEMhancer_env
conda activate deepEMhancer_env
python -m pip install . --no-deps
deepemhancer --download
```

### Install from a source checkout

For development or to install an unreleased branch:

```bash
git clone https://github.com/rsanchezgarc/deepEMhancer.git
cd deepEMhancer
python3 -m venv deepEMhancer_env
source deepEMhancer_env/bin/activate
python -m pip install .
deepemhancer --download
```

Check that TensorFlow detects the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The command should print at least one `PhysicalDevice` with `device_type='GPU'`. If it prints an empty list,
check the NVIDIA driver and reinstall in a clean environment; do not mix the bundled runtime with separately
installed CUDA or cuDNN packages.

## Usage guide

### Input maps

DeepEMhancer was trained with half-maps, so providing both unprocessed half-maps is preferred:

```bash
deepemhancer -i half1.mrc -i2 half2.mrc -o output.mrc
```

An unprocessed full map directly produced by refinement software such as RELION or cryoSPARC is also suitable. Do
not use an already masked or sharpened post-processed map as the default input.

### Deep-learning models

Select a model with `-p` or `--processingType`:

- `tightTarget` is the default. It was trained with tightly masked targets and was used for the statistics reported
  in the publication.
- `wideTarget` uses less tightly masked targets and may preserve regions removed by the other models.
- `highRes` was trained on a subset of maps with resolutions better than 4 Å and fewer empty cubes. It generally
  produces stronger enhancement but may also produce more noise.

Try more than one model when possible. `highRes` is a good starting point for maps better than 4 Å, `tightTarget`
for lower-resolution or heterogeneous maps, and `wideTarget` when the other results appear over-masked.

### Normalization

Correct input normalization is essential. By default, DeepEMhancer estimates noise from a spherical shell and
normalizes it to mean 0 and standard deviation 0.1. Automatic estimation can fail for unusual geometries such as
hollow particles or fibers.

You can provide measured noise statistics with `--noiseStats NOISE_MEAN NOISE_STD`. One way to obtain them is to
crop a noise-only region in UCSF Chimera and calculate its mean and standard deviation.

For masked inputs, supply a binary mask with `-m` or `--binaryMask`. This selects the model designed for masked-map
normalization. Prefer an unprocessed, unmasked input whenever one is available.

### Batch size and GPUs

DeepEMhancer divides a map into cubes and sends batches of cubes to the selected GPU. Reduce `--batch_size` if GPU
memory is exhausted; increase it when GPU utilization is low and sufficient memory is available. The default is 1 so
that inference works on low-memory GPUs. For example, a reasonable starting point for an 8 GB GPU is `--batch_size 6`.

Use `-g 0` for the first GPU, `-g 0,1` for multiple GPUs, `-g all` for every detected GPU, or `-g -1` for CPU-only
inference. GPU indices are zero-based. Setting `TF_FORCE_GPU_ALLOW_GROWTH=true` can help TensorFlow avoid reserving
all GPU memory at startup.

## Examples

Download the published models:

```bash
deepemhancer --download
```

Process two half-maps with the default `tightTarget` model:

```bash
deepemhancer -i half1.mrc -i2 half2.mrc -o output.mrc
```

Process a full map with the default model:

```bash
deepemhancer -i input.mrc -o output.mrc
```

Use the `wideTarget` or `highRes` model:

```bash
deepemhancer -p wideTarget -i input.mrc -o output-wide.mrc
deepemhancer -p highRes -i input.mrc -o output-highres.mrc
```

Use models stored in a custom directory:

```bash
deepemhancer -p highRes --deepLearningModelPath /path/to/models -i input.mrc -o output.mrc
```

Provide measured noise statistics:

```bash
deepemhancer -p highRes -i input.mrc -o output.mrc --noiseStats 0.12 0.03
```

Normalize a masked input with a binary mask:

```bash
deepemhancer -i input.mrc -m mask.mrc -o output.mrc
```

## Troubleshooting

### TensorFlow does not detect the GPU

Check detection with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If this prints an empty list or reports that the driver is insufficient, update the NVIDIA driver to a version
supported by TensorFlow 2.21 and reinstall DeepEMhancer in a clean environment. See TensorFlow's current
[pip installation requirements](https://www.tensorflow.org/install/pip).

### GPU memory or cuDNN initialization errors

Errors such as `CUDNN_STATUS_INTERNAL_ERROR`, `Failed to get convolution algorithm`, or an out-of-memory message
usually indicate insufficient free GPU memory or conflicting CUDA libraries. First try a smaller batch and dynamic
GPU memory allocation:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true deepemhancer -b 1 -i input.mrc -o output.mrc
```

If the problem persists, reinstall in a clean environment. The `tensorflow[and-cuda]` dependency supplies a matched
CUDA runtime and cuDNN set; avoid mixing these with separately installed libraries on `LD_LIBRARY_PATH`.

### Multi-GPU execution fails

Some map dimensions and batch sizes can fail during multi-GPU execution. Retry on one GPU with batch size 1:

```bash
deepemhancer -g 0 -b 1 -i input.mrc -o output.mrc
```

### TensorFlow prints oneDNN or Abseil startup messages

These messages are informational during actual inference and do not indicate a DeepEMhancer failure. Commands that
do not run inference, including `deepemhancer --help` and `deepemhancer --version`, do not initialize TensorFlow.

### Model download fails

Retry `deepemhancer --download`. If the automated download remains unavailable, download
[the TensorFlow 2 model archive from Zenodo](https://zenodo.org/records/7432763) and extract its
`production_checkpoints` directory into `~/.local/share/deepEMhancerModels/`.
