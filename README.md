# Deep cryo-EM Map Enhancer (DeepEMhancer)
**DeepEMhancer** is a python package designed to perform post-processing of
cryo-EM maps as described in "<a href=https://doi.org/10.1038/s42003-021-02399-1 >DeepEMhancer:
a deep learning solution for cryo-EM volume post-processing</a>", by Sanchez-Garcia et al, 2021.<br>>
DeepEMhancer is a deep learning model trained on pairs of experimental volumes and atomic model-corrected volumes that is 
able to obtain post-processed maps using as input raw volumes, preferably half maps. Please notice that post-translational 
modifications and ligands were not included in the traning set and consequently, results for these features could be inaccurate.<br>

Simply speaking, DeepEMhancer performs a non-linear post-processing of cryo-EM maps that produces two main effects:
1) Local sharpening-like post-processing.
2) Automatic masking/denoising of cryo-EM maps.


## Table of Contents  
- [INSTALLATION](#installation)  
- [USAGE GUIDE](#usage-guide)  
- [EXAMPLES](#examples)  
- [TROUBLESHOOTING](#Troubleshooting)  
<br><br>
To get a complete description of usage, execute
`deepemhancer -h`
## INSTALLATION:

- [Requirements](#requirements)
- [Recommended pip installation](#recommended-pip-installation)
- [Reproducible Conda environment](#reproducible-conda-environment)
- [Alternative installation for old versions](#alternative-installation-for-old-versions)
- [Install from a source checkout](#install-from-a-source-checkout)

#### Requirements
DeepEMhancer has been tested on Linux systems (including WSL2) with Python 3.10-3.13 and TensorFlow 2.21.
The supplied environment installs TensorFlow's official `and-cuda` extra, which provides the compatible CUDA 12
runtime libraries. A sufficiently recent NVIDIA driver is still required. CPU-only execution remains available but
is considerably slower.

### Recommended pip installation

Create and activate a Python 3.10-3.13 virtual environment, then install DeepEMhancer directly
from GitHub. GPU support and its CUDA runtime libraries are installed by default.

```
python3 -m venv deepEMhancer_env
source deepEMhancer_env/bin/activate
python -m pip install "git+https://github.com/rsanchezgarc/deepEMhancer.git"
deepemhancer --download
deepemhancer -h
```

For a CPU-only installation, set `DEEPEMHANCER_CPU_ONLY` while installing:

```
DEEPEMHANCER_CPU_ONLY=1 python -m pip install "git+https://github.com/rsanchezgarc/deepEMhancer.git"
```

### Reproducible Conda environment

The pinned environment is useful for reproducing the dependency versions tested for this release:

```
git clone https://github.com/rsanchezgarc/deepEMhancer.git
cd deepEMhancer
conda env create -f deepEMhancer_env.yml -n deepEMhancer_env
conda activate deepEMhancer_env
python -m pip install . --no-deps
deepemhancer --download
```

### Alternative installation for old versions
You can install from the repository the older deepEMhancer version that works well in old GPUs. The process is identical
to the option number 1, "Install from source option", except that you need to use the tag 0.14
<br><br>Steps:
1) Clone the tag "0.14" of the repository repository and cd inside
```
git clone --depth 1 --branch "0.14"  https://github.com/rsanchezgarc/deepEMhancer
cd deepEMhancer
```
The following steps are the same as in option 1.<br>

2) Create a conda environment with the required dependencies
```
conda env create -f deepEMhancer_env.yml  -n deepEMhancer_env
```
3) Activate the environment. You always need to activate the environment before executing deepEMhancer
```
conda activate deepEMhancer_env
```
4) Install deepEMhancer
```
python -m pip install . --no-deps
```
5) Download our deep learning models
```
deepemhancer --download
```
6) Ready! Do not forget to activate the environment for future usages. For a complete help use:
```
deepemhancer -h
```
7) Optionally, you can remove the folder, since deepemhancer will be available anywhere once you activate the environment

If your NVIDIA drivers are too old to work with CUDA > 10.0, you can still install an even older version of Tensorflow. 
This option is only recommended for people with old NVIDIA drivers that are still able to work with
CUDA 10.0.

The steps for this option are exactly the same as above with the exception
of step 2. Thus, instead of using  "deepEMhancer_env.yml" when creating the environment,
```
conda env create -f deepEMhancer_env.yml  -n deepEMhancer_env
```
"alternative_installation/deepEMhancer_cud10.0.env.yml" should be used.

```
conda env create -f alternative_installation/deepEMhancer_cud10.0.env.yml  -n deepEMhancer_env
```
 
It has been reported that some problems with cudnn may occur when using this installation option. Please,
see [TROUBLESHOOTING](#Troubleshooting) section 2 for a proposed solution. 


### Install from a source checkout

For development or to install an unreleased branch:

```
git clone https://github.com/rsanchezgarc/deepEMhancer.git
cd deepEMhancer
python -m pip install .
deepemhancer --download
```

Check that TensorFlow detects the GPU:

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The command should print at least one `PhysicalDevice` with `device_type='GPU'`. If it prints an empty list,
check the NVIDIA driver and reinstall in a clean environment; do not mix the bundled runtime with separately
installed CUDA or cuDNN packages.

## Usage guide:
##### About the input

DeepEMhancer was trained using half-maps. Thus, as input, both half-maps are the preferred option (`deepemhancer -i half1.mrc -i2 half2.mrc`).<br> 
Full maps obtained from refinement process (RELION auto-refine, cryoSPARC heterogenus refinement...) are equally valid.<br>
However, deepEMhancer will not work correctly if post-processed (masked, sharpened...) maps are provided as input 
(e.g. RELION postprocessing maps).

##### About the deep learning models (-p option)
We provide 3 different deep learning models. The default one is the tightTarget model, that was trained using
tightly masked volumes. This is the default option and all the statistics reported in the publication were obtained 
using this model. Additionally, we provide a wideTarget model that was trained using less tightly masked maps. Finally,
we have also trained a model (highRes) using a subset of the maps with resolutions <4 Å and fewer empty cubes.<br>
We recommend our users to try the different options and choose the one that looks nicer to them. As a guidance, 
we suggest to employ the highRes model for maps with overall resolution better than 4 Å and a moderate amount of bad
resolution regions. HighRes solutions tend to be noisier than others, but also more enhanced. 
If the overall resolution is worse, or the number of low resolution regions is high, the tightTarget
model should do a good job. For cases in which both tightTarget and highRes produce too tightly masked solutions, possibly removing
some parts of the protein as if they were noise, we recommend to employ the wideTarget model.

##### About the normaliztion
One of the key aspects to succesfully employ DeepEMhancer is the normalization of the input volumes.
The default normalization mode, mode 1, normalizes the data such that the statistics of the noise regions
are forced to adopt a mean value of 0 and a standard deviation of 0.1.<br>
If no flag is provided, deepEMhancer will try to automatically determine a spherical shell of noise from
which the statistics of the noise are estimated. This automatic normalization tends to work well, although it
may fail in some cases. For example, hollow proteins or fiber proteins could cause problems.<br>
Alternatively, the user can manually determine the statistics of the noise and provide them to the program using
the flag `--noiseStats mean_noise std_noise`. One easy way to determine the noise statistics is to employ UCSF Chimera
to crop a noise-only region of the map (`Volume Viewer>Features>Region bounds`) and then compute the statistics (`Volume Viewer>Tools>Volume Mean, SD, RSD`).
<br>
Finally, as an alternative normalization, mode 2 normalizes the input using a binary mask (1 protein, 0 not protein).
This option was introduced to deal with masked maps (which are not suitable for default DeepEMhancer) and is not recommended
when it is possible to employ normalization mode 1.

##### About the batch size
DeepEMhancer processes input maps by chunking them into smaller cubes that are sent to GPUs. Batch size parameter represent
the number of smaller cubes that are simultaneously processed by the GPUs. A typical value for an 8 GB GPU could be<br> 
```--batch_size 6```. If OUT OF MEMORY error happens, try to lower batch_size, and if low GPU usage is observed (via nvidia-smi), try
to increase it. Setting the environmental variable `TF_FORCE_GPU_ALLOW_GROWTH='true'` prior execution could also help to fix some GPU memory errors. When using multiple GPUs, for certain box sizes, there might happen a reported bug affecting the batch_size, please see [TROUBLESHOOTING](#Troubleshooting) error 3.



## Examples
- Download deep learning models
```
deepemhancer --download
```
- Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using default  deep model (tightTarget)
```
deepemhancer  -i path/to/inputVol.mrc -o  path/to/outputVol.mrc
```
- Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using softer deep model (wideTarget)
```
deepemhancer -p wideTarget -i path/to/inputVol.mrc -o  path/to/outputVol.mrc
```
- Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using high resolution deep model
```
deepemhancer -p highRes -i path/to/inputVol.mrc -o  path/to/outputVol.mrc
```
- Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using high resolution deep learning model located in path/to/deep/learningModel
```
deepemhancer -p highRes  --deepLearningModelDir path/to/deep/learningModel -i path/to/inputVol.mrc -o  path/to/outputVol.mrc
```
- Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using high resolution  deep model and providing normalization information (mean
    and standard deviation of the noise)
```    
deepemhancer -p highRes -i path/to/inputVol.mrc -o  path/to/outputVol.mrc --noiseStats 0.12 0.03
```


## TROUBLESHOOTING

1.  
- Error: 
```
tensorflow.python.framework.errors_impl.InternalError: cudaGetDevice() failed. Status: CUDA driver version is insufficient for CUDA runtime version
```
- Explanation: The drivers of your NVIDA GPU are too old.<br>

- Solution: Update the NVIDIA driver to a version supported by TensorFlow 2.21, then recreate the environment from
`deepEMhancer_env.yml`. See TensorFlow's current [pip installation requirements](https://www.tensorflow.org/install/pip).

2.
- Error:
```
  (1) Unknown: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
	 [[{{node conv3d_1/convolution}}]]
0 successful operations.
0 derived errors ignored.
```
or

```
E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
```
- Explanation: This is a reported issue for Tensorflow, and many times occurs when getting out of GPU memory. 
In other cases is related with an incompatibility between CUDA and cudnn versions.<br>

- Solution: 
  - If it is caused by memory constraints, set dynamic GPU allocation using the environment variable 
TF_FORCE_GPU_ALLOW_GROWTH='true'. E.g. 
```TF_FORCE_GPU_ALLOW_GROWTH='true' deepemhancer -i ~/tmp/useCase/EMD-0193.mrc -o ~/tmp/outVolDeepEMhancer/out.mrc
```

  - If it is caused by incompatible CUDA libraries, recreate the environment from `deepEMhancer_env.yml`. The
    `tensorflow[and-cuda]` dependency installs a matched runtime and cuDNN set.



3. 
- Error:
```
F ./tensorflow/core/kernels/conv_2d_gpu.h:935] Non-OK-status: CudaLaunchKernel( SwapDimension1And2InTensor3UsingTiles<T, kNumThreads, kTileSize, kTileSize, conjugate>, total_tiles_count, kNumThreads, 0, d.stream(), input, input_dims, output) status: Internal: invalid configuration argument

Aborted (core dumped)
```
- Explanation: This is a reported issue for Tensorflow when using multiple GPUS and the number of subcubes or the batch size is not divisible by the number of GPUs
- Solution: Use only one GPU (`-g 1`) and/or batch size 1 (`-b 1`)

4. 
- Error:
It is not possible to download the models. 
- Explanation:
Sometimes our server may not be reachable due to network problems.
- Solution:
Download them from https://zenodo.org/record/7432763
