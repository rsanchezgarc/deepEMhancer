# Deep cryo-EM Map Enhancer (DeepEMhancer)
**DeepEMhancer** is a python package designed to perform post-processing of
cryo-EM maps as described in "<a href=https://doi.org/10.1101/2020.06.12.148296 >DeepEMhancer: a deep learning solution for cryo-EM volume post-processing</a>", by Sanchez-Garcia et al, 2020.<br>
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
- [Install from source option](#install-from-source-option)
- [Install from Anaconda cloud](#install-from-anaconda-cloud)
- [Alternative installation for CUDA 10.0 ](#alternative-installation-for-CUDA-10.0-compatible-systems)
- [No conda installation](#no-conda-installation)

#### Requirements
DeepEMhancer has been tested on Linux systems.
It employs Tensorflow version 1.14 that requires CUDA 10. Our installation recipe will
automatically install, among other packages, Tensorflow and CUDA 10.1, so you will need
NVIDA GPU drivers  >= 418.39. If your drivers are not compatible and you cannot update them,
you can try to compile tensorflow-gpu=1.14 using your library settings instead of installing it using conda.
For those having old drivers but still compatible with CUDA 10.0, see "Alternative installation 
for CUDA 10.0 compatible systems" or "No conda installation".

### Install from source option:
The best option to keep you updated. <br>
Requires anaconda/miniconda, that can be obtained from <ref>https://www.anaconda.com/products/individual</ref>
<br><br>Steps:
1) Clone this repository and cd inside
```
git clone https://github.com/rsanchezgarc/deepEMhancer
cd deepEMhancer
```
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

### Install from Anaconda cloud:
Requires anaconda/miniconda, that can be obtained from https://www.anaconda.com/products/individual 

1) Create a fresh conda environment
```
conda create -n deepEMhancer_env python=3.6
```
2) Activate the environment. You always need to activate the environment before executing deepEMhancer
```
conda activate deepEMhancer_env
```
4) Install deepEMhancer
```
conda install deepEMhancer -c rsanchez1369 -c anaconda -c conda-forge
```
5) Download our deep learning models
```
deepemhancer --download
```
6) Ready! Do not forget to activate the environment for future usages. For a complete help use:
```
deepemhancer -h
```

### Alternative installation for CUDA 10.0 compatible systems

This option is only recommended for people with old NVIDIA drivers that are still able to work with
CUDA 10.0.

The steps for this option are exactly the same that for option "Install from source option" with the exception
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


### No conda installation
Only works for python3. Virtualenv is recommended to isolate packages.

1) Clone this repository and cd inside
```
git clone https://github.com/rsanchezgarc/deepEMhancer
cd deepEMhancer
```

1.1. Optionally, create a virtual environment and activate it
```
pip install virtualenv
virtualenv --system-site-packages -p python3 ./deepEMhancer_env
source ./deepEMhancer_env/bin/activate
```
2) Install deepEMhancer (using Tensorflow 1.14)
- For CPU only use (expect running times ~ 24h)
```
python -m pip install .
```
- With GPU support
  - Install CUDA 10.0 and cudnn >=7.6. Make sure that they are in the LD_LIBRARY_PATH
  - install python packages
```
DEEPEMHANCER_INSTALL_GPU=True pip install .
```
  - Check if GPUs are successfully detected.
```
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```
You will see errors like ```Could not dlopen library 'libcudart.so.10.0'; dlerror``` if CUDA and/or cudnn 
(libcudnn.so.7) are not correctly installed or detected. On the contrary, if you see the message
```I tensorflow/core/common_runtime/gpu/gpu_device.cc:1326] Created TensorFlow device```,
Tensorflow has been able to recognize the GPUs.
  
5) Download our deep learning models
```
deepemhancer --download
```
6) Ready! Do not forget to activate the environment, if used (step 1.1), for future usages. For a complete help use:
```
deepemhancer -h
```

## Usage guide:
##### About the input
DeepEMhancer was trained using half-maps. Thus, as input, both half-maps are the preferred option (deepemhancer -i half1.mrc -i2 half2.mrc).<br> 
Full maps obtained from refinement process (RELION auto-refine, cryoSPARC heterogenus refinement...) are equally valid.<br>
However, deepEMhancer will not work correctly if post-processed (masked, sharpened...) maps are provided as input 
(e.g. RELION postprocessing maps).
##### About the deep learning models (-p option)
We provide 3 different deep learning models. The default one is the tightTarget model, that was trained using
tightly masked volumes. This is the default option and all the statistics reported in the publication were obtained 
using this model. Additionally, we provide a wideTarget model that was trained using less tightly masked maps. Finally,
we have also trained a model (highRes) using a subset of the maps with resolutions <4 Å.<br>
We recommend our users to try the different options and choose the one that looks nicer to them. As a guidance, 
we suggest to employ the highRes model for maps with overall resolution better than 4 Å and a moderate amount of bad
resolution regions. If the overall resolution is worse, or the number of low resolution regions is high, the tightTarget
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
to increase it.



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

- Solution: Update your drivers to version >= 418.39. Alternatively, for driver versions 410.48 to 418.39 you could
try the "Alternative installation for Nvida-Driver 410", the "No conda installation" or install yourself Tensorflow 
using your CUDA setup. Although we have not tested it, deepEMhancer will probably also work with 
older Tensorflow versions that require CUDA 9, so they could be also considered.

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
  - If it is caused by memory constrains, set dynamic GPU allocation using the environment variable 
TF_FORCE_GPU_ALLOW_GROWTH='true'. E.g. 
```TF_FORCE_GPU_ALLOW_GROWTH='true' deepemhancer -i ~/tmp/useCase/EMD-0193.mrc -o ~/tmp/outVolDeepEMhancer/out.mrc```

  - If it is caused by incompatibility between CUDA and cudnn, you should try to reinstall it ensuring that
    CUDA and cudnn versions match and they are compatible with the Tensorflow version. We are using Tensorflow
    version 14, but we think that older versions, compatible with CUDA 9 could also work.