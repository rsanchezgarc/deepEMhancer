# Deep cryo-EM Map Enhancer (DeepEMhancer)
**DeepEMhancer** is a python package designed to perform post-processing of
cryo-EM maps as described in "<a href=https://doi.org/10.1101/2020.06.12.148296 >DeepEMhancer: a deep learning solution for cryo-EM volume post-processing</a>", by Sanchez-Garcia et al, 2020.<br>
Simply speaking, DeepEMhancer performs a non-linear post-processing of cryo-EM maps that produces two main effects:
1) Local sharpening-like post-processing
2) Automatic masking/denoising of cryo-EM maps.
## Table of Contents  
- [INSTALLATION](#installation)  
- [USAGE GUIDE](#usage-guide)  
- [EXAMPLES](#examples)  
<br><br>
To get a complete description of usage, execute
`deepemhancer -h`
## INSTALLATION:
### Install from source option:
The best option to keep you updated. <br>
Requires anaconda/miniconda and makes use of Nvidia GPU(s).
Anaconda/miniconda can be obtained from <ref>https://www.anaconda.com/products/individual</ref>
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
### Anaconda cloud:
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
## Usage guide:
##### About the input
DeepEMhancer was trained using half-maps. Thus, as input, half-maps are the preferred option. 
Full maps obtained from refinement process (Relion auto-refine, cryoSPARC heterogenus refinement...) are equally valid.
However, deepEMhancer will not work correctly if post-processed (masked, sharpened...) maps are provided as input 
(e.g. Relion postprocessing maps).
##### About the deep learning models (-p option)
We provide 3 different deep learning models. The default one is the tightTarget model, that was trained using
tightly masked volumes. This is the default option and all the statistics reported in the publication were obtained 
using this model. Additionally, we provide a wideTarget model that was trained using less tightly masked maps. Finally,
we have also trained a model (highRes) using a subset of the maps with resolutions <4 Å.<br>
We recommend our users to try the different options and choose the one that looks nicer to them. As a guidance, 
we recommend to employ the highRes model for maps with overall resolution better than 4 Å and a moderate amount of bad
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
