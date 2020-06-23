# Deep cryo-EM Map Enhancer (DeepEMhancer)

**DeepEMhancer** is a python package designed to perform post-processing of
cryo-EM maps as described in "<a href=https://doi.org/10.1101/2020.06.12.148296 >DeepEMhancer: a deep learning solution for cryo-EM volume post-processing</a>", by Sanchez-Garcia et al, 2020.<br>
Simply speaking, DeepEMhancer performs a non-linear post-processing of cryo-EM maps that produces two main effects:
1) Local sharpening-like post-processing
2) Automatic masking/denoising of cryo-EM maps.


To get a complete description of usage, execute

`deepemhancer -h`


##### Example

`deepemhancer  -i path/to/inputVol -o path/to/outputVol`


## INSTALLATION:

### Install from source option:
The best option to keep you updated. <br>
Requires anaconda/miniconda and makes use of Nvidia GPU.
Anaconda/miniconda can be obtained from <ref>https://www.anaconda.com/products/individual</ref>
<br>Steps:
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
python setup.py develop --no-deps
```

5) Download our deep learning models

```
deepemhancer --donwload

```

6) Ready! Do not forget to activate the environment for future usages. For a complete help use:

```
deepemhancer -h

```

7) Optionally, you can remove the folder, since deepemhancer will be available anywhere once you activate the environment


### Anaconda cloud:
Coming soon!




### Examples


#### Donwload deep learning model
```
deepemhancer --download
```

#### Post-process input volume path/to/inputVol.mrc and save it at path/to/outputVol.mrc
```
deepemhancer  -i path/to/inputVol.mrc -o  path/to/outputVol.mrc
```
