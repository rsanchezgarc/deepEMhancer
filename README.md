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


### Install from PyPI.
Simple but does not work with GPU, which makes the program quite slow
<br><br>Steps:

1) Optional but recommended. Create a fresh environment (e.g conda environment)

```
conda create -n deepEMhancer_env python=3.6
```

2) If using environments:<br>
Activate the environment. You always need to activate the environment before executing deepEMhancer if it was installed in an environment

```
conda activate deepEMhancer_env
```

4) Install deepEMhancer

```
python -m pip install deepEMhancer
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
Coming soon!


## Examples


- Donwload deep learning models
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