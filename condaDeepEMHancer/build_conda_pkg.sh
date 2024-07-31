#!/usr/bin/env bash
anaconda_path=`which python | rev | cut -f 3- -d"/" | rev` &&
rm $anaconda_path/conda-bld/linux-64/*

conda-build . -c anaconda -c conda-forge --python=3.10 --numpy=1.23 &&

#conda install --use-local deepEMhancer -c anaconda -c conda-forge && exit 0 &&
#conda install   deepEMhancer -c file://${HOME}/app/anaconda3/envs/deepEMhancer_env/conda-bld/ -c anaconda -c conda-forge

echo rsanchez1369 | anaconda upload $anaconda_path/conda-bld/linux-64/deepemhancer-*.bz2  # && conda build purge
#xxxXxx
