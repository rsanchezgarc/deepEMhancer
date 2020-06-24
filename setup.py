import sys, os

import setuptools
from setuptools import setup

VERSION="0.1"

def readme():
  readmePath = os.path.abspath(os.path.join(__file__, "..", "README.md"))
  with open(readmePath) as f:
    return f.read()


if "gpu" in sys.argv:
  tfTarget='-gpu==1.14.*'
else:
  tfTarget='==1.14.*'

install_requires = [
  'scikit-image==0.15.*',
  'scipy==1.3.1',
  'joblib==0.13.*',
  'tensorflow%s'%tfTarget,
  'keras==2.2.*',
  'pandas==0.25.*',
  'mrcfile==1.1.2',
  'requests==2.22.*',
  'tqdm==4.42'

]

dependency_links=['https://github.com/keras-team/keras-contrib/tarball/master#egg=3fc5ef709e061416f4bc8a92ca3750c824b5d2b0']

setup(name='deepEMhancer',
      version=VERSION,
      description='Deep learning for cryo-EM maps post-processing',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cryo-EM deep learning',
      url='https://github.com/rsanchezgarc/deepEMhancer',
      author='Ruben Sanchez-Garcia',
      author_email='rsanchez@cnb.csic.es',
      license='Apache 2.0',
      #packages=['deepEMhancer'],
      packages=setuptools.find_packages(),
      install_requires=install_requires,
      dependency_links=dependency_links,
      entry_points={
        #'console_scripts': ['deepemhancer=deepEMhancer.deepEMhancer:commanLineFun'],
        'console_scripts': ['deepemhancer=deepEMhancer.exeDeepEMhancer:commanLineFun'],
      },
      include_package_data=True,
      zip_safe=False)

