import os

import setuptools
from setuptools import setup

def version():
  initPath = os.path.abspath(os.path.join(__file__, "..", "deepEMhancer", "__init__.py"))
  with open(initPath) as f:
    version = f.read().strip().split('"')[-2]
  return version
      
def readme():
  readmePath = os.path.abspath(os.path.join(__file__, "..", "README.md"))
  try:
    with open(readmePath) as f:
      return f.read()
  except UnicodeDecodeError:
    try:
      with open(readmePath, 'r', encoding='utf-8') as f:
        return f.read()
    except Exception as e:
      return "Description not available due to unexpected error: "+str(e)


install_requires = [
  'h5py>=3.11,<3.15',
  'mrcfile>=1.5,<1.6',
  'numpy>=1.26,<2.3',
  'requests>=2.32,<3',
  'scikit-image>=0.24,<0.27',
  'scipy>=1.13,<1.18',
  'tqdm>=4.67,<5',
]

if os.environ.get('DEEPEMHANCER_CPU_ONLY'):
  install_requires.append('tensorflow==2.21.*')
else:
  install_requires.append('tensorflow[and-cuda]==2.21.*')

setup(name='deepEMhancer',
      version=version(),
      description='Deep learning for cryo-EM maps post-processing',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cryo-EM deep learning',
      url='https://github.com/rsanchezgarc/deepEMhancer',
      author='Ruben Sanchez-Garcia',
      author_email='rsanchez@cnb.csic.es',
      license='Apache 2.0',
      packages=setuptools.find_packages(),
      install_requires=install_requires,
      python_requires='>=3.10,<3.14',
      dependency_links=[],
      entry_points={
        'console_scripts': ['deepemhancer=deepEMhancer.exeDeepEMhancer:commanLineFun'],
      },
      include_package_data=True,
      zip_safe=False)
