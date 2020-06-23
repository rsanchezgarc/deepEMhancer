import sys, os
from setuptools import setup

VERSION="0.01"

def readme():
  readmePath = os.path.abspath(os.path.join(__file__, "..", "README.md"))
  with open(readmePath) as f:
    return f.read()


if "gpu" in sys.argv:
  tfTarget='-gpu==1.14.0'
else:
  tfTarget='==1.14.0'

install_requires = [
  'scikit-image==0.15.0',
  'scipy==1.3.1',
  'joblib==0.13.2',
  'tensorflow%s'%tfTarget,
  'keras==2.2.4',
  'pandas==0.25.1',
  'mrcfile==1.1.2',
  'requests==2.22.0',
]

dependency_links=['https://github.com/keras-team/keras-contrib/tarball/master#egg=3fc5ef709e061416f4bc8a92ca3750c824b5d2b0']

setup(name='deep_em_vol_postprocesser',
      version=VERSION,
      description='Deep learning for cryo-EM volume postprocessing',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cryo-EM deep learning',
      url='https://github.com/rsanchezgarc/deep_vol_processer_em',
      author='Ruben Sanchez-Garcia',
      author_email='rsanchez@cnb.csic.es',
      license='Apache 2.0',
      packages=['deepEMhancer'],
      install_requires=install_requires,
      dependency_links=dependency_links,
      entry_points={
        'console_scripts': ['deepemhancer=deepEMhancer.deepEMhancer:commanLineFun'],
      },
      include_package_data=True,
      zip_safe=False)

