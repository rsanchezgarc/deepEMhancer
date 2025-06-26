import os

#These are the parameters that were used to train the network
RESIZE_VOL_TO= 1.
NNET_INPUT_SIZE=64
NNET_INPUT_STRIDE= NNET_INPUT_SIZE//4


import tensorflow as tf
if int(tf.__version__.split(".")[0]) > 1:
  DOWNLOAD_MODEL_URL = 'https://zenodo.org/record/7432763/files/deepEMhancerModels_tf2.zip'
  MODEL_DOWNLOAD_EXPECTED_SIZE=721852*1024
else:
  DOWNLOAD_MODEL_URL = 'https://zenodo.org/record/7432763/files/deepEMhancerModels_tf1.zip'
  MODEL_DOWNLOAD_EXPECTED_SIZE=721840*1024

DEFAULT_MODEL_DIR = os.path.expanduser("~/.local/share/deepEMhancerModels/production_checkpoints")


NNET_NJOBs= 8
BATCH_SIZE=8

MAX_VAL_AFTER_NORMALIZATION=200



