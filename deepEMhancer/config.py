import os

#These are the parameters that were used to train the network
RESIZE_VOL_TO= 1.
NNET_INPUT_SIZE=64
NNET_INPUT_STRIDE= NNET_INPUT_SIZE//4

DOWNLOAD_MODEL_URL = 'http://biocomp.cnb.csic.es/deepEMhancer/deepEMhancerModels.zip'
MODEL_DOWNLOAD_EXPECTED_SIZE=739163136

DEFAULT_MODEL_DIR = os.path.expanduser("~/.local/share/deepEMhancerModels/production_checkpoints")


NNET_NJOBs= 8
BATCH_SIZE=8

MAX_VAL_AFTER_NORMALIZATION=200


def GET_CUSTOM_OBJECTS():
  from keras_contrib.layers.normalization import groupnormalization
  return {  "GroupNormalization":groupnormalization.GroupNormalization }

