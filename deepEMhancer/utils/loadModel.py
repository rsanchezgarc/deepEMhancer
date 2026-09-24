import os
import tempfile
from contextlib import contextmanager
import tensorflow as tf
import h5py
from .ioUtils import loadVolIfFnameOrIgnoreIfMatrix


@contextmanager
def _keras_compatible_model_path(checkpoint_fname):
  """Expose legacy ``.hd5`` checkpoints with an extension Keras 3 accepts."""
  if checkpoint_fname.lower().endswith(".hd5"):
    with tempfile.TemporaryDirectory(prefix="deepemhancer-model-") as tmp_dir:
      compatible_path = os.path.join(tmp_dir, "checkpoint.h5")
      os.symlink(os.path.abspath(checkpoint_fname), compatible_path)
      yield compatible_path
  else:
    yield checkpoint_fname


def load_model(checkpoint_fname, custom_objects=None, lastLayerToFreeze=None, resetWeights=False, nGpus=1):
  if custom_objects is None:
    __, codes = retrieveParamsFromHd5(checkpoint_fname,  [], ['code/custom_objects'])
    if codes is None:
      from devel_code.trainNet.defaultNet import getCustomObjects
      print("WARNING, no custom obtects in model, using default")
      custom_objects = getCustomObjects()
    else:
      custom_objects= codes["custom_objects"]


  from tensorflow.keras.models import load_model as keras_load_model

  def load_checkpoint():
    # These checkpoints are used for inference only. Avoid deserializing their
    # old optimizer/loss configuration, which is not portable to Keras 3.
    with _keras_compatible_model_path(checkpoint_fname) as compatible_path:
      return keras_load_model(compatible_path, custom_objects=custom_objects, compile=False)
      
  if nGpus>1:
    devices_names = list(map(lambda x:":".join( x.name.split(":")[-2:]), tf.config.list_physical_devices('GPU')))
    mirrored_strategy = tf.distribute.MirroredStrategy(devices= devices_names )
    with mirrored_strategy.scope():
      model = load_checkpoint()
  else:
      model = load_checkpoint()

  if lastLayerToFreeze is not None:
    layerFound= False
    for layer in model.layers:
      layer.trainable=False
      if layer.name.startswith(lastLayerToFreeze):
        layerFound=True
        break
    assert layerFound is True, "Error, %s not found in the model"%lastLayerToFreeze

  if resetWeights:
    print("Model reset")
    for i, layer in enumerate(model.layers):
      initializers = []
      if hasattr(layer, 'kernel_initializer'):
        initializers += [ lambda : layer.kernel_initializer(shape=model.layers[i].get_weights()[0].shape) ]
      if hasattr(layer, 'bias_initializer') and layer.bias is not None:
        initializers += [ lambda : layer.bias_initializer(shape=model.layers[i].get_weights()[1].shape) ]
      if len(initializers)>0:
        model.layers[i].set_weights( [f() for f in initializers ]  )
  return model


def getInputCubeSize(model):
  input_shape = model.input_shape
  if isinstance(input_shape, list):
    input_shape = input_shape[0]
  return input_shape[1]


def retrieveParamsFromHd5(fname, paramsList, codeList):
  '''
  Example:

   retrieveParamsFromHd5(kerasCheckpointFname, ['configParams/*', 'configParams/NNET_INPUT_SIZE'], ['code/normFun', 'code/*'])

  :param fname:
  :param paramsList:
  :param codeList:
  :return:

  '''
  args= {}
  codes={}
  try:
    with h5py.File(fname,'r') as h5File:
#      print(h5File.keys())
      for paramName in paramsList:
        if paramName.endswith("*"):
          paramName=paramName.replace("/*", "")
          for putativeKey in h5File[paramName]:
            param=h5File[paramName+'/'+putativeKey][0]
            args[putativeKey]= param
        else:
          args[os.path.basename(paramName)] = h5File[paramName][0]

      env= globals()
      for codeName in codeList:
        if codeName.endswith("*"):
          codeName=codeName.replace("/*", "")
          for putativeKey in h5File[codeName]:
            codeStr=h5File[codeName+'/'+putativeKey][0]
            exec(codeStr, env)  # normFun is in the code
            codes[putativeKey] =(env.get(putativeKey))
        else:
          codeStr =  h5File[codeName][0]
          exec(codeStr, env)  # normFun is in the code
          codes[os.path.basename(codeName)]= (env.get(os.path.basename(codeName)))
    return args, codes
  except KeyError:
    print("Error loading config from hd5")
    return None, None

def loadNormalizationFunFromModel(kerasCheckpointFname, binary_mask=None, noise_stats=None):
  args, codes = retrieveParamsFromHd5(kerasCheckpointFname, [], ['code/normFun' ])
  if codes is None:
    return None
  if binary_mask is not None:
    binary_mask, __ = loadVolIfFnameOrIgnoreIfMatrix(binary_mask, normalize=None)
    normalizationFunction = lambda x: codes['normFun'](x, binary_mask)
  else:
    normalizationFunction = codes['normFun']

  if noise_stats is not None:
    assert normalizationFunction.__name__=="inputNormalization_3"
    return lambda y: normalizationFunction(y, noise_stats)

  return  normalizationFunction


def loadChunkConfigFromModel(kerasCheckpointFname):
  args, codes = retrieveParamsFromHd5(kerasCheckpointFname, ['configParams/NNET_INPUT_STRIDE', 'configParams/NNET_INPUT_SIZE'], [])
  if args is None:
    return None
  else:
    return args

if __name__ == "__main__":
  from config import DEFAULT_MODEL_DIR
  print( DEFAULT_MODEL_DIR )
  fname = os.path.join(DEFAULT_MODEL_DIR, "deepEMhancer_highRes.hd5")
  conf=loadChunkConfigFromModel( fname); print(conf)
  conf = retrieveParamsFromHd5(fname, [], ["code/*"]); print(conf)
  #conf=loadChunkConfigFromModel("/home/ruben/Tesis/cryoEM_cosas/auto3dMask/data/nnetResults/vt2_m_28_checkpoints/bestCheckpoint_locscale_masked.hd5"); print(conf)
