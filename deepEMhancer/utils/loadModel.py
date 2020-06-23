import os

import h5py
from .ioUtils import loadVolIfFnameOrIgnoreIfMatrix


def load_model(checkpoint_fname, custom_objects=None, lastLayerToFreeze=None, resetWeights=False, nGpus=1):
  if custom_objects is None:
    __, codes = retrieveParamsFromHd5(checkpoint_fname,  [], ['code/custom_objects'])
    if codes is None:
      from devel_code.trainNet.defaultNet import getCustomObjects
      print("WARNING, no custom obtects in model, using default")
      custom_objects = getCustomObjects()
    else:
      custom_objects= codes["custom_objects"]


  from keras.models import load_model
  model= load_model(checkpoint_fname, custom_objects=custom_objects )
  if nGpus>1:
    from keras.utils import multi_gpu_model
    model= multi_gpu_model(model, gpus=nGpus)

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
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers:
      if hasattr(layer, 'kernel_initializer'):
        layer.kernel.initializer.run(session=session)
      if hasattr(layer, 'bias_initializer') and layer.bias is not None:
        layer.bias.initializer.run(session=session)
  return model


def getInputCubeSize(model):
  try:
    return model.layers[0].output_shape[0][1]
  except TypeError:
    return model.layers[0].output_shape[1]


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
      # print(h5File.keys())
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


#conf=loadChunkConfigFromModel("/home/ruben/Tesis/cryoEM_cosas/auto3dMask/data/nnetResults/checkpoints/bestCheckpoint_locscale.hd5"); print(conf)
# conf=loadChunkConfigFromModel("/home/ruben/Tesis/cryoEM_cosas/auto3dMask/data/nnetResults/vt2_m_28_checkpoints/bestCheckpoint_locscale_masked.hd5"); print(conf)