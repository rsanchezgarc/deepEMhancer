import os, sys, gc
import numpy as np
from numpy import pad
from tqdm import tqdm

from ..config import RESIZE_VOL_TO, BATCH_SIZE
from ..utils.dataUtils import getNewShapeForResize, resizeVol
from ..utils.gpuSelector import mask_CUDA_VISIBLE_DEVICES, resolveDesiredGpus
from ..utils.ioUtils import saveVol, loadVolIfFnameOrIgnoreIfMatrix
from ..utils.loadModel import load_model, getInputCubeSize, loadNormalizationFunFromModel, loadChunkConfigFromModel
from .utilsPostprocess import removeSmallCCs, morphologicalDilation

class AutoProcessVol(object):
  def __init__(self, model_fname, gpuIds="0", batch_size=BATCH_SIZE):
    '''

    :param model_fname: the filename where the keras model is saved
    :param gpuIds: the gpu id(s) to use. Comma separated string. Use -1 for cpu only
    :param batch_size:
    '''
    if isinstance(gpuIds, str):
      gpuIds, nGpus=resolveDesiredGpus(gpuIds)
    else:
      nGpus=1
    mask_CUDA_VISIBLE_DEVICES(gpuIds)

    batch_size= BATCH_SIZE if batch_size is None else batch_size
    self.batch_size = batch_size*nGpus
    print("loading model %s ..."%model_fname, end=" ")
    self.model_fname= model_fname
    self.model = load_model(model_fname, nGpus=nGpus )
    self.netInputSize= getInputCubeSize(self.model)
    chunkInfo=loadChunkConfigFromModel(model_fname)
    if chunkInfo is None:
      from ..config import NNET_INPUT_SIZE, NNET_INPUT_STRIDE
      self.nnet_input_size= NNET_INPUT_SIZE
      self.nnet_input_stride= NNET_INPUT_STRIDE
    else:
      self.nnet_input_size= chunkInfo["NNET_INPUT_SIZE"]
      self.nnet_input_stride= chunkInfo["NNET_INPUT_STRIDE"]

    print("DONE!")

  def _updateMask(self, coords_list, batch_y_pred, mask, weights):
    for coord, mask_chunk in zip(coords_list, batch_y_pred):
      if coord is None:
        continue
      (i, j, k) = coord
      mask_chunk= np.squeeze(mask_chunk)
      di, dj, dk = mask_chunk.shape
      mask[i:i + di, j:j + dj, k:k + dk] += mask_chunk
      weights[i:i + di, j:j + dj, k:k + dk] += 1.
    return None


  def _getNormalizationFunction(self, binary_mask=None, noise_stats=None):

    inputNorm_fun=loadNormalizationFunFromModel(self.model_fname, binary_mask, noise_stats)
    if binary_mask is not None:
      binary_mask, __ = loadVolIfFnameOrIgnoreIfMatrix(binary_mask, normalize=None)

    if inputNorm_fun is None:
      print("WARNING, normalization was not included in the model, guessing correct normalization")
      if binary_mask is None:
        if noise_stats is not None:
          inputNorm_fun= lambda x: ConfigManager.getInputNormalization(useMask=False)(x, noise_stats= noise_stats)
        else:
          inputNorm_fun = ConfigManager.getInputNormalization(useMask=False)
      else:
        inputNorm_fun = lambda x: ConfigManager.getInputNormalization(useMask=True)(x, binary_mask)

    return inputNorm_fun, binary_mask

  def _chunkInputVolForPrediction(self, vol, chunk_size=None, stride=None, binary_mask=None):
    '''
    Same as getVolChunks but for inference only
    :param vol:
    :param chunk_size:
    :param stride:
    :return:
    '''

    if stride is None:
      stride= self.nnet_input_stride

    if chunk_size is None:
      chunk_size= self.nnet_input_size

    # Vol must be divisible by NNET_INPUT_SIZE
    vol_shape = np.array(vol.shape)
    i_range= range(0, vol_shape[0] - (chunk_size - 1), stride)
    j_range= range(0, vol_shape[1] - (chunk_size - 1), stride)
    k_range= range(0, vol_shape[2] - (chunk_size - 1), stride)
    progressBar= tqdm(total=len(i_range)*len(j_range))
    count = 0
    for i in i_range:
      for j in j_range:
        for k in k_range:
          if binary_mask is not None:
            if np.count_nonzero(binary_mask[i:i + chunk_size, j:j + chunk_size, k:k + chunk_size])==0:
              continue
          cubeX = vol[i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
          yield cubeX, (i, j, k)
          count += 1
        progressBar.update()
        progressBar.refresh()
        while count % self.batch_size != 0:
            yield cubeX, None
            count += 1
    progressBar.close()

  def _padToDivisibleSize(self, x, fillWith0=False):
    stride =  self.nnet_input_stride
    height, width, depth = x.shape[:3]
    paddingHeight = (0, stride - height % stride)
    paddingWidth = (0, stride - width % stride)
    paddingDepth = (0, stride - width % stride)

    paddingValues = [paddingHeight, paddingWidth, paddingDepth]
    if fillWith0:
      x_pad = pad(x, paddingValues, mode="constant", constant_values=np.min(x))
    else:
      x_pad = pad(x, paddingValues, mode="wrap")
    return x_pad, paddingValues

  def _unPad(self, x, paddingValues):
    return x[paddingValues[0][0]:-paddingValues[0][1], paddingValues[1][0]:-paddingValues[1][1],
           paddingValues[2][0]:-paddingValues[2][1], ...]

  def predict(self, vol_fname_or_matrix, fname_out=None, threshold=None, binary_mask=None, noise_stats=None,
              voxel_size=None, apply_postprocess_cleaning=0.1, fname_prod=None, morph_dilation=None):
    '''

    :param vol_fname_or_matrix: The path containing an mrc file with the volume or a numpy array
    :param fname_out: The path where the post-processed volumne will be saved as an mrc file
    :param threshold: threshold to binarize output. Mask ranges from 0-1. Sharpening values are not bounded
    :param binary_mask: The path containing an mrc file with the matrix or a numpy array for the mask. If None, no mask will be used (default)
    :param noise_stats: (mean_noise, standardDeviation_noise). The statistics of the noise used to normalize the input. If non, they will be automatically
                                                              estimated
    :param voxel_size: in A/voxel. If none, it will be read from vol_fname instead

    :param apply_postprocess_cleaning: If >0, apply leaning of isolated conected components of size <apply_postprocess_cleaning. This option reduces noise
                                       but it may delete somo small intensisty parts of the protein. -1 will skip this option
                                       True recommended for fully automatic workflows.
    :param fname_prod: fnameIn where the product of the prediction and the input will be saved
    :param morph_dilation: Size of the structuring element for morphological dilation.

    :return: prediction: a 3D numpy array representing the post-processed volume
    '''
    assert fname_out is None or not os.path.isfile(fname_out), "Error, output fnameIn already exists"
    assert apply_postprocess_cleaning<1, "Error, required 0<apply_postprocess_cleaning<1  or apply_postprocess_cleaning=-1. Provided %s"%(apply_postprocess_cleaning)
    assert  not (noise_stats is not None and binary_mask is not None), "Error, only one of the following options can be provided: noise_stats, binary_mask "
    inputNormFun, binary_mask = self._getNormalizationFunction(binary_mask, noise_stats)
    vol, boxSize= loadVolIfFnameOrIgnoreIfMatrix(vol_fname_or_matrix, normalize=inputNormFun)

    if boxSize is None:
      assert voxel_size is not None, "Error, if array provided as input, voxel_size should also be provided"
      boxSize= voxel_size

    if voxel_size is not None :
      if boxSize is not None and boxSize!=voxel_size:
        print("Warning: IGNORING voxel size contained in header (%.3f) and using %.3f for %s" % (
        boxSize, voxel_size, vol_fname_or_matrix if isinstance(vol_fname_or_matrix, str)  else "data"))
      boxSize= voxel_size

    originalShape= vol.shape
    assert binary_mask is None or binary_mask.shape== originalShape, "Error, the size of the input volume and the mask does not agree %s -- %s"%(originalShape, binary_mask.shape)

    if RESIZE_VOL_TO:
      newShape = getNewShapeForResize(vol, boxSize, RESIZE_VOL_TO)
      vol = resizeVol(vol, newShape)
      if binary_mask is not None:
        binary_mask= resizeVol(binary_mask, newShape)

    vol, paddingValues= self._padToDivisibleSize(vol)

    print( "DONE!. Shape at %.2f A/voxel after padding-> "%RESIZE_VOL_TO, vol.shape)
    sys.stdout.flush()
    processVol= np.zeros(vol.shape)
    weights= np.ones(vol.shape)

    batch_x=[]
    coords_list=[]
    n_cubes=0

    print("Neural net inference")
    for cube, coords in self._chunkInputVolForPrediction(vol, chunk_size= self.netInputSize, binary_mask= binary_mask):
      batch_x.append( cube )
      coords_list.append( ( coords) )
      n_cubes+=1
      if n_cubes==self.batch_size:
        batch_x= np.stack(batch_x)
        batch_y_pred= self.model.predict_on_batch(np.expand_dims(batch_x, axis=-1))

        self._updateMask(coords_list, batch_y_pred, processVol, weights)
        batch_x, coords_list= [], []
        n_cubes=0
    if n_cubes>0:
      batch_x = np.stack(batch_x)[:n_cubes,...]
      coords_list = coords_list[:n_cubes]

      batch_y_pred = self.model.predict_on_batch(np.expand_dims(batch_x, axis=-1))
      self._updateMask(coords_list, batch_y_pred, processVol, weights)

    processVol= processVol/weights

    processVol= self._unPad(processVol, paddingValues)

    if apply_postprocess_cleaning>0:
      processVol = removeSmallCCs(processVol, min_size_to_preserve=apply_postprocess_cleaning)

    if morph_dilation is not None:
      processVol= morphologicalDilation(processVol, morph_dilation)

    if threshold is not None:
      processVol= np.where(processVol>threshold, 1, 0)

    if fname_prod:
      prod_output= processVol*vol

    if RESIZE_VOL_TO:
      processVol = resizeVol(processVol, originalShape)
      if fname_prod:
        prod_output = resizeVol(prod_output, originalShape)

    if fname_out is not None:
      fname_out=os.path.expanduser(fname_out)
      saveVol(fname_out, processVol, samplingRate=boxSize)

    if fname_prod is not None:
      saveVol(fname_prod, prod_output, samplingRate=boxSize)
      return processVol, prod_output

    return processVol

  def close(self):
    self.model= None
    import keras.backend as K
    K.clear_session()
    gc.collect()

  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.close()


def resolveHalfMapsOrInputMap(inputVol, halfMap2):
  if halfMap2 is None:
    inputVolOrFname= inputVol
    boxSize=None
  else:
    half1Vol, boxSize= loadVolIfFnameOrIgnoreIfMatrix(inputVol, normalize=None)
    half2Vol, __= loadVolIfFnameOrIgnoreIfMatrix(halfMap2, normalize=None)
    inputVolOrFname= 0.5*(half1Vol+half2Vol)
  return inputVolOrFname, boxSize

if __name__=="__main__":
  from ..configManager import ConfigManager
  from ..utils.genericParser import parseProcessingType

  additonalArgs= [("-i", "--input", {
      "type":str,
      "nargs":None,
      "required": True,
      "help": "input volume to process or half map number 1"} ),


      ("-o", "--output", {
         "type":str,
         "nargs": None,
         "required": False,
         "help": "output name for processed volume"}),

      ("--noise_stats", {
          "type": float,
          "nargs": 2, "metavar": ("NOISE_MEAN", "NOISE_STD"),
          "required": False,
          "help": "(Optional) Normalization mode 1: The statisitcs of the noise to normalize (mean and standard deviation) the input. Preferred over binaryMask but ignored if "
                  "binaryMask provided. If not --noise_stats nor --binaryMask provided, nomralization params will be automatically estimated, although estimation may fail or be "
                  "less accurate"}),

      ("-m", "--binaryMask", {
         "type": str,
         "nargs": None,
         "required": False,
         "help": "(Optional) Normalization mode 2: A binaryMask (1 protein, 0 no protein) used to normalize the input. If no normalization mode "
                 "provided, automatic option normalization will be carried out"}),

      ("-p", "--productionModel", {
        "action": 'store_true',
        "default": False,
        "help":"look into productions directory instead currently trained"
      }),

      ("-c", "--checkpoint", {
         "type": str,
         "required": False, "nargs": None,
         "default": None, "metavar": "PATH_TO_CHECKPOINT",
          "help": "directory where deep learning model is located"
       }),


      ( "-s", "--sampling_rate", {
         "type": float,
         "required": False,
         "default": None,
         "help": "sampling rate of the input model. Optional. If not provided, the sampling rate will be read from mrc file header"

       }),

      ("-i2", "--halfMap2", {
      "type": str,
      "nargs": None,
      "required": False,
      "default": None,
      "help": " (optional) input volume half map 2 to process"}),

      ("--cleaningStrengh", {
        "type": float,
        "default": -1,
        "required": False,
        "help": "Max size of connected components to remove 0<s<1 or -1 to deactivate. Default: %(default)s"
      }),

      ("-g", "--gpuId", {
         "type":str,
         "nargs": None,
         "required": False,
         "default": "0",
         "help": "The gpu(s) where the program will be executed. If more that 1, comma seppared. E.g -g 1,2,3 . Default: %(default)s"

      }),

        ("-b", "--batch_size", {
          "type": int,
          "nargs": None,
          "required": False ,
          "default": BATCH_SIZE,
          "help": "Number of cubes to process simultaneously. Lower it if CUDA out of memory error happens. Default: %(default)s"
        }),

   ]

  processingType, args = parseProcessingType("apply neural network to do volume postprocessing", additonalArgs, skypFileOfIds=True)

  if args.checkpoint is None:
    checkpoint_fname= ConfigManager.getModelPath(args.productionModel, args.binaryMask is not None)
  else:
    checkpoint_fname= os.path.expanduser(args.checkpoint)

  inputVolOrFname, boxSize = resolveHalfMapsOrInputMap(args.input, args.halfMap2)

  if args.sampling_rate is not None:
    boxSize= args.sampling_rate

  predictor= AutoProcessVol(checkpoint_fname, gpuIds= args.gpuId, batch_size= args.batch_size)
  predictor.predict(inputVolOrFname, args.output, binary_mask=args.binaryMask, noise_stats=args.noise_stats,
                    voxel_size=boxSize, apply_postprocess_cleaning=args.cleaningStrengh)


