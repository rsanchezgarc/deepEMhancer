import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from .applyProcessVol.processVol import AutoProcessVol, resolveHalfMapsOrInputMap

from .config import DEFAULT_MODEL_DIR


def main(inputMap, outputMap, processingType, halfMap2=None, samplingRate=None, noiseStats=None, binaryMask=None,
         deepLearningModelPath=None, cleaningStrengh=-1, batch_size=None, gpuIds="0"):
  '''

  :param inputMap: The path containing an mrc file with the input map or a numpy array. A half map can also be provided
  :param outputMap: The path where the post-processed volumne will be saved. If not, the post-processed map will not be saved to disk
  :param processingType: The type of post-processing to apply to the maps. Can be any of :"wideTarget", "tightTarget","highRes"
  :param halfMap2: The path containing an mrc file with the half map 2 or a numpy array. Set to None if no half maps used
  :param samplingRate: in A/voxel. If None, it will be read from vol_fname instead

  NORMALIZATION:
    Normalization of input mask is crucial. If no option selected, it will try to automatically normalize de map.

  :param noiseStats=(noiseMean, noiseStd): Normalization mode 1: The statisitcs of the noise to normalize (mean and standard deviation) the input. Preferred over binaryMask but ignored if
               binaryMask provided. If not --noiseStats nor --binaryMask provided, nomralization params will be automatically estimated, although estimation may fail or be less accurate
  :param binaryMask: Normalization mode 2: A path to a binaryMask (1 protein, 0 no protein) used to normalize the input. If no normalization mode
                     provided, automatic normalization will be carried out. Supresses --precomputedModel option to use a tailored model.

  :param deepLearningModelPath: The directory where deep learning models are located or a path to hd5 file containing the model. If None, they will be loaded from
                                .config.DEFAULT_MODEL_DIR
  :param cleaningStrengh: Post-processing step to remove small connected components (Hide dust). Max relative size of connected components to remove 0<s<1 or -1 to deactivate.
  :param batch_size: Batch size used to feed the GPUs
  :param gpuIds: Comma separated gpu Ids

  :return: prediction: a 3D numpy array
  '''

  import tensorflow as tf

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  assert os.path.isfile(inputMap), "Error: input file %s not found" % inputMap
  assert outputMap.endswith("mrc") or outputMap.endswith("map"), "Error: %s output name is not in mrc format. End it with .mrc" % outputMap

  checkpoint_fname=None
  if deepLearningModelPath is None:
    deepLearningModelDir= os.path.expanduser(DEFAULT_MODEL_DIR)
  else:
    if os.path.isfile(deepLearningModelPath):
      checkpoint_fname=deepLearningModelPath
    else:
      deepLearningModelDir= deepLearningModelPath

  if checkpoint_fname is not None:
    assert processingType=="tightTarget", "Error, -p option should not be provided if --deepLearningModelPath points to an hd5 file"
  if binaryMask is not None:
    assert processingType=="tightTarget", "Error, if binary mask provided, -p option should not be provided "
    binaryMask= os.path.expanduser(binaryMask)
    checkpoint_fname= checkpoint_fname if checkpoint_fname is not None else os.path.join(deepLearningModelDir, "deepEMhancer_masked.hd5")
  else:
    checkpoint_fname= checkpoint_fname if checkpoint_fname is not None else os.path.join(deepLearningModelDir, "deepEMhancer_" + processingType + ".hd5")

  inputVolOrFname, boxSize = resolveHalfMapsOrInputMap(inputMap, halfMap2)

  if samplingRate is not None:
    boxSize= samplingRate

  predictor= AutoProcessVol(checkpoint_fname, gpuIds= gpuIds, batch_size= batch_size)

  predVol= predictor.predict(inputVolOrFname, outputMap, binary_mask=binaryMask, noise_stats=noiseStats,
                    voxel_size=boxSize, apply_postprocess_cleaning=cleaningStrengh)

  return predVol


def commanLineFun():
  from .cmdParser import parseArgs
  main( ** parseArgs() )

if __name__=="__main__":
  commanLineFun()
