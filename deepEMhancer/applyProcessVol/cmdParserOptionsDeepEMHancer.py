import deepEMhancer
from ..config import BATCH_SIZE


processVolOptions= [

  ("parser_group", "Main options"),

    ("-i", "--inputMap", {
      "type": str,
      "nargs": None,
      "required": True,
      "help": "Input map to process or half-map 1. Prefer an unmasked, unsharpened map obtained directly from refinement. "
              "When providing half-map 1, also provide half-map 2 with -i2"}),

  ("-o", "--outputMap", {
      "type": str,
      "nargs": None,
      "required": True,
      "help": "Output fname where post-processed map will be saved"}),

    ("-p", "--processingType", {
      "choices": ['wideTarget', 'tightTarget', 'highRes'],
      "default": 'tightTarget',
      "help": "Select the deep-learning model. wideTarget generally produces less sharp results than tightTarget. "
              "highRes is recommended only for overall FSC resolution better than 4 A. Keep the default when using --binaryMask"}),

    ("-i2", "--halfMap2", {
      "type": str,
      "nargs": None,
      "required": False,
      "default": None,
      "help": "(Optional) Half-map 2 to process"}),

    ("-s", "--samplingRate", {
      "type": float,
      "required": False,
      "default": None,
      "help": "(Optional) Sampling rate (A/voxel) of the input map. If not provided, the sampling rate will be read from mrc file header"}),


     ("parser_group", "Normalization options (auto normalization is applied if no option selected)"),

     ("--noiseStats", {
       "type": float,
       "nargs": 2, "metavar": ("NOISE_MEAN", "NOISE_STD"),
       "required": False,
       "help": "(Optional) Noise mean and standard deviation used to normalize the input. If neither --noiseStats nor "
               "--binaryMask is provided, these values are estimated automatically"}),

     ("-m", "--binaryMask", {
       "type": str,
       "nargs": None,
       "required": False,
       "help": "(Optional) Binary mask (1 for protein, 0 for background) used to normalize the input. This selects the "
               "model designed for masked inputs"}),

     ("parser_group", "Alternative options"),

     ( "--deepLearningModelPath", {
       "type": str,
       "required": False, "nargs": None,
       "default": None, "metavar": "PATH_TO_MODELS_DIR",
       "help": "(Optional) Directory containing the DeepEMhancer models, or a path to a specific .hd5 model file"
     }),


     ("--cleaningStrengh", {
       "type": float,
       "default": -1,
       "required": False,
       "help": "(Optional) Post-processing step to remove small connected components (hide dust). Max relative size of connected components to remove 0<s<1 or -1 to deactivate. Default: %(default)s"
     }),


     ("parser_group", "Computing devices options"),

     ("-g", "--gpuIds", {
       "type": str,
      "nargs": None,
      "required": False,
      "default": "0",
      "help": "Comma-separated GPU IDs, for example -g 0,1. Set to -1 for CPU-only inference. Default: %(default)s"

     }),

     ("-b", "--batch_size", {
       "type": int,
      "nargs": None,
      "required": False,
      "default": BATCH_SIZE,
      "help": "Number of cubes processed simultaneously. Reduce it after a GPU out-of-memory error. Default: %(default)s"
     }),

    ("--version", {
        "action": "version",
        "version": deepEMhancer.__version__,
    }),
]
