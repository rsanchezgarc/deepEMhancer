
processVolOptions= [

  ("parser_group", "Main options"),

    ("-i", "--inputMap", {
      "type": str,
      "nargs": None,
      "required": True,
      "help": "input map to process or half map number 1. This map should be unmasked and not sharpened"}),

    ("-o", "--outputMap", {
      "type": str,
      "nargs": None,
      "required": True,
      "help": "output fname where post-processed map will be saved"}),

    ("-p", "--processingType", {
      "choices": ['wideTarget', 'tightTarget', 'highRes'],
      "default": 'tightTarget',
      "help": "select the deep learning model you want to use. WideTarget will produce less sharp results than tightTarget. HighRes is only recommended for overal FSC resolution < 4 A"}),

    ("-i2", "--halfMap2", {
      "type": str,
      "nargs": None,
      "required": False,
      "default": None,
      "help": "(Optional) input half map 2 to process"}),

    ("-s", "--samplingRate", {
      "type": float,
      "required": False,
      "default": None,
      "help": "(Optional) sampling rate (A/voxel) of the input map. If not provided, the sampling rate will be read from mrc file header"}),


     ("parser_group", "Normalization options (auto normalization if non provided)"),

     ("--noiseStats", {
       "type": float,
       "nargs": 2, "metavar": ("NOISE_MEAN", "NOISE_STD"),
       "required": False,
       "help": "(Optional) Normalization mode 1: The statisitcs of the noise to normalize (mean and standard deviation) the input. Preferred over binaryMask but ignored if "
               "binaryMask provided. If not --noiseStats nor --binaryMask provided, nomralization params will be automatically estimated, although estimation may fail or be "
               "less accurate"}),

     ("-m", "--binaryMask", {
       "type": str,
       "nargs": None,
       "required": False,
       "help": "(Optional) Normalization mode 2: A binaryMask (1 protein, 0 no protein) used to normalize the input. If no normalization mode "
               "provided, automatic normalization will be carried out. Supresses --precomputedModel option"}),

     ("parser_group", "Alternative options"),

     ( "--deepLearningModelDir", {
       "type": str,
       "required": False, "nargs": None,
       "default": None, "metavar": "PATH_TO_MODELS_DIR",
       "help": "Directory where a non default deep learning model is located. Supressess --precomputedModel"
     }),


     ("--cleaningStrengh", {
       "type": float,
       "default": -1,
       "required": False,
       "help": "Post-processing step to remove small connected components (hide dust). Max relative size of connected components to remove 0<s<1 or -1 to deactivate. Default: %(default)s"
     }),


     ("parser_group", "Computing devices options"),

     ("-g", "--gpuIds", {
       "type": str,
       "nargs": None,
       "required": False,
       "default": "0",
       "help": "The gpu(s) where the program will be executed. If more that 1, comma seppared. E.g -g 1,2,3. Set to -1 to use only cpu (very slow). Default: %(default)s"

     }),

     ("-b", "--batch_size", {
       "type": int,
       "nargs": None,
       "required": False,
       "default": None,
       "help": "Number of cubes to process simultaneously. Lower it if CUDA Out Of Memory error happens and increase it if low GPU performance observed"
     }),


]