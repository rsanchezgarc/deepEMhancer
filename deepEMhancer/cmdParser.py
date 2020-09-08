import sys, os

from tqdm import tqdm

from .utils.pathUtils import tryToRemove
from .config import DEFAULT_MODEL_DIR, DOWNLOAD_MODEL_URL, MODEL_DOWNLOAD_EXPECTED_SIZE
from .applyProcessVol.cmdParserOptionsDeepEMHancer import processVolOptions

def parseArgs():
  import argparse

  example_text = '''examples:

  + Download deep learning models
deepemhancer --download

  + Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using default  deep model tightTarget
deepemhancer  -i path/to/inputVol.mrc -o  path/to/outputVol.mrc

  + Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using high resolution deep model
deepemhancer -p highRes -i path/to/inputVol.mrc -o  path/to/outputVol.mrc

  + Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using a deep learning model located in path/to/deep/learningModel
deepemhancer -c path/to/deep/learningModel -i path/to/inputVol.mrc -o  path/to/outputVol.mrc

  + Post-process input map path/to/inputVol.mrc and save it at path/to/outputVol.mrc using high resolution  deep model and providing normalization information (mean
    and standard deviation of the noise)
deepemhancer -p highRes -i path/to/inputVol.mrc -o  path/to/outputVol.mrc --noiseStats 0.12 0.03
'''

  parser = argparse.ArgumentParser(
    description='DeepEMHancer. Deep post-processing of cryo-EM maps. https://github.com/rsanchezgarc/deepEMhancer', add_help=False,
    epilog=example_text, formatter_class=argparse.RawDescriptionHelpFormatter)

  group = parser.add_argument_group("Main arguments")
  for kwarg in processVolOptions:
    if kwarg[0].startswith("parser_group"):
      group=parser.add_argument_group(kwarg[1] )
    else:
      group.add_argument(*kwarg[:-1], **kwarg[-1])

  parser.add_argument("-h", "--help", action="help", help="show this help message and exit")

  class _DownloadModel(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, const=None, default=None, required=False, help=None, metavar=None):
      super(_DownloadModel, self).__init__(option_strings=option_strings, dest=dest, default=default,
                                           nargs=nargs, const=const, required=required, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
      import requests
      from zipfile import ZipFile

      if values is None or len(values) == 0:
        downloadPath = os.path.split(DEFAULT_MODEL_DIR)[0]
      else:
        downloadPath = os.path.abspath(os.path.expanduser(values))
      if not os.path.exists(downloadPath):
        os.makedirs(downloadPath)
      print("DOWNLAODING MODELs from %s to %s" % (DOWNLOAD_MODEL_URL, downloadPath))
      print("It may take a while...")


      r = requests.get(DOWNLOAD_MODEL_URL, stream=True)
      if r.status_code != 200:
        raise Exception("It was not possible to download the model")

      totalSize=0
      chunk_size=4096
      progressBar = tqdm(total=MODEL_DOWNLOAD_EXPECTED_SIZE//chunk_size+1)


      with open(downloadPath+".zip", 'wb') as fd:
        for i, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
          fd.write(chunk)
          totalSize+=chunk_size
          progressBar.update()
          if i%1000==0:
            progressBar.refresh()
      progressBar.close()

      print("Total size downloaded: %d.\nUnzipping..."%totalSize)

      with ZipFile(downloadPath+".zip") as zfile:
        zfile.extractall(downloadPath)
      print("DONE!!")
      tryToRemove(downloadPath+".zip")
      parser.exit()


  parser.add_argument('--download', nargs="?", action=_DownloadModel, metavar="DOWNLOAD_DEST",
                      help='download default DeepEMhancer models. ' +
                           'They will be saved at %s if no path provided' % (DEFAULT_MODEL_DIR))

  args = vars(parser.parse_args())
  deepLearningModelPath = args["deepLearningModelPath"]

  if deepLearningModelPath is None:
    if not os.path.exists(DEFAULT_MODEL_DIR):
      os.makedirs(DEFAULT_MODEL_DIR)
    deepLearningModelPath = DEFAULT_MODEL_DIR
  args["deepLearningModelPath"] = deepLearningModelPath

  if not os.path.isfile(deepLearningModelPath) and not os.path.isfile( os.path.join(deepLearningModelPath, "deepEMhancer_tightTarget.hd5") ):
    print(("Deep learning models not found at %s. Downloading default models with --download or " +
           "indicate its location with --deepLearningModelPath.") % DEFAULT_MODEL_DIR)
    sys.exit(1)

  if "-1" in args["gpuIds"]:
    args["gpuIds"] = None
  if "download" in args:
    del args["download"]

  return args


if __name__=="__main__":
  print("Parser")
  parseArgs()
